"""Pi0 with True MoE Action Heads (Navigation + Manipulation).

This module implements a true Mixture of Experts (MoE) approach using separate
expert networks instead of modifying the Gemma transformer layers. This makes
fine-tuning easier and more modular.

TRUE MoE ARCHITECTURE:
1. Router: Computes gating weights from action tokens (2-way classification)
2. Navigation Expert: Extracts navigation-specific features (hidden_dim)
3. Manipulation Expert: Extracts manipulation-specific features (hidden_dim)
4. Gating: Router weights blend expert features (soft routing) or select one (hard routing)
5. Final Projection: Maps blended features to full action space (action_dim=32)

This is TRUE MoE because:
- Router computes weights BEFORE expert outputs are combined
- Expert features are weighted/blended based on router weights
- Final projection happens AFTER gating
- Unlike previous implementation, experts don't predict actions directly
"""

import dataclasses
import logging
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import pi0, pi0_config, model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Pi0DualHeadConfig(pi0_config.Pi0Config):
    """Pi0 configuration with TRUE MoE action heads.

    Attributes:
        nav_action_dim: NOT USED in true MoE (kept for compatibility).
            In true MoE, experts extract features, not action dimensions.
        manip_action_dim: NOT USED in true MoE (kept for compatibility).
            In true MoE, experts extract features, not action dimensions.
        router_loss_coef: Weight for router classification loss.
        freeze_router_after_steps: Freeze router after N steps (for hybrid training).
            Set to None to never freeze, or a step number to freeze after that step.
        use_soft_routing: If True, blend experts with router weights (soft MoE).
            If False, use hard selection (select one expert's features).
    """
    nav_action_dim: int = 3  # UNUSED: kept for backward compatibility
    manip_action_dim: int = 20  # UNUSED: kept for backward compatibility
    router_loss_coef: float = 0.1  # Weight for router loss
    freeze_router_after_steps: int | None = None  # Freeze router after N steps
    use_soft_routing: bool = True  # True MoE: blend experts with router weights

    def create(self, rng: at.KeyArrayLike) -> "Pi0DualHead":
        """Create a Pi0DualHead model instance.

        Args:
            rng: Random number generator key.

        Returns:
            Pi0DualHead model instance.
        """
        return Pi0DualHead(self, rngs=nnx.Rngs(rng))


class Pi0DualHead(pi0.Pi0):
    """Pi0 model with TRUE MoE for navigation and manipulation.

    This model extends Pi0 by replacing the single action output projection with
    a true Mixture of Experts architecture:

    1. Task Router: Computes gating weights (2-way: nav vs manip)
    2. Navigation Expert: Extracts navigation-specific features
    3. Manipulation Expert: Extracts manipulation-specific features
    4. Gating Layer: Blends expert features using router weights (soft routing)
    5. Output Projection: Maps blended features to full action space

    The router can be trained with supervision then frozen (hybrid training).
    """

    def __init__(self, config: Pi0DualHeadConfig, rngs: nnx.Rngs):
        """Initialize Pi0DualHead model.

        Args:
            config: Pi0DualHead configuration.
            rngs: Random number generator for initialization.
        """
        # Initialize base Pi0 model
        super().__init__(config, rngs)

        # Store dual-head specific config
        self.nav_action_dim = config.nav_action_dim
        self.manip_action_dim = config.manip_action_dim
        self.router_loss_coef = config.router_loss_coef
        self.freeze_router_after_steps = config.freeze_router_after_steps
        self.use_soft_routing = config.use_soft_routing

        # Get hidden dimension from the LLM
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        hidden_dim = action_expert_config.width

        # Replace the single action_out_proj with TRUE MoE components:

        # 1. Task Router: 2-layer MLP for learning when to route
        # More capacity than single linear layer
        # Output shape: (batch, 2) - logits for nav vs manip
        self.task_router = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs),
            nnx.swish,
            nnx.Linear(hidden_dim // 2, 2, rngs=rngs),
        )

        # 2. Navigation Expert: 3-layer MLP for navigation-specific features
        # Deeper network for better feature extraction
        # Output shape: (batch, action_horizon, hidden_dim)
        self.nav_expert = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
        )

        # 3. Manipulation Expert: 3-layer MLP for manipulation-specific features
        # Deeper network for better feature extraction
        # Output shape: (batch, action_horizon, hidden_dim)
        self.manip_expert = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
        )

        # 4. Final output projection: Maps combined expert features to actions
        # Output shape: (batch, action_horizon, action_dim)
        self.action_out_proj = nnx.Linear(hidden_dim, config.action_dim, rngs=rngs)

        # Track training step for hybrid router freezing
        # Note: These are not nnx.Variable to avoid checkpoint issues
        # They will be tracked via metrics instead
        self._current_step = 0
        self._router_frozen = False

    def _get_action_predictions(
        self, suffix_out: at.Float[at.Array, "b s hidden"], use_soft_routing: bool = None
    ) -> tuple[
        at.Float[at.Array, "b 2"],  # router_logits
        at.Float[at.Array, "b ah ad"],  # final_action (gated)
        at.Float[at.Array, "b ah hidden"],  # nav_features (for metrics)
        at.Float[at.Array, "b ah hidden"],  # manip_features (for metrics)
    ]:
        """Get predictions with true MoE-style gating.

        TRUE MoE ARCHITECTURE:
        1. Router computes gating weights from first action token
        2. Both experts extract features (hidden_dim) from action tokens
        3. Router weights blend expert features (weighted combination)
        4. Final projection maps blended features to full action space (action_dim)

        Args:
            suffix_out: Output from LLM suffix (state + action tokens).
                Shape: (batch, sequence_length, hidden_dim)
            use_soft_routing: Override config setting. If None, uses self.use_soft_routing.

        Returns:
            router_logits: Task classification logits (batch, 2).
            final_action: Full action prediction (batch, action_horizon, action_dim).
            nav_features: Navigation expert features (batch, action_horizon, hidden_dim).
            manip_features: Manipulation expert features (batch, action_horizon, hidden_dim).
        """
        # Extract the final action token representations
        # Shape: (batch, action_horizon, hidden_dim)
        action_tokens = suffix_out[:, -self.action_horizon :]

        # 1. Router computes gating weights (STEP 1: ROUTING)
        # Use the first action token for task classification
        # Shape: (batch, hidden_dim) -> (batch, 2)
        router_logits = self.task_router(action_tokens[:, 0, :])

        # 2. Experts extract specialized features (STEP 2: EXPERT PROCESSING)
        # Shape: (batch, action_horizon, hidden_dim)
        nav_features = self.nav_expert(action_tokens)
        manip_features = self.manip_expert(action_tokens)

        # 3. TRUE MoE GATING: Router gates expert features (STEP 3: GATING)
        use_soft = use_soft_routing if use_soft_routing is not None else self.use_soft_routing

        if use_soft:
            # SOFT ROUTING: Blend expert features with router weights
            # Compute gating weights (softmax over experts)
            router_weights = jax.nn.softmax(router_logits, axis=-1)  # (batch, 2)
            # router_weights[:, 0] = weight for manipulation expert
            # router_weights[:, 1] = weight for navigation expert

            # Reshape weights for broadcasting: (batch, 1, 1)
            w_nav = router_weights[:, 1:2, None]      # (batch, 1, 1)
            w_manip = router_weights[:, 0:1, None]    # (batch, 1, 1)

            # Weighted combination of expert features (TRUE MoE GATING)
            # Shape: (batch, action_horizon, hidden_dim)
            gated_features = w_nav * nav_features + w_manip * manip_features

        else:
            # HARD ROUTING: Select one expert's features
            expert_idx = jnp.argmax(router_logits, axis=-1)  # (batch,)
            is_nav = (expert_idx == 1)[:, None, None]  # (batch, 1, 1)
            gated_features = jnp.where(is_nav, nav_features, manip_features)

        # 4. Final projection to action space (STEP 4: OUTPUT)
        # Shape: (batch, action_horizon, action_dim)
        final_action = self.action_out_proj(gated_features)

        return router_logits, final_action, nav_features, manip_features

    def compute_loss_with_routing(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        movement_labels: at.Int[at.Array, "b"] | None = None,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, ""], dict]:
        """Compute loss with dual heads and task routing.

        Args:
            rng: Random number generator key.
            observation: Observation data.
            actions: Ground truth actions (batch, action_horizon, action_dim).
            movement_labels: Movement labels for routing (batch,).
                0 = manipulation (base stationary), 1 = navigation (base moving)
            train: Whether in training mode.

        Returns:
            total_loss: Scalar total loss.
            metrics: Dictionary with loss components and routing statistics.
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        batch_size = actions.shape[0]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Forward pass through prefix + suffix (same as Pi0)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )

        # Get predictions with TRUE MoE gating
        router_logits, final_action, nav_features, manip_features = self._get_action_predictions(suffix_out)

        # --- Compute Action Loss ---
        # With true MoE, the router has already gated the experts in the forward pass
        # So we compute loss on the final gated action output
        # Shape: (batch, action_horizon, action_dim)
        loss_action = jnp.mean(jnp.square(final_action - u_t))

        # For monitoring: compute what each expert would have predicted
        # Project expert features to action space
        nav_action_monitoring = self.action_out_proj(nav_features)
        manip_action_monitoring = self.action_out_proj(manip_features)

        # Compute per-expert losses for metrics only
        loss_nav_monitoring = jnp.mean(jnp.square(nav_action_monitoring - u_t))
        loss_manip_monitoring = jnp.mean(jnp.square(manip_action_monitoring - u_t))

        # --- Router Loss ---
        if movement_labels is not None:
            # Check if router should be frozen
            if self.freeze_router_after_steps is not None and self._current_step >= self.freeze_router_after_steps:
                if not self._router_frozen:
                    logger.info(
                        f"Freezing router at step {self._current_step} "
                        f"(threshold: {self.freeze_router_after_steps})"
                    )
                    self._router_frozen = True

            if self._router_frozen:
                # Router is frozen, no gradient update
                router_loss = jnp.array(0.0)
                # Stop gradient to prevent backprop through router
                router_logits = jax.lax.stop_gradient(router_logits)
            else:
                # Router is trainable, compute cross-entropy loss
                # Use softmax cross-entropy with integer labels
                log_probs = jax.nn.log_softmax(router_logits, axis=-1)
                router_loss = -jnp.mean(
                    jnp.take_along_axis(log_probs, movement_labels[:, None], axis=-1)
                )

            # Compute router accuracy
            router_preds = jnp.argmax(router_logits, axis=-1)
            router_accuracy = jnp.mean(router_preds == movement_labels)

        else:
            # No labels provided, no router supervision
            router_loss = jnp.array(0.0)
            router_accuracy = jnp.array(0.0)

        # --- Total Loss ---
        total_loss = loss_action + self.router_loss_coef * router_loss

        # Increment step counter
        if train:
            self._current_step += 1

        # --- Metrics ---
        metrics = {
            "loss": total_loss,
            "loss_action": loss_action,
            "loss_nav_expert": loss_nav_monitoring,  # What nav expert would predict
            "loss_manip_expert": loss_manip_monitoring,  # What manip expert would predict
            "loss_router": router_loss,
            "router_accuracy": router_accuracy,
            "router_frozen": float(self._router_frozen),
            "current_step": float(self._current_step),
        }

        if movement_labels is not None:
            # Add label distribution statistics
            nav_ratio = jnp.mean(movement_labels == 1)
            metrics["nav_task_ratio"] = nav_ratio
            metrics["manip_task_ratio"] = 1.0 - nav_ratio

        return total_loss, metrics

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, ""]:
        """Standard compute_loss interface (without routing outputs).

        For compatibility with base Pi0 interface. If you want routing metrics,
        use compute_loss_with_routing() instead.
        """
        loss, _ = self.compute_loss_with_routing(rng, observation, actions, movement_labels=None, train=train)
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        movement_labels: at.Int[at.Array, "b"] | None = None,
    ) -> _model.Actions:
        """Sample actions using the dual-head model.

        At inference time, we combine predictions from both heads.
        Optionally, movement_labels can be used to select only one head.

        Args:
            rng: Random number generator key.
            observation: Observation data.
            num_steps: Number of diffusion steps.
            noise: Initial noise (optional).
            movement_labels: Movement labels for routing (optional).
                If provided, only use the specified head's predictions.

        Returns:
            Sampled actions (batch, action_horizon, action_dim).
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # First fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_expanded = jnp.broadcast_to(
                prefix_mask[:, None, :], (batch_size, suffix_tokens.shape[1], prefix_mask.shape[1])
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask_expanded, suffix_attn_mask], axis=-1)

            positions_suffix = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_suffix,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None

            # Get predictions with TRUE MoE gating
            # The router has already gated the experts, so final_action is ready to use
            router_logits, final_action, nav_features, manip_features = self._get_action_predictions(suffix_out)

            # Use the gated action prediction directly
            v_t = final_action

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

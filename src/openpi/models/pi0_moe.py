"""Pi0 configuration with MoE support.

This module extends Pi0Config to include MoE parameters for MoE implementation.
Implements Pi0MoE which uses Mixture of Experts for task-specific specialization.
"""

import dataclasses
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import pi0, pi0_config, moe, model as _model, gemma_moe
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
import openpi.shared.array_typing as at


@dataclasses.dataclass(frozen=True, kw_only=True)
class Pi0MoEConfig(pi0_config.Pi0Config):
    """Pi0 configuration with MoE settings.

    Extends Pi0Config with MoE-specific parameters for future MoE implementation.
    Currently creates standard Pi0 models while movement label data is collected.

    Attributes:
        moe_config: Configuration for MoE layers (routing, num experts, losses).
        moe_layers: Which transformer layers should use MoE ("all" or list of indices).
    """
    # MoE configuration
    moe_config: moe.MoEConfig = dataclasses.field(default_factory=lambda: moe.MoEConfig(
        num_experts=2,
        router_type="supervised",
        load_balancing_loss_coef=0.01,
        router_z_loss_coef=0.001,
    ))

    # Which Gemma layers to apply MoE to
    moe_layers: list[int] | str = "all"  # "all" or list of layer indices

    def create(self, rng: at.KeyArrayLike) -> "Pi0MoE":
        """Create a Pi0MoE model instance.

        Args:
            rng: Random number generator key.

        Returns:
            Pi0MoE model instance with MoE-enabled LLM.
        """
        return Pi0MoE(self, rngs=nnx.Rngs(rng))


class Pi0MoE(pi0.Pi0):
    """Pi0 model with Mixture of Experts in the LLM component.

    Extends Pi0 by replacing the standard Gemma LLM with an MoE-enabled version
    that routes between experts based on movement types (manipulation vs navigation).
    """

    def __init__(self, config: Pi0MoEConfig, rngs: nnx.Rngs):
        """Initialize Pi0MoE model.

        Args:
            config: Pi0MoE configuration with MoE settings.
            rngs: Random number generator for initialization.
        """
        # Initialize base model structure (without calling super().__init__)
        # We need to manually set up the same structure as Pi0 but with MoE LLM
        _model.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)

        self.pi05 = config.pi05

        # Get base Gemma configs
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Convert to MoE configs
        paligemma_moe_config = gemma_moe.convert_to_moe_config(
            paligemma_config,
            moe_config=config.moe_config,
            moe_layers=config.moe_layers,
        )
        action_expert_moe_config = gemma_moe.convert_to_moe_config(
            action_expert_config,
            moe_config=config.moe_config,
            moe_layers=config.moe_layers,
        )

        # Create MoE-enabled LLM
        llm = nnx_bridge.ToNNX(
            gemma_moe.MoEModule(
                configs=[paligemma_moe_config, action_expert_moe_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])

        # Vision encoder (same as Pi0)
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Action projection layers (same as Pi0)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval()
        self.deterministic = True

    def compute_loss_with_moe(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        movement_labels: at.Int[at.Array, "b"] | None,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict]:
        """Compute loss with MoE auxiliary outputs.

        Args:
            rng: Random number generator key.
            observation: Observation data.
            actions: Ground truth actions.
            movement_labels: Movement labels for routing [batch].
                0 = manipulation (base stationary), 1 = navigation (base moving)
            train: Whether in training mode.

        Returns:
            loss: Per-sample action prediction loss [batch, action_horizon].
            moe_aux: Dictionary with MoE auxiliary outputs including:
                - total_moe_loss: Total MoE auxiliary loss
                - total_load_balance_loss: Load balancing loss
                - total_router_z_loss: Router z-loss
                - avg_expert_usage: Average expert usage statistics
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed prefix (vision + language)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # Embed suffix (state + actions)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        # Create attention mask
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # Forward pass through MoE LLM
        (prefix_out, suffix_out), _, moe_aux = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            positions=positions,
            mask=attn_mask,
            adarms_cond=[None, adarms_cond],
            movement_labels=movement_labels,
            deterministic=self.deterministic,
        )

        # Decode actions
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # Compute per-sample loss
        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        return loss, moe_aux

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """Standard compute_loss interface (without MoE outputs).

        For compatibility with base Pi0 interface. If you want MoE auxiliary outputs,
        use compute_loss_with_moe() instead.
        """
        loss, _ = self.compute_loss_with_moe(rng, observation, actions, movement_labels=None, train=train)
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
        """Sample actions using the MoE model.

        Args:
            rng: Random number generator key.
            observation: Observation data.
            num_steps: Number of diffusion steps.
            noise: Initial noise (optional).
            movement_labels: Movement labels for routing (optional).

        Returns:
            Sampled actions.
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

        _, kv_cache, _ = self.PaliGemma.llm(
            [prefix_tokens, None],
            positions=positions,
            mask=prefix_attn_mask,
            movement_labels=movement_labels,
        )

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

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _, _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                positions=positions,
                mask=full_attn_mask,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                movement_labels=movement_labels,
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

"""Pi0 with Auxiliary Task Classification Loss.

This module implements a simpler alternative to MoE using auxiliary loss.
Instead of routing/gating experts, the model learns to predict both:
1. Actions (main task)
2. Task type - navigation vs manipulation (auxiliary task)

ARCHITECTURE:
- Base Pi0 model (frozen vision + LLM)
- Action Head: Predicts full action (32D)
- Task Classifier: Predicts task type (2-way classification)
- Multi-task learning with auxiliary loss

Benefits vs MoE:
- Simpler architecture (no expert gating)
- Model learns task-aware representations
- Auxiliary task helps learn better features
- No complex routing logic needed
"""

import dataclasses
import logging
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import pi0, pi0_config, model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Pi0AuxLossConfig(pi0_config.Pi0Config):
    """Pi0 configuration with auxiliary task classification loss.

    Attributes:
        aux_loss_coef: Weight for auxiliary task classification loss.
            Higher values make the model focus more on learning task representations.
        task_classifier_hidden_dim: Hidden dimension for task classifier MLP.
            If None, uses hidden_dim // 2.
    """
    aux_loss_coef: float = 0.1  # Weight for auxiliary task loss
    task_classifier_hidden_dim: int | None = None  # Hidden dim for classifier

    def create(self, rng: at.KeyArrayLike) -> "Pi0AuxLoss":
        """Create a Pi0AuxLoss model instance.

        Args:
            rng: Random number generator key.

        Returns:
            Pi0AuxLoss model instance.
        """
        return Pi0AuxLoss(self, rngs=nnx.Rngs(rng))


class Pi0AuxLoss(pi0.Pi0):
    """Pi0 model with auxiliary task classification.

    This model extends Pi0 by adding an auxiliary task classifier that predicts
    whether the current task is navigation or manipulation. This helps the model
    learn better task-aware representations through multi-task learning.

    Architecture:
    1. LLM produces action token representations
    2. Action Head: Predicts full action sequence (main task)
    3. Task Classifier: Predicts task type from first action token (auxiliary task)

    Loss:
    total_loss = action_loss + aux_loss_coef * task_classification_loss
    """

    def __init__(self, config: Pi0AuxLossConfig, rngs: nnx.Rngs):
        """Initialize Pi0AuxLoss model.

        Args:
            config: Pi0AuxLoss configuration.
            rngs: Random number generator for initialization.
        """
        # Initialize base Pi0 model
        super().__init__(config, rngs)

        # Store auxiliary loss specific config
        self.aux_loss_coef = config.aux_loss_coef

        # Get hidden dimension from the LLM
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        hidden_dim = action_expert_config.width
        classifier_hidden = config.task_classifier_hidden_dim or (hidden_dim // 2)

        # Task Classifier: 2-layer MLP for task classification
        # Predicts whether task is navigation (1) or manipulation (0)
        # Uses first action token representation
        self.task_classifier = nnx.Sequential(
            nnx.Linear(hidden_dim, classifier_hidden, rngs=rngs),
            nnx.swish,
            nnx.Linear(classifier_hidden, 2, rngs=rngs),  # 2-way classification
        )

        # Note: action_out_proj already exists from Pi0 base class
        # It projects action tokens to action_dim (32)

    def compute_loss_with_aux(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        movement_labels: at.Int[at.Array, "b"] | None = None,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, ""], dict]:
        """Compute loss with auxiliary task classification.

        Args:
            rng: Random number generator key.
            observation: Observation data.
            actions: Ground truth actions (batch, action_horizon, action_dim).
            movement_labels: Movement labels for auxiliary task (batch,).
                0 = manipulation (base stationary), 1 = navigation (base moving)
            train: Whether in training mode.

        Returns:
            total_loss: Scalar total loss.
            metrics: Dictionary with loss components and task classification accuracy.
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

        # Extract action token representations
        action_tokens = suffix_out[:, -self.action_horizon :]  # (batch, action_horizon, hidden_dim)

        # --- Main Task: Action Prediction ---
        # Use standard Pi0 action head
        predicted_actions = self.action_out_proj(action_tokens)  # (batch, action_horizon, action_dim)

        # Compute action loss (flow matching / diffusion loss)
        action_loss = jnp.mean(jnp.square(predicted_actions - u_t))

        # --- Auxiliary Task: Task Classification ---
        if movement_labels is not None:
            # Use first action token for task classification
            # This encourages the model to encode task type in action representations
            task_logits = self.task_classifier(action_tokens[:, 0, :])  # (batch, 2)

            # Compute classification loss
            log_probs = jax.nn.log_softmax(task_logits, axis=-1)
            aux_loss = -jnp.mean(
                jnp.take_along_axis(log_probs, movement_labels[:, None], axis=-1)
            )

            # Compute classification accuracy
            task_preds = jnp.argmax(task_logits, axis=-1)
            task_accuracy = jnp.mean(task_preds == movement_labels)

            # Total loss
            total_loss = action_loss + self.aux_loss_coef * aux_loss

        else:
            # No auxiliary labels provided, use only action loss
            aux_loss = jnp.array(0.0)
            task_accuracy = jnp.array(0.0)
            total_loss = action_loss

        # --- Metrics ---
        metrics = {
            "loss": total_loss,
            "loss_action": action_loss,
            "loss_aux_task": aux_loss,
            "task_accuracy": task_accuracy,
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
        """Standard compute_loss interface (without auxiliary task outputs).

        For compatibility with base Pi0 interface. If you want auxiliary task metrics,
        use compute_loss_with_aux() instead.
        """
        loss, _ = self.compute_loss_with_aux(rng, observation, actions, movement_labels=None, train=train)
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample actions using the model.

        At inference time, we only use the action head. The task classifier
        is not needed for generating actions - it was only used during training
        to help learn better representations.

        Args:
            rng: Random number generator key.
            observation: Observation data.
            num_steps: Number of diffusion steps.
            noise: Initial noise (optional).

        Returns:
            Sampled actions (batch, action_horizon, action_dim).
        """
        # Use standard Pi0 sampling - task classifier not needed at inference
        return super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)

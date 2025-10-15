"""Mixture of Experts (MoE) implementation for robot learning.

This module provides MoE layers that can replace standard FFN layers with
expert routing based on movement types (base/torso vs arm movement).
"""

import dataclasses
from typing import Literal, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
import einops

import openpi.shared.array_typing as at


@dataclasses.dataclass
class MoEConfig:
    """Configuration for Mixture of Experts layer.

    Attributes:
        num_experts: Number of expert networks (typically 2: base/torso expert + arm expert).
        expert_capacity_factor: Capacity factor for expert buffers (for load balancing).
        router_type: Type of router to use:
            - "learned": Learned gating network
            - "supervised": Use movement labels directly
            - "top_k": Top-K routing with learned gating
        top_k: Number of experts to route to per token (for top_k router).
        expert_dropout: Dropout rate for expert outputs.
        load_balancing_loss_coef: Coefficient for auxiliary load balancing loss.
        router_z_loss_coef: Coefficient for router z-loss (encourages lower router logits).
        use_expert_choice: If True, use expert-choice routing instead of token-choice.
    """
    num_experts: int = 2
    expert_capacity_factor: float = 1.25
    router_type: Literal["learned", "supervised", "top_k"] = "supervised"
    top_k: int = 1
    expert_dropout: float = 0.0
    load_balancing_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    use_expert_choice: bool = False


@at.typecheck
class Router(nn.Module):
    """Router for MoE that determines which expert(s) to use for each token.

    The router can operate in different modes:
    - supervised: Uses movement labels directly (no learnable parameters)
    - learned: Learns a gating network to predict expert assignments
    - top_k: Selects top-K experts based on learned router logits
    """

    num_experts: int
    router_type: Literal["learned", "supervised", "top_k"]
    top_k: int = 1
    dtype: str = "bfloat16"

    @nn.compact
    def __call__(
        self,
        x: at.Float[at.Array, "b t d"],
        movement_labels: Optional[at.Int[at.Array, "b"]] = None,
        deterministic: bool = True,
    ) -> tuple[at.Float[at.Array, "b t e"], dict]:
        """Compute routing probabilities.

        Args:
            x: Input tokens [batch, seq_len, features].
            movement_labels: Movement labels [batch] where:
                0 = no movement, 1 = arm only, 2 = base/torso
            deterministic: Whether to use deterministic routing (for eval).

        Returns:
            router_probs: Routing probabilities [batch, seq_len, num_experts]
            router_aux: Dictionary with auxiliary outputs (for loss computation)
        """
        batch_size, seq_len, features = x.shape
        dtype = jnp.dtype(self.dtype)

        if self.router_type == "supervised":
            # Use movement labels directly
            if movement_labels is None:
                raise ValueError("movement_labels required for supervised routing")

            # Map movement labels to expert indices
            # 0 (no movement) -> expert 0 (arm expert, as fallback)
            # 1 (arm only) -> expert 0 (arm expert)
            # 2 (base/torso) -> expert 1 (base/torso expert)
            expert_indices = jnp.where(movement_labels == 2, 1, 0)

            # Create one-hot routing (hard assignment)
            router_probs = jax.nn.one_hot(expert_indices, self.num_experts, dtype=dtype)
            router_probs = router_probs[:, None, :]  # [batch, 1, num_experts]
            router_probs = jnp.broadcast_to(router_probs, (batch_size, seq_len, self.num_experts))

            router_aux = {
                "router_logits": jnp.log(router_probs + 1e-8),  # log probs
                "expert_indices": expert_indices,
            }

        elif self.router_type == "learned":
            # Learned gating network
            # Pool sequence to get single representation per example
            pooled = jnp.mean(x, axis=1)  # [batch, features]

            # Router logits
            router_logits = nn.Dense(
                self.num_experts,
                dtype=dtype,
                kernel_init=nn.initializers.normal(stddev=0.01),
                name="router_dense",
            )(pooled)  # [batch, num_experts]

            # Softmax to get probabilities
            router_probs = jax.nn.softmax(router_logits, axis=-1)

            # Broadcast to all tokens in sequence
            router_probs = router_probs[:, None, :]  # [batch, 1, num_experts]
            router_probs = jnp.broadcast_to(router_probs, (batch_size, seq_len, self.num_experts))

            router_aux = {
                "router_logits": router_logits,
                "router_probs_pooled": router_probs[:, 0, :],  # [batch, num_experts]
            }

        elif self.router_type == "top_k":
            # Top-K routing with learned router
            # Per-token routing
            router_logits = nn.Dense(
                self.num_experts,
                dtype=dtype,
                kernel_init=nn.initializers.normal(stddev=0.01),
                name="router_dense",
            )(x)  # [batch, seq_len, num_experts]

            # Get top-k experts
            top_k_logits, top_k_indices = jax.lax.top_k(router_logits, self.top_k)
            top_k_probs = jax.nn.softmax(top_k_logits, axis=-1)

            # Create sparse routing tensor
            router_probs = jnp.zeros((batch_size, seq_len, self.num_experts), dtype=dtype)

            # Scatter top-k probabilities
            batch_indices = jnp.arange(batch_size)[:, None, None]
            seq_indices = jnp.arange(seq_len)[None, :, None]
            router_probs = router_probs.at[batch_indices, seq_indices, top_k_indices].set(top_k_probs)

            router_aux = {
                "router_logits": router_logits,
                "top_k_indices": top_k_indices,
                "top_k_probs": top_k_probs,
            }

        else:
            raise ValueError(f"Unknown router_type: {self.router_type}")

        return router_probs, router_aux


@at.typecheck
class MoEFeedForward(nn.Module):
    """Mixture of Experts Feed-Forward layer.

    Replaces standard FFN with multiple expert FFNs and a router.
    """

    features: int
    hidden_dim: int
    moe_config: MoEConfig
    dtype: str = "bfloat16"

    @nn.compact
    def __call__(
        self,
        x: at.Float[at.Array, "b t d"],
        movement_labels: Optional[at.Int[at.Array, "b"]] = None,
        deterministic: bool = True,
    ) -> tuple[at.Float[at.Array, "b t d"], dict]:
        """Forward pass with expert routing.

        Args:
            x: Input tensor [batch, seq_len, features].
            movement_labels: Movement labels for supervised routing [batch].
            deterministic: Whether to use deterministic mode (no dropout).

        Returns:
            output: Expert-routed output [batch, seq_len, features].
            moe_aux: Auxiliary outputs (router probs, losses, etc.).
        """
        batch_size, seq_len, _ = x.shape
        dtype = jnp.dtype(self.dtype)

        # Route tokens to experts
        router = Router(
            num_experts=self.moe_config.num_experts,
            router_type=self.moe_config.router_type,
            top_k=self.moe_config.top_k,
            dtype=self.dtype,
            name="router",
        )
        router_probs, router_aux = router(x, movement_labels, deterministic)

        # Create expert networks
        expert_outputs = []
        for i in range(self.moe_config.num_experts):
            # Each expert is a standard FFN
            expert_output = self._expert_forward(x, expert_idx=i)
            expert_outputs.append(expert_output)

        # Stack experts: [num_experts, batch, seq_len, features]
        expert_outputs = jnp.stack(expert_outputs, axis=0)

        # Weighted combination of expert outputs
        # router_probs: [batch, seq_len, num_experts]
        # expert_outputs: [num_experts, batch, seq_len, features]
        # output: [batch, seq_len, features]
        output = jnp.einsum("bte,ebtn->btn", router_probs, expert_outputs)

        # Apply expert dropout if configured
        if self.moe_config.expert_dropout > 0.0 and not deterministic:
            dropout_rng = self.make_rng("dropout")
            keep_prob = 1.0 - self.moe_config.expert_dropout
            mask = jax.random.bernoulli(dropout_rng, keep_prob, output.shape)
            output = jnp.where(mask, output / keep_prob, 0.0)

        # Compute auxiliary losses
        moe_aux = self._compute_auxiliary_losses(router_probs, router_aux, x)

        return output, moe_aux

    def _expert_forward(
        self,
        x: at.Float[at.Array, "b t d"],
        expert_idx: int,
    ) -> at.Float[at.Array, "b t d"]:
        """Forward pass through a single expert.

        Args:
            x: Input tensor [batch, seq_len, features].
            expert_idx: Index of the expert.

        Returns:
            Expert output [batch, seq_len, features].
        """
        dtype = jnp.dtype(self.dtype)

        # Gating network (GLU-style)
        w_gating = self.param(
            f"expert_{expert_idx}_gating",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)

        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        # Linear projection back to features
        w_linear = self.param(
            f"expert_{expert_idx}_linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)

        output = jnp.dot(activations, w_linear)
        return output

    def _compute_auxiliary_losses(
        self,
        router_probs: at.Float[at.Array, "b t e"],
        router_aux: dict,
        inputs: at.Float[at.Array, "b t d"],
    ) -> dict:
        """Compute auxiliary losses for MoE training.

        Args:
            router_probs: Router probabilities [batch, seq_len, num_experts].
            router_aux: Auxiliary outputs from router.
            inputs: Input tensor [batch, seq_len, features].

        Returns:
            Dictionary with auxiliary outputs and losses.
        """
        aux = {**router_aux}

        # Load balancing loss (encourages balanced expert usage)
        if self.moe_config.load_balancing_loss_coef > 0:
            # Average probability of routing to each expert
            expert_probs = jnp.mean(router_probs, axis=(0, 1))  # [num_experts]

            # Ideal uniform distribution
            uniform = 1.0 / self.moe_config.num_experts

            # Load balancing loss (encourages uniform expert usage)
            load_balance_loss = jnp.sum(jnp.square(expert_probs - uniform))
            aux["load_balance_loss"] = load_balance_loss * self.moe_config.load_balancing_loss_coef
        else:
            aux["load_balance_loss"] = 0.0

        # Router z-loss (encourages lower router logits magnitude)
        if self.moe_config.router_z_loss_coef > 0 and "router_logits" in router_aux:
            router_logits = router_aux["router_logits"]
            # Z-loss from ST-MoE paper
            z_loss = jnp.mean(jnp.square(jax.nn.logsumexp(router_logits, axis=-1)))
            aux["router_z_loss"] = z_loss * self.moe_config.router_z_loss_coef
        else:
            aux["router_z_loss"] = 0.0

        # Total auxiliary loss
        aux["moe_aux_loss"] = aux["load_balance_loss"] + aux["router_z_loss"]

        # Expert usage statistics
        expert_usage = jnp.sum(router_probs, axis=(0, 1))  # [num_experts]
        aux["expert_usage"] = expert_usage
        aux["expert_usage_fraction"] = expert_usage / jnp.sum(expert_usage)

        return aux


def replace_ffn_with_moe(
    features: int,
    hidden_dim: int,
    moe_config: MoEConfig,
    dtype: str = "bfloat16",
) -> MoEFeedForward:
    """Helper to create a MoE layer that replaces a standard FFN.

    Args:
        features: Feature dimension.
        hidden_dim: Hidden dimension of FFN.
        moe_config: MoE configuration.
        dtype: Data type for computations.

    Returns:
        MoEFeedForward layer.
    """
    return MoEFeedForward(
        features=features,
        hidden_dim=hidden_dim,
        moe_config=moe_config,
        dtype=dtype,
    )


# Utility functions for MoE training

def aggregate_moe_losses(moe_aux_list: list[dict]) -> dict:
    """Aggregate MoE auxiliary losses from multiple layers.

    Args:
        moe_aux_list: List of MoE auxiliary dictionaries from each layer.

    Returns:
        Dictionary with aggregated losses and statistics.
    """
    if not moe_aux_list:
        return {"total_moe_loss": 0.0}

    # Aggregate losses
    total_load_balance_loss = sum(aux.get("load_balance_loss", 0.0) for aux in moe_aux_list)
    total_router_z_loss = sum(aux.get("router_z_loss", 0.0) for aux in moe_aux_list)
    total_moe_loss = sum(aux.get("moe_aux_loss", 0.0) for aux in moe_aux_list)

    # Aggregate expert usage statistics
    expert_usage_fractions = [aux.get("expert_usage_fraction") for aux in moe_aux_list if "expert_usage_fraction" in aux]
    if expert_usage_fractions:
        avg_expert_usage = jnp.mean(jnp.stack(expert_usage_fractions), axis=0)
    else:
        avg_expert_usage = None

    return {
        "total_moe_loss": total_moe_loss,
        "total_load_balance_loss": total_load_balance_loss,
        "total_router_z_loss": total_router_z_loss,
        "avg_expert_usage": avg_expert_usage,
        "num_moe_layers": len(moe_aux_list),
    }

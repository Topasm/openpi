"""Gemma model with Mixture of Experts (MoE) FFN layers.

This module extends the standard Gemma architecture by replacing FFN layers
with MoE layers that route based on movement types.
"""

import dataclasses
from typing import Sequence, Literal
import jax
import jax.numpy as jnp
import flax.linen as nn

import openpi.models.gemma as gemma
import openpi.models.moe as moe
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding


@dataclasses.dataclass
class MoEGemmaConfig:
    """Configuration for Gemma with MoE.

    Extends the base Gemma config with MoE-specific settings.
    """
    # Base Gemma config
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)

    # MoE-specific config
    moe_config: moe.MoEConfig = dataclasses.field(default_factory=moe.MoEConfig)
    moe_layers: Sequence[int] | Literal["all"] = "all"  # Which layers to apply MoE to


def convert_to_moe_config(
    base_config: gemma.Config,
    moe_config: moe.MoEConfig | None = None,
    moe_layers: Sequence[int] | Literal["all"] = "all",
) -> MoEGemmaConfig:
    """Convert a base Gemma config to MoE config.

    Args:
        base_config: Base Gemma configuration.
        moe_config: MoE configuration. If None, uses default.
        moe_layers: Which layers to apply MoE to. Either "all" or list of layer indices.

    Returns:
        MoEGemmaConfig with MoE settings.
    """
    if moe_config is None:
        moe_config = moe.MoEConfig()

    return MoEGemmaConfig(
        width=base_config.width,
        depth=base_config.depth,
        mlp_dim=base_config.mlp_dim,
        num_heads=base_config.num_heads,
        num_kv_heads=base_config.num_kv_heads,
        head_dim=base_config.head_dim,
        lora_configs=base_config.lora_configs,
        moe_config=moe_config,
        moe_layers=moe_layers,
    )


@at.typecheck
class MoEBlock(nn.Module):
    """Transformer block with MoE FFN."""

    configs: tuple[MoEGemmaConfig, ...]
    layer_idx: int  # Index of this layer in the stack

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    def _should_use_moe(self, config: MoEGemmaConfig) -> bool:
        """Determine if this layer should use MoE."""
        if config.moe_layers == "all":
            return True
        return self.layer_idx in config.moe_layers

    @nn.compact
    def __call__(
        self,
        xs,
        kv_cache,
        positions,
        attn_mask,
        adarms_cond,
        movement_labels=None,
        deterministic=True,
    ):
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        # Attention (same as base Gemma)
        # Convert MoEGemmaConfig to base Config for attention
        base_configs = tuple(
            gemma.Config(
                width=c.width,
                depth=c.depth,
                mlp_dim=c.mlp_dim,
                num_heads=c.num_heads,
                num_kv_heads=c.num_kv_heads,
                head_dim=c.head_dim,
                lora_configs=c.lora_configs,
            )
            for c in self.configs
        )

        attn = gemma.Attention(configs=base_configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = gemma.RMSNorm(name=gemma._name("pre_attention_norm", i))(x, adarms_cond[i])
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [gemma._gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        # FFN (with optional MoE)
        out = []
        gates = []
        moe_aux_list = []

        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x_norm, gate = gemma.RMSNorm(name=gemma._name("pre_ffw_norm", i))(x, adarms_cond[i])

                # Check if we should use MoE for this expert/layer
                if self._should_use_moe(config):
                    # Use MoE FFN
                    moe_layer = moe.MoEFeedForward(
                        features=config.width,
                        hidden_dim=config.mlp_dim,
                        moe_config=config.moe_config,
                        dtype="bfloat16",
                        name=gemma._name("moe_mlp", i),
                    )

                    x_out, moe_aux = moe_layer(
                        x_norm,
                        movement_labels=movement_labels,
                        deterministic=deterministic,
                    )
                    moe_aux_list.append(moe_aux)
                else:
                    # Use standard FFN
                    x_out = lora.FeedForward(
                        features=config.width,
                        hidden_dim=config.mlp_dim,
                        name=gemma._name("mlp", i),
                        lora_config=config.lora_configs.get("ffn"),
                    )(x_norm)

                x = x_out
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [gemma._gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        # Attach MoE auxiliary info
        if moe_aux_list:
            moe_aux = moe.aggregate_moe_losses(moe_aux_list)
        else:
            moe_aux = None

        return xs, kv_cache, moe_aux


@at.typecheck
class MoEModule(nn.Module):
    """Gemma transformer with MoE layers."""

    configs: Sequence[MoEGemmaConfig]
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False

    @nn.compact
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        movement_labels: at.Int[at.Array, "b"] | None = None,
        *,
        kv_cache: gemma.KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], gemma.KVCache, dict]:
        """Forward pass with MoE.

        Args:
            embedded: List of embedded tokens per expert.
            positions: Token positions.
            mask: Attention mask.
            adarms_cond: AdaRMS conditioning.
            movement_labels: Movement labels for MoE routing [batch].
            kv_cache: Optional KV cache.
            deterministic: Whether to use deterministic mode.

        Returns:
            outputs: List of outputs per expert.
            kv_cache: Updated KV cache.
            moe_aux: Aggregated MoE auxiliary outputs.
        """
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        # Run through layers
        all_moe_aux = []

        # Handle KV cache properly - it should be a tuple of (keys, values) for each layer
        # or None if no caching
        if kv_cache is not None:
            # Split KV cache for each layer
            kv_caches = [(kv_cache[0][layer_idx], kv_cache[1][layer_idx]) for layer_idx in range(self.configs[0].depth)]
            new_kv_caches_k = []
            new_kv_caches_v = []
        else:
            kv_caches = [None] * self.configs[0].depth
            new_kv_caches_k = None
            new_kv_caches_v = None

        for layer_idx in range(self.configs[0].depth):
            block = nn.remat(
                MoEBlock,
                prevent_cse=False,
                static_argnums=(7,),  # deterministic
                policy=jax.checkpoint_policies.nothing_saveable,
            )(
                configs=self.configs,
                layer_idx=layer_idx,
                dropout=self.dropout,
                dropout_bdims=self.dropout_bdims,
                name=f"layer_{layer_idx}",
            )
            embedded, layer_kv_cache, moe_aux = block(
                embedded,
                kv_caches[layer_idx],  # Per-layer cache
                positions,
                mask,
                adarms_cond,
                movement_labels,
                deterministic,
            )

            # Collect new KV cache for this layer
            if kv_cache is not None and layer_kv_cache is not None:
                new_kv_caches_k.append(layer_kv_cache[0])
                new_kv_caches_v.append(layer_kv_cache[1])

            if moe_aux is not None:
                all_moe_aux.append(moe_aux)

        # Reconstruct full KV cache
        if kv_cache is not None:
            kv_cache = (jnp.stack(new_kv_caches_k), jnp.stack(new_kv_caches_v))
        else:
            kv_cache = None

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        # Final norm
        outputs = [
            gemma.RMSNorm(name=gemma._name("final_norm", i))(e, a)[0] if e is not None else e
            for i, (e, a) in enumerate(zip(embedded, adarms_cond, strict=True))
        ]

        # Aggregate MoE losses across all layers
        if all_moe_aux:
            # Stack auxiliary info from all layers
            total_moe_loss = sum(aux.get("moe_aux_loss", 0.0) for aux in all_moe_aux)
            total_load_balance = sum(aux.get("load_balance_loss", 0.0) for aux in all_moe_aux)
            total_router_z = sum(aux.get("router_z_loss", 0.0) for aux in all_moe_aux)

            # Average expert usage across layers
            expert_usages = [aux.get("expert_usage_fraction") for aux in all_moe_aux if "expert_usage_fraction" in aux]
            if expert_usages:
                avg_expert_usage = jnp.mean(jnp.stack(expert_usages), axis=0)
            else:
                avg_expert_usage = None

            aggregated_moe_aux = {
                "total_moe_loss": total_moe_loss,
                "total_load_balance_loss": total_load_balance,
                "total_router_z_loss": total_router_z,
                "avg_expert_usage": avg_expert_usage,
                "num_moe_layers": len(all_moe_aux),
            }
        else:
            aggregated_moe_aux = {"total_moe_loss": 0.0}

        return outputs, kv_cache, aggregated_moe_aux

    @at.typecheck
    @nn.compact
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        embedder = gemma.Embedder(
            vocab_size=gemma.PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        return embedder.encode(tokens).astype(self.embed_dtype)

    def init(self, use_adarms: Sequence[bool]):
        """Initialize all parameters."""
        # Initialize embedder
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))

        # Initialize transformer layers
        # Number of tokens = 1 token per config (simplified for init)
        num_tokens_per_config = 1
        total_tokens = len(self.configs) * num_tokens_per_config

        self(
            [jnp.zeros((1, num_tokens_per_config, c.width)) for c in self.configs],
            jnp.zeros((1, total_tokens), dtype=jnp.int32),
            jnp.zeros((1, total_tokens, total_tokens), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)
            ],
            movement_labels=jnp.zeros((1,), dtype=jnp.int32),
            kv_cache=None,  # No KV cache during init
        )

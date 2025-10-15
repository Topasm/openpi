"""Pi0 configuration with MoE support.

This module extends Pi0Config to include MoE parameters for future implementation.
Currently creates standard Pi0 models while movement labels are being collected.
"""

import dataclasses
import flax.nnx as nnx

from openpi.models import pi0, pi0_config, moe
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

    def create(self, rng: at.KeyArrayLike) -> pi0.Pi0:
        """Create a Pi0 model instance.

        Note: Full MoE implementation is in progress. This currently creates a standard
        Pi0 model. Movement labels are computed during training for future MoE routing.

        Args:
            rng: Random number generator key.

        Returns:
            Standard Pi0 model instance.
        """
        # Create standard Pi0 until full MoE implementation is complete
        return pi0.Pi0(self, rngs=nnx.Rngs(rng))


# TODO: Full Pi0MoE model class implementation will be added here
# This will include:
# - Pi0MoE class extending Pi0
# - MoE-enabled Gemma transformer blocks
# - compute_loss_with_moe() method
# - MoE auxiliary loss computation

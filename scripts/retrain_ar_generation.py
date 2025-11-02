#!/usr/bin/env python3
"""
Retrain hierarchical model with scheduled sampling for proper autoregressive generation.

This script fine-tunes an existing checkpoint to fix the exposure bias problem that causes
"cococo..." loops during autoregressive skill generation. It uses scheduled sampling to
gradually expose the model to its own predictions during training.

Usage:
    # Fine-tune from existing checkpoint
    python scripts/retrain_ar_generation.py \\
        --checkpoint_path=dataset/openpi/checkpoints/5000 \\
        --output_dir=outputs/ar_retrain \\
        --num_steps=50000

    # Resume from interrupted training
    python scripts/retrain_ar_generation.py \\
        --checkpoint_path=outputs/ar_retrain/checkpoints/25000 \\
        --output_dir=outputs/ar_retrain \\
        --num_steps=50000 \\
        --resume=True
"""

import logging
import pathlib
from typing import Optional

import jax
import tyro
from dataclasses import dataclass

from openpi.training import config as _config
from openpi.training import train as _train
from openpi.models import pi0_hierarchical

logger = logging.getLogger(__name__)


@dataclass
class RetrainingConfig:
    """Configuration for autoregressive generation retraining."""

    # Checkpoint Settings
    checkpoint_path: str = "../../../dataset/openpi/checkpoints/5000"
    """Path to existing checkpoint to fine-tune from"""

    output_dir: str = "./outputs/ar_retrain"
    """Directory for saving checkpoints and logs"""

    resume: bool = False
    """Resume from checkpoint in output_dir instead of checkpoint_path"""

    # Training Settings
    num_steps: int = 50000
    """Number of training steps (50K recommended for scheduled sampling)"""

    learning_rate: float = 1e-5
    """Learning rate for fine-tuning (lower than initial training)"""

    # Scheduled Sampling Settings
    use_scheduled_sampling: bool = True
    """Enable scheduled sampling (should always be True for retraining)"""

    initial_teacher_forcing: float = 1.0
    """Initial teacher forcing ratio (start with 100% ground truth)"""

    final_teacher_forcing: float = 0.3
    """Final teacher forcing ratio (end with 30% ground truth, 70% predictions)"""

    tf_decay_steps: int = 50000
    """Steps over which to decay teacher forcing ratio"""

    # Validation Settings
    val_log_interval: int = 2000
    """Steps between validation runs"""

    save_interval: int = 5000
    """Steps between checkpoint saves"""

    # Data Settings
    batch_size: int = 8
    """Batch size for training"""

    num_workers: int = 0
    """Number of data loading workers (0=disable multiprocessing)"""


def main():
    """Main retraining function."""
    cfg = tyro.cli(RetrainingConfig)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("=" * 80)
    logger.info("Autoregressive Generation Retraining")
    logger.info("=" * 80)
    logger.info(f"Checkpoint path: {cfg.checkpoint_path}")
    logger.info(f"Output directory: {cfg.output_dir}")
    logger.info(f"Training steps: {cfg.num_steps}")
    logger.info(f"Scheduled sampling: {cfg.use_scheduled_sampling}")
    logger.info(f"Teacher forcing: {cfg.initial_teacher_forcing} → {cfg.final_teacher_forcing}")
    logger.info(f"Decay steps: {cfg.tf_decay_steps}")
    logger.info("=" * 80)

    # Load base hierarchical config
    base_config = _config.get_config("pi0_b1k_hierarchical")

    # Override model config with scheduled sampling settings
    model_config = pi0_hierarchical.Pi0HierarchicalConfig(
        action_horizon=50,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        skill_loss_weight=10.0,
        action_loss_weight=1.0,
        max_skill_tokens=64,
        # Enable scheduled sampling
        use_scheduled_sampling=cfg.use_scheduled_sampling,
        initial_teacher_forcing=cfg.initial_teacher_forcing,
        final_teacher_forcing=cfg.final_teacher_forcing,
        tf_decay_steps=cfg.tf_decay_steps,
    )

    # Update training config
    train_config = _config.TrainConfig(
        name="pi0_b1k_hierarchical_ar_retrain",
        exp_name="openpi_hierarchical_ar_retrain",
        project_name="B1K_Hierarchical_AR_Retrain",
        model=model_config,
        data=base_config.data,  # Reuse same data config
        weight_loader=None,  # We'll load from checkpoint, not base weights
        num_train_steps=cfg.num_steps,
        freeze_filter=model_config.get_freeze_filter(),
        ema_decay=None,
        val_log_interval=cfg.val_log_interval,
        val_repo_id=base_config.val_repo_id,
        val_episodes_index=base_config.val_episodes_index,
        assets_base_dir=str(pathlib.Path(cfg.output_dir) / "assets"),
        checkpoint_base_dir=str(pathlib.Path(cfg.output_dir) / "checkpoints"),
        num_workers=cfg.num_workers,
        # Lower learning rate for fine-tuning
        optimizer_kwargs={"learning_rate": cfg.learning_rate},
    )

    logger.info("Starting training with scheduled sampling...")
    logger.info(f"Model will see {cfg.initial_teacher_forcing:.0%} ground truth at start")
    logger.info(f"Model will see {cfg.final_teacher_forcing:.0%} ground truth at end")
    logger.info(f"This means model predictions used {1-cfg.initial_teacher_forcing:.0%} → {1-cfg.final_teacher_forcing:.0%}")

    # Determine initial checkpoint
    if cfg.resume:
        initial_checkpoint = pathlib.Path(cfg.output_dir) / "checkpoints" / "latest"
        logger.info(f"Resuming from: {initial_checkpoint}")
    else:
        initial_checkpoint = pathlib.Path(cfg.checkpoint_path)
        logger.info(f"Fine-tuning from: {initial_checkpoint}")

    # Run training
    try:
        _train.main(
            config=train_config,
            initial_checkpoint=str(initial_checkpoint) if initial_checkpoint.exists() else None,
        )
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Checkpoints saved to: {cfg.output_dir}/checkpoints")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Evaluate AR generation quality:")
        logger.info(f"   python scripts/eval_ar_generation.py --checkpoint_path={cfg.output_dir}/checkpoints/50000")
        logger.info("2. Test with serve_b1k.py:")
        logger.info(f"   python scripts/serve_b1k.py --policy.dir={cfg.output_dir}/checkpoints/50000")
        logger.info("")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info(f"Checkpoints saved to: {cfg.output_dir}/checkpoints")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

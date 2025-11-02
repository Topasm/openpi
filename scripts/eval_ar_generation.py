#!/usr/bin/env python3
"""
Evaluate autoregressive skill generation quality.

This script tests the model's ability to generate valid skill JSON text autoregressively
without falling into "cococo..." loops. It measures various quality metrics including
valid JSON rate, skill/object accuracy, and EOS detection.

Usage:
    # Evaluate checkpoint
    python scripts/eval_ar_generation.py \\
        --checkpoint_path=outputs/ar_retrain/checkpoints/50000 \\
        --num_samples=100

    # Compare before/after retraining
    python scripts/eval_ar_generation.py \\
        --checkpoint_path_before=dataset/openpi/checkpoints/5000 \\
        --checkpoint_path_after=outputs/ar_retrain/checkpoints/50000 \\
        --num_samples=100
"""

import json
import logging
import pathlib
from typing import Optional, Dict, List
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tyro
import numpy as np
from tqdm import tqdm

from openpi.training import config as _config
from openpi.models import pi0_hierarchical, model as _model
from openpi.training import hierarchical_transforms

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for AR generation evaluation."""

    checkpoint_path: str = "../../../dataset/openpi/checkpoints/5000"
    """Path to checkpoint to evaluate"""

    checkpoint_path_before: Optional[str] = None
    """Optional: Path to 'before' checkpoint for comparison"""

    checkpoint_path_after: Optional[str] = None
    """Optional: Path to 'after' checkpoint for comparison"""

    num_samples: int = 100
    """Number of test samples to evaluate"""

    max_length: int = 64
    """Maximum skill token length"""

    temperature: float = 0.0
    """Sampling temperature (0.0 = greedy)"""

    dataset_split: str = "validation"
    """Dataset split to use (train/validation)"""


def parse_skill_json(skill_text: str) -> Optional[Dict]:
    """Parse skill JSON text, handling various formats."""
    try:
        # Remove <EOS_SKILL> if present
        skill_text = skill_text.replace("<EOS_SKILL>", "").strip()

        # Try to parse as JSON
        skill_dict = json.loads(skill_text)
        return skill_dict
    except json.JSONDecodeError:
        return None


def is_garbled(skill_text: str) -> bool:
    """Check if skill text is garbled (e.g., 'cococo...' or 'unun...')."""
    # Check for repeating 2-character patterns
    if len(skill_text) < 10:
        return False

    # Extract 2-char sequences
    sequences = [skill_text[i:i+2] for i in range(0, len(skill_text)-1, 2)]

    # Count most common sequence
    if sequences:
        from collections import Counter
        counter = Counter(sequences)
        most_common = counter.most_common(1)[0]
        # If most common sequence appears more than 5 times, it's likely garbled
        if most_common[1] > 5:
            return True

    return False


def evaluate_model(
    model: pi0_hierarchical.Pi0Hierarchical,
    tokenizer,
    test_data: List[Dict],
    config: EvalConfig
) -> Dict[str, float]:
    """Evaluate AR generation quality on test data."""

    metrics = {
        "valid_json": [],
        "is_garbled": [],
        "skill_name_match": [],
        "object_match": [],
        "exact_match": [],
        "has_eos": [],
        "avg_length": [],
    }

    rng = jax.random.PRNGKey(42)

    for sample in tqdm(test_data[:config.num_samples], desc="Evaluating"):
        # Get observation and ground truth
        observation = sample["observation"]
        ground_truth_skill = sample["skill_text"]

        # Generate skill
        rng, gen_rng = jax.random.split(rng)
        generated_text, plan_embedding, has_eos = model.generate_skill_autoregressive(
            gen_rng,
            observation,
            memory_tokens=None,
            memory_mask=None,
            tokenizer=tokenizer,
            max_length=config.max_length,
            temperature=config.temperature
        )

        # Parse generated and ground truth
        gen_skill = parse_skill_json(generated_text)
        gt_skill = parse_skill_json(ground_truth_skill)

        # Compute metrics
        metrics["valid_json"].append(gen_skill is not None)
        metrics["is_garbled"].append(is_garbled(generated_text))
        metrics["has_eos"].append(has_eos)
        metrics["avg_length"].append(len(generated_text))

        if gen_skill and gt_skill:
            metrics["skill_name_match"].append(
                gen_skill.get("skill") == gt_skill.get("skill")
            )
            metrics["object_match"].append(
                gen_skill.get("obj") == gt_skill.get("obj")
            )
            metrics["exact_match"].append(gen_skill == gt_skill)
        else:
            metrics["skill_name_match"].append(False)
            metrics["object_match"].append(False)
            metrics["exact_match"].append(False)

    # Aggregate metrics
    results = {}
    for key, values in metrics.items():
        if key == "avg_length":
            results[key] = float(np.mean(values))
        else:
            results[key] = float(np.mean(values))

    return results


def print_results(results: Dict[str, float], title: str = "Results"):
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(f"  Valid JSON:        {results['valid_json']:>6.1%}  (should be >95%)")
    print(f"  Garbled output:    {results['is_garbled']:>6.1%}  (should be <5%)")
    print(f"  Skill name match:  {results['skill_name_match']:>6.1%}  (should be >80%)")
    print(f"  Object match:      {results['object_match']:>6.1%}  (should be >80%)")
    print(f"  Exact match:       {results['exact_match']:>6.1%}  (should be >60%)")
    print(f"  EOS detection:     {results['has_eos']:>6.1%}  (context-dependent)")
    print(f"  Avg length:        {results['avg_length']:>6.1f} chars")
    print("=" * 80)


def load_checkpoint_and_tokenizer(checkpoint_path: str):
    """Load model checkpoint and tokenizer."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Load config
    train_config = _config.get_config("pi0_b1k_hierarchical")

    # Create model
    rng = jax.random.PRNGKey(0)
    model = train_config.model.create(rng)

    # Load parameters
    import orbax.checkpoint as ocp
    checkpointer = ocp.PyTreeCheckpointer()
    params_path = pathlib.Path(checkpoint_path) / "params"

    if params_path.exists():
        params = checkpointer.restore(params_path)
        # Update model parameters
        model = model.replace(params)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {params_path}")

    # Create tokenizer
    tokenizer = hierarchical_transforms.create_hierarchical_tokenizer()

    logger.info(f"Model loaded successfully")
    logger.info(f"Tokenizer vocab size: {len(tokenizer.tokenizer)}")
    logger.info(f"EOS_SKILL token ID: {tokenizer.eos_skill_token_id}")

    return model, tokenizer


def main():
    """Main evaluation function."""
    cfg = tyro.cli(EvalConfig)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("=" * 80)
    logger.info("Autoregressive Generation Evaluation")
    logger.info("=" * 80)

    # Load test data (simplified - you'll need to implement actual data loading)
    logger.info("Loading test dataset...")
    # TODO: Load actual test data from B1K dataset
    # For now, using placeholder
    test_data = []  # Replace with actual data loading
    logger.info(f"Loaded {len(test_data)} test samples")

    if cfg.checkpoint_path_before and cfg.checkpoint_path_after:
        # Compare before/after
        logger.info("\nComparing checkpoints...")

        # Evaluate before
        model_before, tokenizer = load_checkpoint_and_tokenizer(cfg.checkpoint_path_before)
        results_before = evaluate_model(model_before, tokenizer, test_data, cfg)
        print_results(results_before, title="Before Retraining")

        # Evaluate after
        model_after, tokenizer = load_checkpoint_and_tokenizer(cfg.checkpoint_path_after)
        results_after = evaluate_model(model_after, tokenizer, test_data, cfg)
        print_results(results_after, title="After Retraining")

        # Print improvement
        print("\n" + "=" * 80)
        print("  Improvement")
        print("=" * 80)
        for key in results_before.keys():
            if key != "avg_length":
                delta = results_after[key] - results_before[key]
                symbol = "✅" if delta > 0 else "⚠️" if delta < 0 else "→"
                print(f"  {key:20s}: {delta:+6.1%} {symbol}")
        print("=" * 80)

    else:
        # Evaluate single checkpoint
        model, tokenizer = load_checkpoint_and_tokenizer(cfg.checkpoint_path)
        results = evaluate_model(model, tokenizer, test_data, cfg)
        print_results(results, title="Evaluation Results")

        # Print pass/fail
        print("\n" + "=" * 80)
        print("  Success Criteria")
        print("=" * 80)
        checks = [
            ("Valid JSON >95%", results["valid_json"] > 0.95),
            ("Garbled <5%", results["is_garbled"] < 0.05),
            ("Skill match >80%", results["skill_name_match"] > 0.80),
            ("Object match >80%", results["object_match"] > 0.80),
        ]

        for check_name, passed in checks:
            symbol = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {check_name:25s}: {symbol}")

        all_passed = all(passed for _, passed in checks)
        print("=" * 80)
        if all_passed:
            print("  ✅ All checks passed! AR generation is working correctly.")
        else:
            print("  ⚠️  Some checks failed. Consider retraining or adjusting parameters.")
        print("=" * 80)


if __name__ == "__main__":
    main()

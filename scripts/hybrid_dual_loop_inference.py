"""
Hybrid Dual Loop Inference with Trained EOS Detection.

This script implements the complete "Pragmatic Dual Loop VLA" architecture:

- Slow-Loop (Phase 1): Autoregressive skill generation for planning
- Fast-Loop (Phase 2): Real-time action generation + trained EOS detection
- Observe-then-Plan (Phase 3): Cold start fix with initial context gathering
- Explicit Router (Phase 4): Triggers re-planning based on EOS probability

Architecture:
    - Fast-Loop runs every frame (real-time)
    - Slow-Loop runs intermittently when EOS detected (every ~50-100 steps)
    - Plan embedding conditions Fast-Loop actions (goal-directed)
    - Memory accumulates completed skills for long-horizon tasks

Usage:
    python scripts/hybrid_dual_loop_inference.py \
        --checkpoint_path /path/to/checkpoint \
        --task_prompt "Put the trash in the bin" \
        --observe_steps 10 \
        --eos_threshold 0.7 \
        --max_steps 1000
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional
import sys

import jax
import jax.numpy as jnp
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from openpi.models.pi0_hierarchical import Pi0Hierarchical
from openpi.models.model import Observation
from openpi.models.tokenizer import HierarchicalTokenizer

logger = logging.getLogger(__name__)


def observe_then_plan(
    model: Pi0Hierarchical,
    env,
    task_prompt_tokens: jnp.ndarray,
    task_prompt_mask: jnp.ndarray,
    tokenizer: HierarchicalTokenizer,
    observe_steps: int = 10,
    rng: jax.random.PRNGKey = None
) -> tuple[str, jnp.ndarray]:
    """
    Phase 3: Observe initial frames before making first plan.

    Inspired by CMeRT's "near-past context" approach. Instead of planning
    from a single frame (t=0), we gather visual context from multiple frames
    to make a more informed initial plan.

    Args:
        model: Pi0Hierarchical model
        env: Environment/robot interface
        task_prompt_tokens: Tokenized task prompt [1, max_len]
        task_prompt_mask: Prompt mask [1, max_len]
        tokenizer: HierarchicalTokenizer for skill decoding
        observe_steps: Number of frames to observe (default: 10)
        rng: Random key for exploration actions

    Returns:
        initial_plan_text: Generated skill JSON
        initial_plan_embedding: Hidden state [1, hidden_dim] for Fast-Loop
    """
    logger.info(f"=== OBSERVE-THEN-PLAN: Collecting {observe_steps} frames ===")

    # Collect observations
    observation_buffer = []
    for i in range(observe_steps):
        obs = env.get_observation()
        observation_buffer.append(obs)

        # Execute exploration action (could be no-op, random, or pre-defined)
        # For simplicity: no-op (zero action)
        if i < observe_steps - 1:  # Don't step on last observation
            zero_action = np.zeros(model.action_dim)
            env.step(zero_action)

        logger.debug(f"  Observation {i+1}/{observe_steps} collected")

    # Use the LAST observation (has most context from camera movement)
    final_observation = observation_buffer[-1]

    # Package into model.Observation format
    # (Assuming env.get_observation() returns compatible format)
    model_obs = Observation(
        images=final_observation['images'],  # Dict of camera views
        image_masks={k: jnp.ones(1, dtype=jnp.bool_) for k in final_observation['images']},
        state=jnp.array(final_observation['state'])[None, :],  # [1, state_dim]
        tokenized_prompt=task_prompt_tokens,
        tokenized_prompt_mask=task_prompt_mask
    )

    # Generate initial plan using Slow-Loop
    if rng is None:
        rng = jax.random.PRNGKey(42)

    plan_text, plan_embedding, has_eos = model.generate_skill_autoregressive(
        rng=rng,
        observation=model_obs,
        memory_tokens=None,  # No memory yet
        memory_mask=None,
        tokenizer=tokenizer,
        max_length=64,
        temperature=0.0  # Greedy for initial plan
    )

    logger.info(f"=== INITIAL PLAN GENERATED ===")
    logger.info(f"  Plan: {plan_text}")
    logger.info(f"  Has EOS: {has_eos}")

    return plan_text, plan_embedding


def hybrid_dual_loop_inference(
    model: Pi0Hierarchical,
    env,
    task_prompt: str,
    tokenizer: HierarchicalTokenizer,
    max_steps: int = 1000,
    observe_steps: int = 10,
    eos_threshold: float = 0.7,
    replan_on_deviation: bool = False,
    max_memory_tokens: int = 256,
    save_trajectory: bool = False,
    output_dir: Optional[Path] = None
) -> dict:
    """
    Main Hybrid Dual Loop inference with trained EOS detection.

    Architecture:
    - Slow-Loop: Autoregressive skill generation (intermittent, ~every 50-100 steps)
    - Fast-Loop: Action generation + EOS classification (every step, real-time)
    - Explicit Router: Triggers Slow-Loop based on EOS probability

    Args:
        model: Trained Pi0Hierarchical model
        env: Environment/robot interface
        task_prompt: Natural language task description
        tokenizer: HierarchicalTokenizer
        max_steps: Maximum execution steps
        observe_steps: Frames to observe before initial plan
        eos_threshold: EOS probability threshold for re-planning (0-1)
        replan_on_deviation: Whether to re-plan on plan deviation
        max_memory_tokens: Maximum memory length
        save_trajectory: Whether to save execution trajectory
        output_dir: Directory to save trajectory (if save_trajectory=True)

    Returns:
        results: Dict with execution statistics
    """
    # Initialize
    rng = jax.random.PRNGKey(42)
    memory_tokens = None
    memory_mask = None
    step_count = 0
    replan_count = 0
    done = False

    # Trajectory logging
    trajectory = {
        'task_prompt': task_prompt,
        'steps': [],
        'skills': [],
        'replans': []
    }

    # Tokenize task prompt
    task_prompt_tokens, task_prompt_mask = tokenizer.tokenize(task_prompt)
    task_prompt_tokens = jnp.array(task_prompt_tokens)[None, :]  # [1, max_len]
    task_prompt_mask = jnp.array(task_prompt_mask)[None, :]

    # ============================================================
    # PHASE 3: Observe-then-Plan (Cold Start)
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: OBSERVE-THEN-PLAN (COLD START)")
    logger.info("=" * 60)

    current_plan_text, current_plan_embedding = observe_then_plan(
        model=model,
        env=env,
        task_prompt_tokens=task_prompt_tokens,
        task_prompt_mask=task_prompt_mask,
        tokenizer=tokenizer,
        observe_steps=observe_steps,
        rng=rng
    )

    trajectory['skills'].append({
        'step': 0,
        'skill': current_plan_text,
        'type': 'initial'
    })

    # ============================================================
    # MAIN HYBRID DUAL LOOP
    # ============================================================
    logger.info("=" * 60)
    logger.info("MAIN LOOP: HYBRID DUAL EXECUTION")
    logger.info("=" * 60)

    while not done and step_count < max_steps:
        # Get current observation
        obs_dict = env.get_observation()

        # Convert to model.Observation format
        observation = Observation(
            images=obs_dict['images'],
            image_masks={k: jnp.ones(1, dtype=jnp.bool_) for k in obs_dict['images']},
            state=jnp.array(obs_dict['state'])[None, :],
            tokenized_prompt=task_prompt_tokens,
            tokenized_prompt_mask=task_prompt_mask
        )

        # ========================================================
        # PHASE 2: FAST-LOOP (Execute & Detect)
        # ========================================================
        rng, fast_rng = jax.random.split(rng)

        actions, eos_probability, skill_logits = model.execute_fast_loop(
            rng=fast_rng,
            observation=observation,
            current_plan_embedding=current_plan_embedding,
            num_ode_steps=10  # Could reduce to 5 for speed
        )

        # Execute first action from chunk
        action_to_execute = np.array(actions[0, 0, :])  # [action_dim]
        obs_dict, reward, done, info = env.step(action_to_execute)
        step_count += 1

        logger.debug(
            f"Step {step_count}: "
            f"EOS prob={eos_probability:.3f}, "
            f"Plan: {current_plan_text[:50]}..."
        )

        # Log trajectory step
        if save_trajectory:
            trajectory['steps'].append({
                'step': step_count,
                'eos_prob': float(eos_probability),
                'action': action_to_execute.tolist(),
                'reward': float(reward) if reward is not None else None,
                'current_skill': current_plan_text
            })

        # ========================================================
        # EXPLICIT ROUTER: Decide whether to trigger Slow-Loop
        # ========================================================
        should_replan = False
        replan_reason = ""

        # Trigger 1: EOS Detection (trained)
        if eos_probability > eos_threshold:
            should_replan = True
            replan_reason = f"EOS detected (prob={eos_probability:.3f})"

        # Trigger 2: Plan Deviation (optional, heuristic)
        if replan_on_deviation and not should_replan:
            # Get most likely skill from skill_logits
            predicted_skill_id = int(jnp.argmax(skill_logits[0]))
            # Compare to current plan (would need to encode plan to token ID)
            # For simplicity, skip this check in MVP
            pass

        # ========================================================
        # PHASE 1: SLOW-LOOP (Re-plan if triggered)
        # ========================================================
        if should_replan:
            logger.info("=" * 60)
            logger.info(f"RE-PLANNING at step {step_count}: {replan_reason}")
            logger.info("=" * 60)

            replan_count += 1

            # 1. Update memory with completed skill
            past_skill_text = f'<PAST_SKILL>{current_plan_text}</PAST_SKILL>'

            # Tokenize and append to memory
            new_mem_tokens, new_mem_mask = tokenizer.tokenize(past_skill_text)
            new_mem_tokens = jnp.array(new_mem_tokens)[None, :]
            new_mem_mask = jnp.array(new_mem_mask)[None, :]

            if memory_tokens is None:
                memory_tokens = new_mem_tokens
                memory_mask = new_mem_mask
            else:
                memory_tokens = jnp.concatenate([memory_tokens, new_mem_tokens], axis=1)
                memory_mask = jnp.concatenate([memory_mask, new_mem_mask], axis=1)

                # Truncate if exceeds max length
                if memory_tokens.shape[1] > max_memory_tokens:
                    memory_tokens = memory_tokens[:, -max_memory_tokens:]
                    memory_mask = memory_mask[:, -max_memory_tokens:]

            logger.info(f"  Memory updated: {int(jnp.sum(memory_mask))} tokens")

            # 2. Generate new plan (Slow-Loop)
            rng, plan_rng = jax.random.split(rng)

            current_plan_text, current_plan_embedding, plan_has_eos = model.generate_skill_autoregressive(
                rng=plan_rng,
                observation=observation,
                memory_tokens=memory_tokens,
                memory_mask=memory_mask,
                tokenizer=tokenizer,
                max_length=64,
                temperature=0.0
            )

            logger.info(f"  New Plan: {current_plan_text}")
            logger.info("=" * 60)

            # Log replan
            trajectory['replans'].append({
                'step': step_count,
                'reason': replan_reason,
                'new_skill': current_plan_text
            })
            trajectory['skills'].append({
                'step': step_count,
                'skill': current_plan_text,
                'type': 'replan'
            })

        # Check for episode termination
        if done:
            logger.info(f"Episode completed at step {step_count}")
            break

    # ============================================================
    # RESULTS
    # ============================================================
    results = {
        'total_steps': step_count,
        'replan_count': replan_count,
        'final_plan': current_plan_text,
        'success': done,
        'avg_steps_per_skill': step_count / max(replan_count, 1)
    }

    logger.info("=" * 60)
    logger.info("EXECUTION COMPLETE")
    logger.info(f"  Total steps: {step_count}")
    logger.info(f"  Re-plans: {replan_count}")
    logger.info(f"  Avg steps/skill: {results['avg_steps_per_skill']:.1f}")
    logger.info("=" * 60)

    # Save trajectory if requested
    if save_trajectory and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = output_dir / f"trajectory_{step_count}steps.json"
        with open(trajectory_path, 'w') as f:
            json.dump(trajectory, f, indent=2)
        logger.info(f"Trajectory saved to {trajectory_path}")

    return results


class DummyEnv:
    """
    Dummy environment for testing the hybrid dual loop.

    Replace this with your actual environment/robot interface.
    """

    def __init__(self, action_dim: int = 32, state_dim: int = 32):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.step_count = 0
        self.max_steps = 500

    def get_observation(self) -> dict:
        """Return dummy observation."""
        return {
            'images': {
                'base_camera': np.random.randn(224, 224, 3).astype(np.float32),
                'left_wrist': np.random.randn(224, 224, 3).astype(np.float32),
                'right_wrist': np.random.randn(224, 224, 3).astype(np.float32),
            },
            'state': np.random.randn(self.state_dim).astype(np.float32)
        }

    def step(self, action: np.ndarray) -> tuple:
        """Execute action and return next observation."""
        self.step_count += 1
        done = self.step_count >= self.max_steps
        reward = 0.0 if not done else 1.0
        info = {}

        obs = self.get_observation()
        return obs, reward, done, info

    def reset(self):
        """Reset environment."""
        self.step_count = 0
        return self.get_observation()


def load_model_from_checkpoint(checkpoint_path: Path, config_name: str = "pi0_b1k_hierarchical") -> Pi0Hierarchical:
    """
    Load a trained Pi0Hierarchical model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory
        config_name: Name of the config to use

    Returns:
        Loaded model
    """
    # TODO: Implement actual checkpoint loading using orbax
    # This is a placeholder
    logger.warning("Model loading not yet fully implemented - would load from checkpoint here")
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Config name: {config_name}")

    # For now, return None and user should replace with actual loading logic
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Dual Loop Inference with Trained EOS Detection"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi0_b1k_hierarchical",
        help="Config name for model loading"
    )
    parser.add_argument(
        "--task_prompt",
        type=str,
        default="Put the trash in the bin",
        help="Natural language task description"
    )
    parser.add_argument(
        "--observe_steps",
        type=int,
        default=10,
        help="Number of frames to observe before initial planning"
    )
    parser.add_argument(
        "--eos_threshold",
        type=float,
        default=0.7,
        help="EOS probability threshold for re-planning (0-1)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum execution steps"
    )
    parser.add_argument(
        "--max_memory_tokens",
        type=int,
        default=256,
        help="Maximum memory length (tokens)"
    )
    parser.add_argument(
        "--num_ode_steps",
        type=int,
        default=10,
        help="ODE integration steps for flow matching (lower = faster)"
    )
    parser.add_argument(
        "--use_dummy_env",
        action="store_true",
        help="Use dummy environment for testing"
    )
    parser.add_argument(
        "--save_trajectory",
        action="store_true",
        help="Save execution trajectory to JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./hybrid_inference_outputs"),
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set random seed
    np.random.seed(args.seed)

    # Load model
    logger.info(f"Loading model from {args.checkpoint_path}")
    model = load_model_from_checkpoint(args.checkpoint_path, args.config_name)

    if model is None:
        logger.error("Model loading failed or not implemented. Please implement load_model_from_checkpoint()")
        logger.info("For testing, you can use --use_dummy_env flag, but model must still be loaded")
        return

    # Initialize tokenizer
    logger.info("Initializing HierarchicalTokenizer")
    tokenizer = HierarchicalTokenizer(max_len=64, add_eos_skill_token=True)
    logger.info(f"  Vocabulary size: {len(tokenizer)}")
    logger.info(f"  EOS token ID: {tokenizer.eos_skill_token_id}")

    # Initialize environment
    if args.use_dummy_env:
        logger.info("Using DummyEnv for testing")
        env = DummyEnv(action_dim=model.action_dim, state_dim=32)
    else:
        # TODO: Load actual environment
        logger.error("Real environment loading not implemented. Use --use_dummy_env for testing")
        return

    # Run hybrid dual loop inference
    logger.info("=" * 60)
    logger.info("STARTING HYBRID DUAL LOOP INFERENCE")
    logger.info(f"  Task: {args.task_prompt}")
    logger.info(f"  Observe steps: {args.observe_steps}")
    logger.info(f"  EOS threshold: {args.eos_threshold}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info("=" * 60)

    results = hybrid_dual_loop_inference(
        model=model,
        env=env,
        task_prompt=args.task_prompt,
        tokenizer=tokenizer,
        max_steps=args.max_steps,
        observe_steps=args.observe_steps,
        eos_threshold=args.eos_threshold,
        max_memory_tokens=args.max_memory_tokens,
        save_trajectory=args.save_trajectory,
        output_dir=args.output_dir if args.save_trajectory else None
    )

    # Print final results
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    for key, value in results.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()

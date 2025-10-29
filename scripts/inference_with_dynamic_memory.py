"""
Inference loop implementation with Dynamic Interleaved Cache (ContextMemory).

This script demonstrates how to use the ContextMemory state manager during
inference to maintain a dynamic, interleaved cache of past skills (text) and
current skill observations (vision).

The inference loop follows the pattern outlined in the Dynamic Memory plan:
1. Assemble model input from memory cache
2. Run model inference
3. Execute action
4. Add current frame to short-term memory
5. Check for skill completion signal and verbalize if needed

Usage:
    python scripts/inference_with_dynamic_memory.py \
        --checkpoint_path /path/to/checkpoint \
        --annotation_path /path/to/annotations \
        --episode_index 170 \
        --task_index 0
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from openpi.training.dynamic_memory import ContextMemory, summarize_skill_annotation
from openpi.training import skill_utils
from openpi.models.pi0_hierarchical import Pi0Hierarchical

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DynamicMemoryInferenceLoop:
    """
    Inference loop that uses ContextMemory for dynamic interleaved caching.

    This class manages the state during inference and coordinates between:
    - ContextMemory: Maintains the dynamic cache state
    - Model: Predicts actions and skills
    - Environment/Robot: Executes actions and provides observations
    """

    def __init__(
        self,
        model: Pi0Hierarchical,
        annotation_path: Path,
        max_short_term_frames: int = 10
    ):
        """
        Initialize the inference loop.

        Args:
            model: Trained Pi0Hierarchical model
            annotation_path: Path to the skill annotation JSON for the episode
            max_short_term_frames: Maximum number of visual frames in short-term memory
        """
        self.model = model
        self.memory_cache = ContextMemory(max_short_term_frames=max_short_term_frames)

        # Load ground-truth annotations (for skill completion detection)
        with open(annotation_path, 'r') as f:
            self.annotation_data = json.load(f)

        self.all_skills = self.annotation_data["skill_annotation"]
        self.current_skill_idx = 0
        self.current_frame_idx = 0

        logger.info(f"Initialized inference loop with {len(self.all_skills)} skills")

    def step(
        self,
        observation: dict,
        rng: jax.random.PRNGKey
    ) -> tuple[np.ndarray, Optional[dict]]:
        """
        Perform one inference step.

        Args:
            observation: Current observation dict with images, state, etc.
            rng: JAX random key

        Returns:
            (action, skill_prediction) tuple:
                - action: Predicted action [action_horizon, action_dim]
                - skill_prediction: Predicted skill dict (if skill prediction is needed)
        """
        # 1. Assemble model input from memory cache
        past_text, current_images = self.memory_cache.assemble_model_input()

        logger.debug(
            f"Frame {self.current_frame_idx}: "
            f"Past text: {len(past_text)} chars, "
            f"Current images: {current_images.shape if current_images is not None else None}"
        )

        # 2. Model inference
        # TODO: Implement actual model forward pass
        # For now, this is a placeholder that shows the structure
        # predicted_action, predicted_skill = self.model.predict(
        #     past_text=past_text,
        #     current_images=current_images,
        #     task_prompt=self.annotation_data["task_name"],
        #     rng=rng
        # )

        # Placeholder: Return dummy action
        predicted_action = np.zeros((50, 32))  # [action_horizon, action_dim]
        predicted_skill = None

        # 3. Execute action (in real deployment)
        # robot.execute(predicted_action)

        # 4. Add current frame to short-term memory
        # Extract the current visual observation
        # Assuming observation has a key like "image" or "rgb"
        if "image" in observation:
            current_frame = observation["image"]
        elif "rgb" in observation:
            current_frame = observation["rgb"]
        else:
            # Fallback: use any image data
            current_frame = np.random.randn(224, 224, 3)  # Placeholder

        self.memory_cache.add_visual_frame(
            frame_data=current_frame,
            current_skill_idx=self.current_skill_idx,
            timestamp=self.current_frame_idx
        )

        # 5. Check for skill completion signal
        self._check_skill_completion()

        # Increment frame counter
        self.current_frame_idx += 1

        return predicted_action, predicted_skill

    def _check_skill_completion(self):
        """
        Check if the current skill has completed and verbalize if needed.

        This uses the ground-truth annotations to determine when a skill ends.
        In a real deployment without ground-truth, you would use:
        - Model's skill prediction confidence
        - Environment state changes
        - Time-based heuristics
        """
        if self.current_skill_idx >= len(self.all_skills):
            # All skills completed
            return

        current_skill = self.all_skills[self.current_skill_idx]
        skill_end_frame = current_skill["frame_duration"][1]

        if self.current_frame_idx >= skill_end_frame:
            # Skill has completed - verbalize it
            logger.info(f"Skill {self.current_skill_idx} completed at frame {self.current_frame_idx}")

            # Get the summary for the completed skill
            summary_text = summarize_skill_annotation(current_skill)

            # Execute the "purge and replace" logic
            self.memory_cache.verbalize_skill(
                completed_skill_idx=self.current_skill_idx,
                summary_json_text=summary_text
            )

            # Print memory stats
            stats = self.memory_cache.get_memory_stats()
            logger.info(f"Memory stats: {stats}")

            # Advance to next skill
            self.current_skill_idx += 1

    def run_episode(
        self,
        observations: list[dict],
        rng: jax.random.PRNGKey
    ) -> list[np.ndarray]:
        """
        Run inference for an entire episode.

        Args:
            observations: List of observation dicts for each frame
            rng: JAX random key

        Returns:
            List of predicted actions
        """
        actions = []

        for obs in observations:
            rng, step_rng = jax.random.split(rng)
            action, _ = self.step(obs, step_rng)
            actions.append(action)

        return actions


def load_model_from_checkpoint(checkpoint_path: Path) -> Pi0Hierarchical:
    """
    Load a trained Pi0Hierarchical model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Loaded model
    """
    # TODO: Implement actual checkpoint loading
    # This is a placeholder
    logger.warning("Model loading not yet implemented - using placeholder")
    return None


def load_episode_observations(
    episode_index: int,
    task_index: int,
    data_root: Path
) -> list[dict]:
    """
    Load observations for an episode from the dataset.

    Args:
        episode_index: Episode index
        task_index: Task index
        data_root: Root directory of the dataset

    Returns:
        List of observation dicts
    """
    # TODO: Implement actual data loading
    # This is a placeholder that creates dummy observations
    logger.warning("Data loading not yet implemented - using placeholder")

    # Create dummy observations for testing
    num_frames = 2000  # Example episode length
    observations = []
    for i in range(num_frames):
        obs = {
            "image": np.random.randn(224, 224, 3),
            "state": np.random.randn(32),
            "timestamp": i,
        }
        observations.append(obs)

    return observations


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Dynamic Interleaved Cache"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--annotation_root",
        type=Path,
        default=Path(os.getenv("DATASET_PATH", Path.cwd() / "dataset")) / "2025-challenge-demos/annotations",
        help="Root directory of skill annotations"
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(os.getenv("DATASET_PATH", Path.cwd() / "dataset")) / "2025-challenge-demos",
        help="Root directory of dataset"
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        default=170,
        help="Episode index to run inference on"
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Task index"
    )
    parser.add_argument(
        "--max_short_term_frames",
        type=int,
        default=10,
        help="Maximum number of visual frames in short-term memory"
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

    # Load model
    logger.info(f"Loading model from {args.checkpoint_path}")
    model = load_model_from_checkpoint(args.checkpoint_path)

    # Construct annotation path
    annotation_path = (
        args.annotation_root /
        f"task-{args.task_index:04d}" /
        f"episode_{args.episode_index:08d}.json"
    )

    if not annotation_path.exists():
        logger.error(f"Annotation file not found: {annotation_path}")
        return

    # Initialize inference loop
    logger.info(f"Initializing inference loop for episode {args.episode_index}")
    inference_loop = DynamicMemoryInferenceLoop(
        model=model,
        annotation_path=annotation_path,
        max_short_term_frames=args.max_short_term_frames
    )

    # Load episode observations
    logger.info("Loading episode observations")
    observations = load_episode_observations(
        args.episode_index,
        args.task_index,
        args.data_root
    )

    # Run inference
    logger.info(f"Running inference on {len(observations)} frames")
    rng = jax.random.PRNGKey(args.seed)
    actions = inference_loop.run_episode(observations, rng)

    logger.info(f"Inference complete. Generated {len(actions)} actions")

    # Print final memory stats
    final_stats = inference_loop.memory_cache.get_memory_stats()
    logger.info(f"Final memory stats: {final_stats}")


if __name__ == "__main__":
    main()

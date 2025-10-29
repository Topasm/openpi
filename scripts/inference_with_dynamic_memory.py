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

Skill Boundary Detection Modes:
    - ground_truth: Use ground-truth annotations (Phase 0/1)
    - eos_token: Detect <EOS_SKILL> token from model predictions (Phase 3)

Usage:
    # Phase 0/1: Ground truth boundary detection
    python scripts/inference_with_dynamic_memory.py \
        --checkpoint_path /path/to/checkpoint \
        --annotation_path /path/to/annotations \
        --episode_index 170 \
        --task_index 0 \
        --boundary_detection ground_truth

    # Phase 3: EOS token boundary detection
    python scripts/inference_with_dynamic_memory.py \
        --checkpoint_path /path/to/checkpoint \
        --episode_index 170 \
        --task_index 0 \
        --boundary_detection eos_token \
        --use_hierarchical_tokenizer
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

# Import EOS_SKILL_TOKEN for Phase 3
try:
    from openpi.models.tokenizer import EOS_SKILL_TOKEN
except ImportError:
    EOS_SKILL_TOKEN = "<EOS_SKILL>"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DynamicMemoryInferenceLoop:
    """
    Inference loop that uses ContextMemory for dynamic interleaved caching.

    This class manages the state during inference and coordinates between:
    - ContextMemory: Maintains the dynamic cache state
    - Model: Predicts actions and skills
    - Environment/Robot: Executes actions and provides observations

    Skill Boundary Detection:
        - ground_truth: Use annotations (Phase 0/1, requires annotation_path)
        - eos_token: Detect <EOS_SKILL> token from model output (Phase 3)
    """

    def __init__(
        self,
        model: Pi0Hierarchical,
        annotation_path: Optional[Path] = None,
        max_short_term_frames: int = 10,
        boundary_detection: str = "ground_truth",
        tokenizer=None
    ):
        """
        Initialize the inference loop.

        Args:
            model: Trained Pi0Hierarchical model
            annotation_path: Path to the skill annotation JSON (required for ground_truth mode)
            max_short_term_frames: Maximum number of visual frames in short-term memory
            boundary_detection: "ground_truth" or "eos_token"
            tokenizer: Tokenizer for decoding model predictions (required for eos_token mode)
        """
        self.model = model
        self.memory_cache = ContextMemory(max_short_term_frames=max_short_term_frames)
        self.boundary_detection = boundary_detection
        self.tokenizer = tokenizer

        # Load ground-truth annotations if using ground_truth mode
        if boundary_detection == "ground_truth":
            if annotation_path is None:
                raise ValueError("annotation_path is required for ground_truth boundary detection")
            with open(annotation_path, 'r') as f:
                self.annotation_data = json.load(f)
            self.all_skills = self.annotation_data["skill_annotation"]
            logger.info(f"Using ground-truth boundary detection with {len(self.all_skills)} skills")
        elif boundary_detection == "eos_token":
            if tokenizer is None:
                raise ValueError("tokenizer is required for eos_token boundary detection")
            self.annotation_data = None
            self.all_skills = []
            logger.info(f"Using EOS token boundary detection with token: {EOS_SKILL_TOKEN}")
        else:
            raise ValueError(f"Unknown boundary_detection mode: {boundary_detection}")

        self.current_skill_idx = 0
        self.current_frame_idx = 0
        self.last_predicted_skill_text = ""  # For EOS detection

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
        # Construct observation for the model
        # The model expects: images, state, tokenized_prompt
        # Since we're using dynamic memory, current_images are already in short-term memory
        # We need to build proper observation from current observation dict

        # Note: This is still a simplified version. In production, you would:
        # 1. Extract images from observation dict
        # 2. Tokenize task prompt
        # 3. Package into _model.Observation format
        # 4. Call model.infer_with_memory()

        if self.model is not None:
            # Call the new inference method with memory
            predicted_action, predicted_skill_text = self.model.infer_with_memory(
                rng=rng,
                observation=observation,  # Pass through raw observation (model will process)
                memory_text=past_text,  # Dynamic memory text
                tokenizer=self.tokenizer,
                generate_skill=(self.boundary_detection == "eos_token"),
                num_steps=10
            )
            # Package skill prediction
            predicted_skill = {"text": predicted_skill_text} if predicted_skill_text else None
        else:
            # Fallback: No model loaded (for testing/demo)
            logger.warning("No model loaded - using dummy predictions")
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
        # For eos_token mode, pass the predicted skill text
        if self.boundary_detection == "eos_token" and predicted_skill is not None:
            # Extract skill text from prediction (assuming predicted_skill is a dict with "text" key)
            if isinstance(predicted_skill, dict) and "text" in predicted_skill:
                predicted_skill_text = predicted_skill["text"]
            elif isinstance(predicted_skill, str):
                predicted_skill_text = predicted_skill
            else:
                predicted_skill_text = None
            self._check_skill_completion(predicted_skill_text=predicted_skill_text)
        else:
            self._check_skill_completion()

        # Increment frame counter
        self.current_frame_idx += 1

        return predicted_action, predicted_skill

    def _check_skill_completion(self, predicted_skill_text: Optional[str] = None):
        """
        Check if the current skill has completed and verbalize if needed.

        Args:
            predicted_skill_text: Model's predicted skill text (for eos_token mode)

        Mode-specific behavior:
            ground_truth: Uses annotations to determine when a skill ends
            eos_token: Detects <EOS_SKILL> token in predicted_skill_text
        """
        skill_completed = False
        summary_text = ""

        if self.boundary_detection == "ground_truth":
            # Phase 0/1: Use ground-truth annotations
            if self.current_skill_idx >= len(self.all_skills):
                return

            current_skill = self.all_skills[self.current_skill_idx]
            skill_end_frame = current_skill["frame_duration"][1]

            if self.current_frame_idx >= skill_end_frame:
                skill_completed = True
                summary_text = summarize_skill_annotation(current_skill)
                logger.info(f"[Ground Truth] Skill {self.current_skill_idx} completed at frame {self.current_frame_idx}")

        elif self.boundary_detection == "eos_token":
            # Phase 3: Detect EOS token from model predictions
            if predicted_skill_text is None:
                return

            # Check if the predicted skill text ends with EOS token
            # Expected format: '{"skill":"move to","obj":"trash_can"} <EOS_SKILL>'
            if predicted_skill_text.strip().endswith(EOS_SKILL_TOKEN):
                skill_completed = True
                # Remove EOS token to get just the JSON
                # Extract: '{"skill":"move to","obj":"trash_can"}'
                summary_text = predicted_skill_text.replace(EOS_SKILL_TOKEN, "").strip()
                logger.info(
                    f"[EOS Detection] Skill {self.current_skill_idx} completed at frame {self.current_frame_idx}"
                )
                logger.info(f"  Skill JSON: {summary_text}")
                logger.info(f"  Will store as: <PAST_SKILL>{summary_text}</PAST_SKILL>")

        if skill_completed:
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
    parser.add_argument(
        "--boundary_detection",
        type=str,
        default="ground_truth",
        choices=["ground_truth", "eos_token"],
        help="Skill boundary detection mode: ground_truth (Phase 0/1) or eos_token (Phase 3)"
    )
    parser.add_argument(
        "--use_hierarchical_tokenizer",
        action="store_true",
        help="Use HierarchicalTokenizer (required for eos_token mode)"
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

    # Initialize tokenizer if needed for eos_token mode
    tokenizer = None
    if args.boundary_detection == "eos_token" or args.use_hierarchical_tokenizer:
        from openpi.models.tokenizer import HierarchicalTokenizer
        tokenizer = HierarchicalTokenizer(max_len=64, add_eos_skill_token=True)
        logger.info("Initialized HierarchicalTokenizer for EOS detection")

    # Construct annotation path (required for ground_truth mode)
    annotation_path = None
    if args.boundary_detection == "ground_truth":
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
    logger.info(f"  Boundary detection mode: {args.boundary_detection}")
    inference_loop = DynamicMemoryInferenceLoop(
        model=model,
        annotation_path=annotation_path,
        max_short_term_frames=args.max_short_term_frames,
        boundary_detection=args.boundary_detection,
        tokenizer=tokenizer
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

"""
Hierarchical transforms for adding skill annotations to training batches.

This module provides transform functions that augment the training data with
skill-level annotations, enabling multi-task hierarchical training where the
model learns to predict both high-level skills and low-level actions.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from openpi.training import skill_utils

logger = logging.getLogger(__name__)


class AddSkillAnnotation:
    """
    Transform that adds skill annotation information to the training batch.

    This transform loads skill annotations for each episode and adds the current
    skill information at each timestep. This enables the model to learn hierarchical
    representations where it predicts both the high-level skill and the low-level action.

    IMPORTANT: For efficiency, we only predict skill text at the beginning of each skill
    (within the first N frames, where N=skill_prediction_window). This avoids redundant
    predictions since the skill remains constant throughout its duration (which can be
    hundreds of frames). The `predict_skill` flag indicates whether this frame should
    contribute to the skill prediction loss.

    Args:
        annotation_root: Root directory containing skill annotation JSON files.
                        Expected structure: {annotation_root}/task-XXXX/episode_XXXXXXXX.json
        cache_annotations: If True, cache loaded annotations in memory to avoid repeated file reads
        skill_format: Format for skill representation - "text" or "dict"
        skill_prediction_window: Number of frames at the start of each skill where we predict
                               the skill text (default: 10). This creates ~1-5% of frames
                               contributing to skill loss, which is efficient but still provides
                               sufficient training signal.
    """

    def __init__(
        self,
        annotation_root: str | Path,
        cache_annotations: bool = True,
        skill_format: str = "text",
        skill_prediction_window: int = 10
    ):
        self.annotation_root = Path(annotation_root)
        self.cache_annotations = cache_annotations
        self.skill_format = skill_format
        self.skill_prediction_window = skill_prediction_window

        if skill_format not in ["text", "dict"]:
            raise ValueError(f"skill_format must be 'text' or 'dict', got {skill_format}")

        # Cache for annotations: episode_index -> annotation_data
        self._annotation_cache = {} if cache_annotations else None

    def __call__(self, data: dict) -> dict:
        """
        Add skill annotation to the data dictionary.

        Args:
            data: Training data dictionary containing at minimum:
                  - "episode_index": int or array with episode index
                  - "index": int or array with frame index within episode
                  - "task_index": int or array with task index

        Returns:
            data: Same dictionary with added fields:
                  - "skill_text": str, text representation of current skill
                  - "skill_dict": dict, dictionary representation of current skill
                  - "skill_idx": int, index of current skill within episode
                  - "has_skill": bool, whether skill annotation is available for this frame
                  - "predict_skill": bool, whether to predict skill at this frame
                                    (True only for first N frames of each skill)
        """
        # Extract episode and frame information
        # These could be scalars (unbatched) or arrays (batched)
        episode_idx = data.get("episode_index")
        frame_idx = data.get("index")  # Frame index within the episode
        task_idx = data.get("task_index")

        if episode_idx is None or frame_idx is None or task_idx is None:
            logger.warning("Missing episode_index, index, or task_index in data. Skipping skill annotation.")
            data["has_skill"] = False
            data["skill_text"] = ""
            data["skill_dict"] = {}
            data["skill_idx"] = -1
            return data

        # Handle batched vs unbatched data
        is_batched = isinstance(episode_idx, np.ndarray) or hasattr(episode_idx, "shape")

        if is_batched:
            # TODO: For batched data, we need to process each item in the batch
            # For now, we'll handle the simple case of single samples
            logger.warning("Batched skill annotation not yet implemented. Using first item only.")
            episode_idx = int(episode_idx.flat[0]) if hasattr(episode_idx, "flat") else int(episode_idx[0])
            frame_idx = int(frame_idx.flat[0]) if hasattr(frame_idx, "flat") else int(frame_idx[0])
            task_idx = int(task_idx.flat[0]) if hasattr(task_idx, "flat") else int(task_idx[0])
        else:
            episode_idx = int(episode_idx)
            frame_idx = int(frame_idx)
            task_idx = int(task_idx)

        # Load annotation for this episode
        try:
            annotation = self._load_annotation(episode_idx, task_idx)

            # Get skill prediction info (checks if within prediction window)
            should_predict, skill = skill_utils.get_skill_prediction_info(
                annotation["skill_annotation"],
                frame_idx,
                window_size=self.skill_prediction_window
            )

            if skill is None:
                # Frame is in a gap between skills (should be rare)
                logger.debug(f"No skill found for episode {episode_idx}, frame {frame_idx}")
                data["has_skill"] = False
                data["predict_skill"] = False
                data["skill_text"] = ""
                data["skill_dict"] = {}
                data["skill_idx"] = -1
            else:
                data["has_skill"] = True
                data["predict_skill"] = should_predict  # NEW: Whether to predict at this frame
                data["skill_text"] = skill_utils.skill_to_text(skill)
                data["skill_dict"] = skill_utils.skill_to_dict(skill)
                data["skill_idx"] = skill["skill_idx"]

        except FileNotFoundError as e:
            logger.warning(f"Annotation not found for episode {episode_idx}: {e}")
            data["has_skill"] = False
            data["predict_skill"] = False
            data["skill_text"] = ""
            data["skill_dict"] = {}
            data["skill_idx"] = -1
        except Exception as e:
            logger.error(f"Error loading skill annotation for episode {episode_idx}: {e}")
            data["has_skill"] = False
            data["predict_skill"] = False
            data["skill_text"] = ""
            data["skill_dict"] = {}
            data["skill_idx"] = -1

        return data

    def _load_annotation(self, episode_idx: int, task_idx: int) -> dict:
        """
        Load skill annotation for an episode, with optional caching.

        Args:
            episode_idx: Episode index (8-digit identifier)
            task_idx: Task index (4-digit identifier)

        Returns:
            Annotation dictionary
        """
        # Check cache first
        if self._annotation_cache is not None and episode_idx in self._annotation_cache:
            return self._annotation_cache[episode_idx]

        # Construct path: annotation_root/task-TTTT/episode_EEEEEEEE.json
        episode_name = f"episode_{episode_idx:08d}"
        annotation_path = self.annotation_root / f"task-{task_idx:04d}" / f"{episode_name}.json"

        # Load annotation
        annotation = skill_utils.load_skill_annotation(annotation_path)

        # Cache if enabled
        if self._annotation_cache is not None:
            self._annotation_cache[episode_idx] = annotation

        return annotation


class AddPastSkillsCache:
    """
    Transform that adds a cache of past completed skills for long-horizon memory.

    This is used in Phase 1 to implement the ProVideLLM-style interleaved cache
    where past skills are represented as text tokens instead of full video frames.

    Args:
        annotation_root: Root directory containing skill annotation JSON files
        cache_annotations: If True, cache loaded annotations in memory
        format_with_tokens: If True, wrap skill text with special tokens (e.g., "<L> ... </L>")
        special_token: Special token to use for wrapping (default: "<L>")
        include_current_skill: If True, include the current skill in the cache (for debugging)
    """

    def __init__(
        self,
        annotation_root: str | Path,
        cache_annotations: bool = True,
        format_with_tokens: bool = True,
        special_token: str = "<L>",
        include_current_skill: bool = False
    ):
        self.annotation_root = Path(annotation_root)
        self.cache_annotations = cache_annotations
        self.format_with_tokens = format_with_tokens
        self.special_token = special_token
        self.include_current_skill = include_current_skill

        # Cache for annotations
        self._annotation_cache = {} if cache_annotations else None

    def __call__(self, data: dict) -> dict:
        """
        Add past skills cache to the data dictionary.

        Args:
            data: Training data dictionary

        Returns:
            data: Same dictionary with added fields:
                  - "past_skills_cache": list of str, text representations of past skills
                  - "num_past_skills": int, number of past skills in the cache
        """
        episode_idx = data.get("episode_index")
        skill_idx = data.get("skill_idx", -1)
        task_idx = data.get("task_index")

        if episode_idx is None or skill_idx == -1 or task_idx is None:
            data["past_skills_cache"] = []
            data["num_past_skills"] = 0
            return data

        # Handle batched vs unbatched
        is_batched = isinstance(episode_idx, np.ndarray) or hasattr(episode_idx, "shape")
        if is_batched:
            episode_idx = int(episode_idx.flat[0]) if hasattr(episode_idx, "flat") else int(episode_idx[0])
            skill_idx = int(skill_idx.flat[0]) if hasattr(skill_idx, "flat") else int(skill_idx[0])
            task_idx = int(task_idx.flat[0]) if hasattr(task_idx, "flat") else int(task_idx[0])
        else:
            episode_idx = int(episode_idx)
            skill_idx = int(skill_idx)
            task_idx = int(task_idx)

        try:
            annotation = self._load_annotation(episode_idx, task_idx)
            past_skills = skill_utils.create_skill_cache(
                annotation["skill_annotation"],
                skill_idx,
                include_current=self.include_current_skill
            )

            if self.format_with_tokens:
                past_skills = [
                    skill_utils.format_skill_with_special_tokens(skill, self.special_token)
                    for skill in past_skills
                ]

            data["past_skills_cache"] = past_skills
            data["num_past_skills"] = len(past_skills)

        except Exception as e:
            logger.error(f"Error creating past skills cache for episode {episode_idx}: {e}")
            data["past_skills_cache"] = []
            data["num_past_skills"] = 0

        return data

    def _load_annotation(self, episode_idx: int, task_idx: int) -> dict:
        """Load skill annotation for an episode, with optional caching."""
        if self._annotation_cache is not None and episode_idx in self._annotation_cache:
            return self._annotation_cache[episode_idx]

        episode_name = f"episode_{episode_idx:08d}"
        annotation_path = self.annotation_root / f"task-{task_idx:04d}" / f"{episode_name}.json"
        annotation = skill_utils.load_skill_annotation(annotation_path)

        if self._annotation_cache is not None:
            self._annotation_cache[episode_idx] = annotation

        return annotation


class TokenizeSkills:
    """
    Transform that tokenizes skill text into token IDs.

    This transform takes skill_text from the batch and converts it to
    skill_tokens and skill_masks using the PaliGemma tokenizer.

    Args:
        tokenizer: PaliGemma tokenizer instance (optional, will create default if None)
        max_len: Maximum length for skill tokens (default: 64)
    """

    def __init__(self, tokenizer=None, max_len: int = 64):
        if tokenizer is None:
            # Create default tokenizer
            from openpi.models.tokenizer import PaligemmaTokenizer
            self.tokenizer = PaligemmaTokenizer(max_len=max_len)
        else:
            self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, data: dict) -> dict:
        """
        Tokenize skill text.

        Args:
            data: Data dictionary with "skill_text" field

        Returns:
            data: Same dictionary with added fields:
                  - "skill_tokens": [skill_len] integer token IDs
                  - "skill_mask": [skill_len] boolean mask
        """
        if "skill_text" in data and data.get("has_skill", False):
            skill_text = data["skill_text"]

            # Tokenize using PaliGemma tokenizer
            tokens, mask = self.tokenizer.tokenize(skill_text)

            data["skill_tokens"] = tokens
            data["skill_mask"] = mask
        else:
            # No skill available - create dummy tokens
            import numpy as np
            data["skill_tokens"] = np.zeros(self.max_len, dtype=np.int32)
            data["skill_mask"] = np.zeros(self.max_len, dtype=np.bool_)

        return data


class TokenizeMemory:
    """
    Tokenizes past skills memory cache for Phase 1 (long-horizon memory).

    This transform takes the past_skills_cache (list of skill text with <L> tokens)
    and tokenizes it into a single flat sequence for input to the model.

    Example input:
        past_skills_cache = ["<L> skill0 </L>", "<L> skill1 </L>"]

    Example output:
        memory_tokens = [257001, 1234, 5678, 257002, 257001, 2345, 6789, 257002, ...]
        memory_mask = [True, True, True, True, True, True, True, True, ...]

    Args:
        tokenizer: PaliGemma tokenizer (if None, creates default)
        max_memory_len: Maximum number of memory tokens (default: 256)
    """

    def __init__(self, tokenizer=None, max_memory_len: int = 256):
        if tokenizer is None:
            from openpi.models.tokenizer import PaligemmaTokenizer
            self.tokenizer = PaligemmaTokenizer(max_len=max_memory_len)
        else:
            self.tokenizer = tokenizer
        self.max_memory_len = max_memory_len

    def __call__(self, data: dict) -> dict:
        """
        Tokenize past skills memory cache.

        Args:
            data: Dictionary with optional "past_skills_cache" field

        Returns:
            data with added fields:
            - "memory_tokens": [memory_len] integer token IDs
            - "memory_mask": [memory_len] boolean mask
        """
        if "past_skills_cache" in data and data.get("num_past_skills", 0) > 0:
            # Concatenate all memory items into single string
            memory_text = " ".join(data["past_skills_cache"])

            # Tokenize
            tokens, mask = self.tokenizer.tokenize(memory_text)

            # Truncate if too long
            if len(tokens) > self.max_memory_len:
                tokens = tokens[:self.max_memory_len]
                mask = mask[:self.max_memory_len]

            # Pad if too short
            elif len(tokens) < self.max_memory_len:
                import numpy as np
                pad_len = self.max_memory_len - len(tokens)
                tokens = np.concatenate([tokens, np.zeros(pad_len, dtype=tokens.dtype)])
                mask = np.concatenate([mask, np.zeros(pad_len, dtype=mask.dtype)])

            data["memory_tokens"] = tokens
            data["memory_mask"] = mask
        else:
            # No memory - create empty placeholders
            import numpy as np
            data["memory_tokens"] = np.zeros(self.max_memory_len, dtype=np.int32)
            data["memory_mask"] = np.zeros(self.max_memory_len, dtype=np.bool_)

        return data


class CreateDynamicMemoryBatch:
    """
    Transform that creates a dynamic memory batch for "Dynamic Interleaved Cache" training.

    This is an advanced transform for Phase 2 (Dynamic Memory) that simulates the
    inference-time memory state. Unlike Phase 1's static prepended cache, this creates
    a time-ordered interleaved cache that matches what the model sees during inference.

    For each training frame at time t within skill k:
    1. Long-term memory (text): Summaries of all completed skills (0 to k-1)
    2. Short-term memory (vision): Recent visual frames from current skill k

    This ensures the model learns to use the same input format during training
    as it will receive from the ContextMemory manager during inference.

    Args:
        annotation_root: Root directory containing skill annotation JSON files
        max_short_term_frames: Maximum number of visual frames to include (default: 10)
        cache_annotations: If True, cache loaded annotations in memory
        special_token: Special token to wrap past skills (default: "<PAST_SKILL>")
    """

    def __init__(
        self,
        annotation_root: str | Path,
        max_short_term_frames: int = 10,
        cache_annotations: bool = True,
        special_token: str = "<PAST_SKILL>"
    ):
        self.annotation_root = Path(annotation_root)
        self.max_short_term_frames = max_short_term_frames
        self.cache_annotations = cache_annotations
        self.special_token = special_token

        # Cache for annotations
        self._annotation_cache = {} if cache_annotations else None

    def __call__(self, data: dict) -> dict:
        """
        Create dynamic memory batch for a training sample.

        Args:
            data: Training data dictionary containing:
                  - "episode_index": Episode index
                  - "index": Frame index within episode
                  - "task_index": Task index
                  - "skill_idx": Current skill index (from AddSkillAnnotation)
                  - "observation.images.*": Image observations

        Returns:
            data with added fields:
            - "dynamic_memory_text": str, concatenated past skill summaries
            - "dynamic_memory_images": array, stacked visual frames from current skill
            - "num_past_skills": int, number of skills in long-term memory
            - "num_current_frames": int, number of frames in short-term memory
        """
        episode_idx = data.get("episode_index")
        frame_idx = data.get("index")
        task_idx = data.get("task_index")
        skill_idx = data.get("skill_idx", -1)

        # Handle batched vs unbatched
        is_batched = isinstance(episode_idx, np.ndarray) or hasattr(episode_idx, "shape")
        if is_batched:
            logger.warning("Batched dynamic memory not yet fully implemented. Using first item.")
            episode_idx = int(episode_idx.flat[0]) if hasattr(episode_idx, "flat") else int(episode_idx[0])
            frame_idx = int(frame_idx.flat[0]) if hasattr(frame_idx, "flat") else int(frame_idx[0])
            task_idx = int(task_idx.flat[0]) if hasattr(task_idx, "flat") else int(task_idx[0])
            skill_idx = int(skill_idx.flat[0]) if hasattr(skill_idx, "flat") else int(skill_idx[0])
        else:
            episode_idx = int(episode_idx)
            frame_idx = int(frame_idx)
            task_idx = int(task_idx)
            skill_idx = int(skill_idx)

        if skill_idx == -1 or not data.get("has_skill", False):
            # No skill annotation - return empty memory
            data["dynamic_memory_text"] = ""
            data["dynamic_memory_images"] = np.array([])
            data["num_past_skills"] = 0
            data["num_current_frames"] = 0
            return data

        try:
            # Load annotation for this episode
            annotation = self._load_annotation(episode_idx, task_idx)
            all_skills = annotation["skill_annotation"]
            current_skill = all_skills[skill_idx]

            # 1. Generate long-term memory: text summaries for completed skills
            past_skill_texts = []
            for skill in all_skills[:skill_idx]:  # All skills before current
                summary_text = skill_utils.skill_to_text(skill)
                past_skill_texts.append(f"{self.special_token} {summary_text} </{self.special_token[1:]}")

            # Concatenate into single text string
            dynamic_memory_text = " ".join(past_skill_texts)

            # 2. Generate short-term memory: visual frames from current skill
            # Note: This is a simplified version that just passes through current frame
            # In a full implementation, you would load multiple frames from the skill
            # For now, we use the current observation image
            # TODO: Implement loading multiple frames from skill start to current frame

            # For the simplified version, we'll store metadata that can be used
            # by subsequent transforms
            skill_start_frame = current_skill["frame_duration"][0]
            skill_end_frame = current_skill["frame_duration"][1]

            # Calculate which frames to include (sub-sampling logic)
            frames_in_skill = frame_idx - skill_start_frame + 1
            if frames_in_skill > self.max_short_term_frames:
                # We would need to sub-sample frames, but since we're working with
                # single-frame batches, we'll just pass through metadata
                num_current_frames = self.max_short_term_frames
            else:
                num_current_frames = frames_in_skill

            # Store metadata for potential use by other transforms
            data["dynamic_memory_text"] = dynamic_memory_text
            data["num_past_skills"] = len(past_skill_texts)
            data["num_current_frames"] = num_current_frames
            data["skill_start_frame"] = skill_start_frame
            data["skill_end_frame"] = skill_end_frame
            data["frames_into_skill"] = frames_in_skill

            logger.debug(
                f"Created dynamic memory: {len(past_skill_texts)} past skills, "
                f"{num_current_frames} current frames"
            )

        except Exception as e:
            logger.error(f"Error creating dynamic memory batch for episode {episode_idx}: {e}")
            data["dynamic_memory_text"] = ""
            data["dynamic_memory_images"] = np.array([])
            data["num_past_skills"] = 0
            data["num_current_frames"] = 0

        return data

    def _load_annotation(self, episode_idx: int, task_idx: int) -> dict:
        """Load skill annotation for an episode, with optional caching."""
        if self._annotation_cache is not None and episode_idx in self._annotation_cache:
            return self._annotation_cache[episode_idx]

        episode_name = f"episode_{episode_idx:08d}"
        annotation_path = self.annotation_root / f"task-{task_idx:04d}" / f"{episode_name}.json"
        annotation = skill_utils.load_skill_annotation(annotation_path)

        if self._annotation_cache is not None:
            self._annotation_cache[episode_idx] = annotation

        return annotation


class TokenizeDynamicMemory:
    """
    Tokenizes dynamic memory text for the model input.

    This converts the dynamic_memory_text (past skill summaries) into tokens
    that can be fed to the model's memory embedding layer.

    Args:
        tokenizer: PaliGemma tokenizer (if None, creates default)
        max_memory_len: Maximum number of memory tokens (default: 256)
    """

    def __init__(self, tokenizer=None, max_memory_len: int = 256):
        if tokenizer is None:
            from openpi.models.tokenizer import PaligemmaTokenizer
            self.tokenizer = PaligemmaTokenizer(max_len=max_memory_len)
        else:
            self.tokenizer = tokenizer
        self.max_memory_len = max_memory_len

    def __call__(self, data: dict) -> dict:
        """
        Tokenize dynamic memory text.

        Args:
            data: Dictionary with "dynamic_memory_text" field

        Returns:
            data with added fields:
            - "dynamic_memory_tokens": [memory_len] integer token IDs
            - "dynamic_memory_mask": [memory_len] boolean mask
        """
        if "dynamic_memory_text" in data and data.get("num_past_skills", 0) > 0:
            memory_text = data["dynamic_memory_text"]

            # Tokenize
            tokens, mask = self.tokenizer.tokenize(memory_text)

            # Truncate if too long
            if len(tokens) > self.max_memory_len:
                tokens = tokens[:self.max_memory_len]
                mask = mask[:self.max_memory_len]

            # Pad if too short
            elif len(tokens) < self.max_memory_len:
                pad_len = self.max_memory_len - len(tokens)
                tokens = np.concatenate([tokens, np.zeros(pad_len, dtype=tokens.dtype)])
                mask = np.concatenate([mask, np.zeros(pad_len, dtype=mask.dtype)])

            data["dynamic_memory_tokens"] = tokens
            data["dynamic_memory_mask"] = mask
        else:
            # No memory - create empty placeholders
            data["dynamic_memory_tokens"] = np.zeros(self.max_memory_len, dtype=np.int32)
            data["dynamic_memory_mask"] = np.zeros(self.max_memory_len, dtype=np.bool_)

        return data


# Example usage
if __name__ == "__main__":
    # Test the transform with example data (uses DATASET_PATH environment variable)
    import os
    from pathlib import Path
    dataset_base = Path(os.getenv("DATASET_PATH", Path.cwd() / "dataset"))
    annotation_root = str(dataset_base / "2025-challenge-demos/annotations")

    # Create transform
    transform = AddSkillAnnotation(annotation_root)

    # Test data (simulating what comes from the dataset)
    test_data = {
        "episode_index": 170,  # episode_00000170
        "index": 1000,  # Frame 1000
        "task_index": 0,  # Task 0
        "actions": np.random.randn(50, 32),  # Dummy actions
    }

    print("Before transform:")
    print(f"  Keys: {test_data.keys()}")

    # Apply transform
    transformed_data = transform(test_data)

    print("\nAfter transform:")
    print(f"  Keys: {transformed_data.keys()}")
    print(f"  has_skill: {transformed_data['has_skill']}")
    print(f"  skill_idx: {transformed_data['skill_idx']}")
    print(f"  skill_text: {transformed_data['skill_text']}")
    print(f"  skill_dict: {transformed_data['skill_dict']}")

    # Test past skills cache
    print("\n" + "="*80)
    print("Testing Past Skills Cache Transform")
    print("="*80)

    past_skills_transform = AddPastSkillsCache(annotation_root)
    transformed_data = past_skills_transform(transformed_data)

    print(f"\n  num_past_skills: {transformed_data['num_past_skills']}")
    print(f"  past_skills_cache:")
    for i, skill in enumerate(transformed_data['past_skills_cache']):
        print(f"    {i}: {skill}")

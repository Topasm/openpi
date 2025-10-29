"""
Dynamic Interleaved Cache system for hierarchical VLA training.

This module implements the advanced "Dynamic Interleaved Cache" memory system
that separates memory into "long-term" (text) and "short-term" (vision) blocks.
It "verbalizes" short-term memories into long-term memories upon skill completion.

The system consists of:
1. Memory Blocks: LongTermBlock (text) and ShortTermBlock (vision)
2. ContextMemory: State manager for dynamic memory
3. Helper functions for memory assembly and verbalization
"""

import dataclasses
import logging
from pathlib import Path
from typing import List, Optional, Union

import jax.numpy as jnp
import numpy as np

from openpi.training import skill_utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LongTermBlock:
    """
    Long-term memory block: A text summary of a completed skill.

    This represents "what was done" - a compressed semantic representation
    of a past skill that has been completed.

    Attributes:
        skill_idx: Index of the skill this block represents
        summary_text: Text summary (e.g., "<PAST_SKILL>{'skill':'move to',...}</PAST_SKILL>")
    """
    skill_idx: int
    summary_text: str


@dataclasses.dataclass
class ShortTermBlock:
    """
    Short-term memory block: Visual frames from the currently executing skill.

    This represents "how it is being done" - raw visual observations
    from the skill that is currently in progress.

    Attributes:
        skill_idx: Index of the skill this frame belongs to
        frame_timestamp: Frame index within the episode (for identification/purging)
        visual_features: Image data or encoded visual features
    """
    skill_idx: int
    frame_timestamp: int
    visual_features: np.ndarray  # Image array or encoded features


class ContextMemory:
    """
    State manager for dynamic interleaved memory.

    This class manages a time-ordered list of memory blocks (both long-term text
    and short-term vision). It provides the "verbalization" logic that converts
    completed skills from visual frames to text summaries.

    Usage:
        # Initialize
        memory = ContextMemory(max_short_term_frames=10)

        # During inference loop:
        for frame in episode:
            # Add visual frame to short-term memory
            memory.add_visual_frame(frame_data, current_skill_idx, timestamp)

            # Assemble model input
            text_input, image_input = memory.assemble_model_input()

            # When skill completes:
            if skill_completed:
                memory.verbalize_skill(completed_skill_idx, summary_json_text)

    Attributes:
        memory: Time-ordered list of LongTermBlock and ShortTermBlock
        max_short_term_frames: Maximum number of visual frames to keep
    """

    def __init__(self, max_short_term_frames: int = 10):
        """
        Initialize the context memory manager.

        Args:
            max_short_term_frames: Maximum number of visual frames to keep in memory
                                   (to limit context size). Default: 10 frames.
        """
        self.memory: List[Union[LongTermBlock, ShortTermBlock]] = []
        self.max_short_term_frames = max_short_term_frames

    def add_visual_frame(
        self,
        frame_data: np.ndarray,
        current_skill_idx: int,
        timestamp: int
    ):
        """
        Add a visual frame to short-term memory.

        This is called during inference for each new observation frame.

        Args:
            frame_data: Image array or visual features
            current_skill_idx: Index of the skill currently being executed
            timestamp: Frame index within the episode
        """
        new_block = ShortTermBlock(
            skill_idx=current_skill_idx,
            frame_timestamp=timestamp,
            visual_features=frame_data
        )
        self.memory.append(new_block)

        # Optionally prune oldest frames if we exceed the limit
        self._purge_oldest_short_term_memory()

    def verbalize_skill(self, completed_skill_idx: int, summary_json_text: str):
        """
        "Verbalize" a completed skill: purge visual frames and replace with text summary.

        This is the core "purge-and-replace" logic. When a skill completes:
        1. Remove all ShortTermBlock (vision) for the completed skill
        2. Add a single LongTermBlock (text summary) for that skill

        Args:
            completed_skill_idx: Index of the skill that just completed
            summary_json_text: Compact JSON text summary of the skill
                              (e.g., '{"skill":"move to","obj":["radio_89"],...]')
        """
        # Create the new long-term text summary block
        summary_block = LongTermBlock(
            skill_idx=completed_skill_idx,
            summary_text=f"<PAST_SKILL>{summary_json_text}</PAST_SKILL>"
        )

        # Purge: Filter out all vision frames from the completed skill
        new_memory_list = []
        for block in self.memory:
            if isinstance(block, ShortTermBlock) and block.skill_idx == completed_skill_idx:
                # Discard the ShortTermBlock (vision) for the completed skill
                continue
            new_memory_list.append(block)

        # Replace: Append the new summary text block
        new_memory_list.append(summary_block)

        # Atomically update the memory
        self.memory = new_memory_list

        logger.info(
            f"Verbalized skill {completed_skill_idx}: "
            f"Replaced visual frames with text summary"
        )

    def assemble_model_input(self) -> tuple[str, Optional[np.ndarray]]:
        """
        Assemble the current memory state into model input format.

        This creates the input format expected by the model:
        - All LongTermBlock (past skills) → concatenated text
        - All ShortTermBlock (current skill) → stacked images

        Returns:
            (text_input, image_input) tuple:
                - text_input: Concatenated text from all long-term blocks
                - image_input: Stacked images from all short-term blocks (or None if empty)
        """
        # Extract all long-term text blocks
        past_skill_texts = [
            block.summary_text
            for block in self.memory
            if isinstance(block, LongTermBlock)
        ]

        # Extract all short-term visual blocks
        current_skill_images = [
            block.visual_features
            for block in self.memory
            if isinstance(block, ShortTermBlock)
        ]

        # Assemble final input
        # Format: [All past skill text concatenated] + [All current skill images stacked]
        final_text_input = " ".join(past_skill_texts)
        final_image_input = np.stack(current_skill_images) if current_skill_images else None

        return final_text_input, final_image_input

    def _purge_oldest_short_term_memory(self):
        """
        Prune oldest short-term visual frames if we exceed max_short_term_frames.

        This keeps memory usage bounded by discarding the oldest visual frames
        when we have too many.
        """
        # Count current short-term blocks
        short_term_blocks = [b for b in self.memory if isinstance(b, ShortTermBlock)]

        if len(short_term_blocks) > self.max_short_term_frames:
            # Remove oldest short-term blocks
            num_to_remove = len(short_term_blocks) - self.max_short_term_frames

            new_memory = []
            removed_count = 0
            for block in self.memory:
                if isinstance(block, ShortTermBlock) and removed_count < num_to_remove:
                    # Skip this block (remove it)
                    removed_count += 1
                    continue
                new_memory.append(block)

            self.memory = new_memory
            logger.debug(f"Pruned {removed_count} oldest visual frames from memory")

    def get_memory_stats(self) -> dict:
        """
        Get statistics about current memory state.

        Returns:
            Dictionary with memory statistics:
                - num_long_term_blocks: Number of text summary blocks
                - num_short_term_blocks: Number of visual frame blocks
                - total_blocks: Total number of blocks
                - short_term_skill_indices: Set of skill indices in short-term memory
        """
        long_term_blocks = [b for b in self.memory if isinstance(b, LongTermBlock)]
        short_term_blocks = [b for b in self.memory if isinstance(b, ShortTermBlock)]

        return {
            "num_long_term_blocks": len(long_term_blocks),
            "num_short_term_blocks": len(short_term_blocks),
            "total_blocks": len(self.memory),
            "short_term_skill_indices": set(b.skill_idx for b in short_term_blocks),
            "long_term_skill_indices": set(b.skill_idx for b in long_term_blocks),
        }

    def clear(self):
        """Clear all memory blocks."""
        self.memory = []
        logger.debug("Cleared all memory blocks")


# Helper functions for creating memory summaries

def summarize_skill_annotation(skill: dict) -> str:
    """
    Create a compact JSON text summary of a skill annotation.

    This is the function used to create the text summary when "verbalizing"
    a completed skill.

    Args:
        skill: Skill annotation dictionary

    Returns:
        Compact JSON string representation of the skill

    Example:
        >>> skill = {
        ...     "skill_description": ["move to"],
        ...     "object_id": [["radio_89"]],
        ...     "manipulating_object_id": [""],
        ...     "skill_type": ["uncoordinated"]
        ... }
        >>> summarize_skill_annotation(skill)
        '{"skill":"move to","obj":["radio_89"],"manip":"","type":"uncoordinated"}'
    """
    # Reuse existing skill_to_text utility
    return skill_utils.skill_to_text(skill)


def create_dynamic_memory_state(
    all_skills: List[dict],
    current_frame_idx: int,
    current_skill_idx: int,
    frames_since_skill_start: List[np.ndarray],
    max_short_term_frames: int = 10
) -> tuple[List[str], List[np.ndarray]]:
    """
    Simulate the dynamic memory state for a given training frame.

    This function reconstructs what the ContextMemory state would look like
    during inference at a specific frame. Used by the data pipeline to create
    training batches that match the inference input format.

    Args:
        all_skills: List of all skill annotations for the episode
        current_frame_idx: The frame index we're generating data for
        current_skill_idx: Index of the skill at current_frame_idx
        frames_since_skill_start: List of visual frames from skill start to current frame
        max_short_term_frames: Maximum number of visual frames to include

    Returns:
        (past_skill_texts, current_skill_images) tuple:
            - past_skill_texts: List of text summaries for completed skills
            - current_skill_images: List of visual frames for current skill

    Example:
        >>> # At frame 1000 of skill 2:
        >>> past_texts, current_imgs = create_dynamic_memory_state(
        ...     all_skills=episode_skills,
        ...     current_frame_idx=1000,
        ...     current_skill_idx=2,
        ...     frames_since_skill_start=[frame_265, frame_266, ..., frame_1000]
        ... )
        >>> # past_texts contains summaries of skills 0 and 1
        >>> # current_imgs contains recent frames from skill 2
    """
    # Generate long-term memory: text summaries for all completed skills
    past_skill_texts = []
    for skill in all_skills[:current_skill_idx]:  # All skills before current
        summary_text = summarize_skill_annotation(skill)
        past_skill_texts.append(f"<PAST_SKILL>{summary_text}</PAST_SKILL>")

    # Generate short-term memory: visual frames from current skill
    # Sub-sample if we have too many frames
    if len(frames_since_skill_start) > max_short_term_frames:
        # Evenly sub-sample frames
        indices = np.linspace(
            0,
            len(frames_since_skill_start) - 1,
            max_short_term_frames,
            dtype=int
        )
        current_skill_images = [frames_since_skill_start[i] for i in indices]
    else:
        current_skill_images = frames_since_skill_start

    return past_skill_texts, current_skill_images


# Example usage and testing
if __name__ == "__main__":
    import json

    # Test ContextMemory
    print("="*80)
    print("Testing ContextMemory")
    print("="*80)

    memory = ContextMemory(max_short_term_frames=5)

    # Simulate adding frames for skill 0
    print("\n1. Adding frames for skill 0...")
    for i in range(10):
        frame = np.random.randn(224, 224, 3)
        memory.add_visual_frame(frame, skill_idx=0, timestamp=i)

    stats = memory.get_memory_stats()
    print(f"   Memory stats: {stats}")

    # Assemble input
    text, images = memory.assemble_model_input()
    print(f"   Text input: '{text}'")
    print(f"   Image input shape: {images.shape if images is not None else None}")

    # Verbalize skill 0
    print("\n2. Skill 0 completed, verbalizing...")
    skill_0_summary = '{"skill":"move to","obj":["table"],"manip":"","type":"uncoordinated"}'
    memory.verbalize_skill(0, skill_0_summary)

    stats = memory.get_memory_stats()
    print(f"   Memory stats after verbalization: {stats}")

    # Assemble input again
    text, images = memory.assemble_model_input()
    print(f"   Text input: '{text}'")
    print(f"   Image input: {images}")  # Should be None now

    # Add frames for skill 1
    print("\n3. Adding frames for skill 1...")
    for i in range(3):
        frame = np.random.randn(224, 224, 3)
        memory.add_visual_frame(frame, skill_idx=1, timestamp=10+i)

    stats = memory.get_memory_stats()
    print(f"   Memory stats: {stats}")

    # Assemble input
    text, images = memory.assemble_model_input()
    print(f"   Text input: '{text}'")
    print(f"   Image input shape: {images.shape if images is not None else None}")

    # Verbalize skill 1
    print("\n4. Skill 1 completed, verbalizing...")
    skill_1_summary = '{"skill":"pick up","obj":["cup"],"manip":"cup","type":"coordinated"}'
    memory.verbalize_skill(1, skill_1_summary)

    stats = memory.get_memory_stats()
    print(f"   Memory stats after verbalization: {stats}")

    # Assemble final input
    text, images = memory.assemble_model_input()
    print(f"   Text input: '{text}'")
    print(f"   Image input: {images}")  # Should be None

    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)

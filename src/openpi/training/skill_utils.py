"""
Utilities for loading and processing skill annotations for hierarchical VLA training.

This module provides functions to load skill annotations from BEHAVIOR-1K dataset
and convert them to format suitable for multi-task hierarchical training.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_skill_annotation(annotation_path: str | Path) -> Dict[str, Any]:
    """
    Load skill annotation JSON file for a given episode.

    Args:
        annotation_path: Path to the annotation JSON file

    Returns:
        Dictionary containing task_name, skill_annotation, and primitive_annotation
    """
    annotation_path = Path(annotation_path)

    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)

    return annotation_data


def get_skill_at_frame(
    skill_annotations: List[Dict[str, Any]],
    frame_idx: int
) -> Optional[Dict[str, Any]]:
    """
    Get the skill annotation for a specific frame index.

    Args:
        skill_annotations: List of skill annotation dictionaries
        frame_idx: Frame index to query

    Returns:
        Skill annotation dict if frame is within any skill's duration, else None
    """
    for skill in skill_annotations:
        start_frame, end_frame = skill["frame_duration"]
        if start_frame <= frame_idx < end_frame:
            return skill

    return None


def _clean_object_id(obj_id_list: List[List[str]]) -> List[str]:
    """
    Remove trailing numbers from object IDs for concise representation.

    This converts object IDs like "trash_can_123" to "trash_can" for
    a more compact and generalizable skill representation.

    Args:
        obj_id_list: Nested list of object IDs from skill annotation

    Returns:
        List of cleaned object ID strings (without trailing numbers)

    Example:
        Input: [["radio_89", "coffee_table_koagbh_0"]]
        Output: ["radio", "coffee_table_koagbh"]
    """
    if not obj_id_list or not obj_id_list[0]:
        return []

    cleaned_objects = []
    for obj_id in obj_id_list[0]:
        # Remove trailing underscore followed by numbers
        cleaned_obj = re.sub(r'_\d+$', '', obj_id)
        cleaned_objects.append(cleaned_obj)

    return cleaned_objects


def create_concise_skill_summary_json(skill_annotation: Dict[str, Any]) -> str:
    """
    Create a concise JSON string summary for dense skill prediction (Phase 3).

    This creates an ultra-compact representation without trailing numbers in object IDs,
    suitable for prediction at every frame. The format is more concise than skill_to_text()
    to reduce token count during dense prediction.

    Args:
        skill_annotation: Skill annotation dictionary

    Returns:
        Concise JSON string without spaces

    Example:
        Input: {
            "skill_description": ["move to"],
            "object_id": [["trash_can_123"]],
            "skill_type": ["navigation"]
        }
        Output: '{"skill":"move to","obj":"trash_can","type":"navigation"}'

    Note:
        - Only includes skill, obj (single primary object), and type
        - Removes trailing numbers from object IDs (trash_can_123 -> trash_can)
        - No manip, mem, spatial fields for maximum conciseness
    """
    summary_dict = {}

    # Add skill description
    if "skill_description" in skill_annotation and skill_annotation["skill_description"]:
        summary_dict["skill"] = skill_annotation["skill_description"][0]

    # Add primary object only (cleaned of trailing numbers)
    if "object_id" in skill_annotation and skill_annotation["object_id"]:
        cleaned_objs = _clean_object_id(skill_annotation["object_id"])
        if cleaned_objs:
            # Use only the first object for maximum conciseness
            summary_dict["obj"] = cleaned_objs[0]

    # Add skill type
    if "skill_type" in skill_annotation and skill_annotation["skill_type"]:
        summary_dict["type"] = skill_annotation["skill_type"][0]

    # Convert to JSON string (compact, no whitespace)
    return json.dumps(summary_dict, separators=(',', ':'))


def skill_to_text(skill: Dict[str, Any]) -> str:
    """
    Convert skill annotation dictionary to a text string for tokenization.

    This creates a compact, JSON-like representation that the model will learn to predict.

    Args:
        skill: Skill annotation dictionary

    Returns:
        Text representation of the skill

    Example:
        Input: {
            "skill_idx": 1,
            "skill_id": [2],
            "skill_description": ["pick up from"],
            "object_id": [["radio_89", "coffee_table_koagbh_0"]],
            "manipulating_object_id": ["radio_89"],
            "skill_type": ["uncoordinated"]
        }
        Output: '{"skill":"pick up from","obj":["radio_89","coffee_table_koagbh_0"],"manip":"radio_89","type":"uncoordinated"}'
    """
    # Create a compact representation focusing on the most important fields
    compact_skill = {
        "skill": skill["skill_description"][0] if skill["skill_description"] else "",
        "obj": skill["object_id"][0] if skill["object_id"] else [],
        "manip": skill["manipulating_object_id"][0] if skill["manipulating_object_id"] else "",
        "type": skill["skill_type"][0] if skill["skill_type"] else "",
    }

    # Add optional fields if present
    if skill.get("memory_prefix") and len(skill["memory_prefix"]) > 0:
        compact_skill["mem"] = skill["memory_prefix"][0]

    if skill.get("spatial_prefix") and len(skill["spatial_prefix"]) > 0:
        compact_skill["spatial"] = skill["spatial_prefix"][0]

    # Convert to JSON string (compact, no whitespace)
    return json.dumps(compact_skill, separators=(',', ':'))


def skill_to_dict(skill: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert skill annotation to a compact dictionary representation.

    This is an alternative to skill_to_text that returns a dictionary
    which can be serialized differently depending on the use case.

    Args:
        skill: Skill annotation dictionary

    Returns:
        Compact dictionary representation of the skill
    """
    compact_skill = {
        "skill_idx": skill["skill_idx"],
        "skill": skill["skill_description"][0] if skill["skill_description"] else "",
        "objects": skill["object_id"][0] if skill["object_id"] else [],
        "manipulating": skill["manipulating_object_id"][0] if skill["manipulating_object_id"] else "",
        "skill_type": skill["skill_type"][0] if skill["skill_type"] else "",
    }

    if skill.get("memory_prefix") and len(skill["memory_prefix"]) > 0:
        compact_skill["memory"] = skill["memory_prefix"][0]

    if skill.get("spatial_prefix") and len(skill["spatial_prefix"]) > 0:
        compact_skill["spatial"] = skill["spatial_prefix"][0]

    return compact_skill


def get_annotation_path_from_episode_name(
    episode_name: str,
    annotation_root: str | Path
) -> Path:
    """
    Construct the annotation file path from episode name and annotation root.

    Episode names are in format: episode_TTTTIIIT where
    - TTTT is the task index (4 digits)
    - III is the instance id (3 digits)
    - T is the trajectory id (1 digit)

    Args:
        episode_name: Episode identifier (e.g., "episode_00000170")
        annotation_root: Root directory containing annotations

    Returns:
        Path to the annotation JSON file

    Example:
        episode_name = "episode_00000170"
        -> task_index = "0000"
        -> path = "{annotation_root}/task-0000/episode_00000170.json"
    """
    annotation_root = Path(annotation_root)

    # Extract task index from episode name (first 4 digits after "episode_")
    # episode_00000170 -> task 0000
    task_idx = episode_name.split('_')[1][:4]

    annotation_path = annotation_root / f"task-{task_idx}" / f"{episode_name}.json"

    return annotation_path


def create_skill_cache(
    skill_annotations: List[Dict[str, Any]],
    current_skill_idx: int,
    include_current: bool = False
) -> List[str]:
    """
    Create a list of past skill text representations for the memory cache.

    This is used in Phase 1 (Long-Horizon Memory) to provide the model with
    a compressed memory of past completed skills.

    Args:
        skill_annotations: List of all skill annotations for the episode
        current_skill_idx: Index of the current skill being executed
        include_current: Whether to include the current skill in the cache

    Returns:
        List of text representations of past skills
    """
    end_idx = current_skill_idx + 1 if include_current else current_skill_idx
    past_skills = skill_annotations[:end_idx]

    return [skill_to_text(skill) for skill in past_skills]


def format_skill_with_special_tokens(skill_text: str, token: str = "<L>") -> str:
    """
    Format skill text with special tokens for the model input.

    This is used in Phase 1 to wrap past skills with special tokens
    that signal to the model that this is long-term memory.

    Args:
        skill_text: Text representation of the skill
        token: Special token to use for wrapping (default: "<L>" from ProVideLLM)

    Returns:
        Formatted skill text: "<L> {skill_text} </L>"
    """
    return f"{token} {skill_text} </{token[1:]}"


def get_skill_boundaries(skill_annotations: List[Dict[str, Any]]) -> List[tuple[int, int]]:
    """
    Extract frame boundaries for all skills in the episode.

    Args:
        skill_annotations: List of skill annotation dictionaries

    Returns:
        List of (start_frame, end_frame) tuples
    """
    return [tuple(skill["frame_duration"]) for skill in skill_annotations]


def is_skill_transition(
    skill_annotations: List[Dict[str, Any]],
    frame_idx: int,
    tolerance: int = 1
) -> bool:
    """
    Check if a frame is near a skill transition boundary.

    This can be useful for special handling of frames at skill boundaries.

    Args:
        skill_annotations: List of skill annotation dictionaries
        frame_idx: Frame index to check
        tolerance: Number of frames around the boundary to consider a transition

    Returns:
        True if the frame is within tolerance of any skill boundary
    """
    for skill in skill_annotations:
        start_frame, end_frame = skill["frame_duration"]
        if abs(frame_idx - start_frame) <= tolerance or abs(frame_idx - end_frame) <= tolerance:
            return True

    return False


def is_within_skill_prediction_window(
    frame_idx: int,
    skill: Dict[str, Any],
    window_size: int = 10
) -> bool:
    """
    Check if a frame is within the prediction window of a skill.

    For efficient training, we only predict skill text at the beginning of each skill
    rather than at every frame. This function checks if the current frame is within
    the first N frames of a skill, where skill prediction should occur.

    Args:
        frame_idx: Frame index to check
        skill: Skill annotation dictionary
        window_size: Number of frames at the start of the skill to predict (default: 10)

    Returns:
        True if frame is within [start_frame, start_frame + window_size)

    Example:
        skill = {"frame_duration": [265, 1162], ...}
        is_within_skill_prediction_window(265, skill, 10)  # True (start frame)
        is_within_skill_prediction_window(270, skill, 10)  # True (within window)
        is_within_skill_prediction_window(275, skill, 10)  # False (outside window)
        is_within_skill_prediction_window(1000, skill, 10) # False (way outside)
    """
    start_frame = skill["frame_duration"][0]
    end_frame = min(start_frame + window_size, skill["frame_duration"][1])
    return start_frame <= frame_idx < end_frame


def get_skill_prediction_info(
    skill_annotations: List[Dict[str, Any]],
    frame_idx: int,
    window_size: int = 10
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Get skill prediction information for a given frame.

    Returns whether the frame should predict a skill, and the current skill.

    Args:
        skill_annotations: List of skill annotation dictionaries
        frame_idx: Frame index to check
        window_size: Number of frames at start of skill to predict

    Returns:
        (should_predict, skill): Tuple of:
            - should_predict: bool, whether to predict skill at this frame
            - skill: dict or None, the current skill (None if no skill at this frame)

    Example:
        >>> should_predict, skill = get_skill_prediction_info(skills, 265, window_size=10)
        >>> if skill:
        ...     print(f"Current skill: {skill['skill_description']}")
        ...     if should_predict:
        ...         print("  -> Should predict at this frame")
    """
    skill = get_skill_at_frame(skill_annotations, frame_idx)
    if skill is None:
        return False, None

    should_predict = is_within_skill_prediction_window(frame_idx, skill, window_size)
    return should_predict, skill  # Always return the skill, regardless of should_predict


# Example usage and testing
if __name__ == "__main__":
    # Example annotation path (uses DATASET_PATH environment variable)
    import os
    dataset_base = Path(os.getenv("DATASET_PATH", Path.cwd() / "dataset"))
    annotation_path = dataset_base / "2025-challenge-demos/annotations/task-0000/episode_00000170.json"

    if annotation_path.exists():
        # Load annotation
        annotation = load_skill_annotation(annotation_path)
        print(f"Task: {annotation['task_name']}")
        print(f"Duration: {annotation['meta_data']['task_duration']} frames")
        print(f"Number of skills: {len(annotation['skill_annotation'])}\n")

        # Display each skill
        for skill in annotation['skill_annotation']:
            print(f"Skill {skill['skill_idx']}: {skill['skill_description'][0]}")
            print(f"  Frame duration: {skill['frame_duration']}")
            print(f"  Objects: {skill['object_id']}")
            print(f"  Manipulating: {skill['manipulating_object_id']}")
            print(f"  Type: {skill['skill_type']}")
            print(f"  Text representation: {skill_to_text(skill)}")
            print()

        # Test frame lookup
        test_frame = 1000
        skill_at_frame = get_skill_at_frame(annotation['skill_annotation'], test_frame)
        if skill_at_frame:
            print(f"Frame {test_frame} is in skill: {skill_at_frame['skill_description'][0]}")

        # Test skill cache creation
        current_skill_idx = 2
        skill_cache = create_skill_cache(annotation['skill_annotation'], current_skill_idx)
        print(f"\nSkill cache for skill_idx={current_skill_idx}:")
        for i, skill_text in enumerate(skill_cache):
            print(f"  Past skill {i}: {format_skill_with_special_tokens(skill_text)}")
    else:
        print(f"Annotation file not found: {annotation_path}")
        print("This is expected if running without the dataset.")

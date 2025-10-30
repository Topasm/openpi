# Bug Fix: Missing task_index in Dataset

## Problem

During training, the following warning appeared frequently:

```
17:52:35.232 [W] Missing episode_index, index, or task_index in data. Skipping skill annotation.
```

This caused skill annotations to be skipped for ALL frames, breaking hierarchical training.

## Root Cause

The BEHAVIOR-1K dataset provides these fields in each sample:
- ✅ `episode_index` (int64): Episode identifier (e.g., 10, 1234, 51234)
- ✅ `index` (int64): Frame index within episode (e.g., 0, 100, 500)
- ❌ `task_index`: **NOT PROVIDED** by the dataset

The hierarchical transforms expected `task_index` to be present in the data dictionary, but the LeRobot dataset does not include this field.

## Solution

**Extract `task_index` from `episode_index`** using the episode naming convention.

### Episode Naming Convention

Episodes follow the format: `episode_{TTTTIIIT}` where:
- **TTTT**: Task index (4 digits) - identifies which of 50 tasks
- **III**: Instance ID (3 digits) - identifies which instance of the task
- **T**: Trajectory ID (1 digit) - identifies which trajectory

Examples:
- `episode_00000010` → task_index = `0` (00000010 // 10000 = 0)
- `episode_00051234` → task_index = `5` (00051234 // 10000 = 5)
- `episode_00491234` → task_index = `49` (00491234 // 10000 = 49)

### Implementation

The fix was applied to three transforms in [hierarchical_transforms.py](../src/openpi/training/hierarchical_transforms.py):

1. **AddSkillAnnotation** (lines 112-140)
2. **AddPastSkillsCache** (lines 302-327)
3. **CreateDynamicMemoryBatch** (lines 550-574)

#### Code Change Pattern

```python
# OLD CODE (would fail if task_index not in data):
task_idx = data.get("task_index")
if episode_idx is None or frame_idx is None or task_idx is None:
    logger.warning("Missing episode_index, index, or task_index in data. Skipping skill annotation.")
    return data

# NEW CODE (derives task_index from episode_index):
task_idx = data.get("task_index")

# If episode_index is missing, we can't proceed
if episode_idx is None or frame_idx is None:
    logger.warning("Missing episode_index or index in data. Skipping skill annotation.")
    return data

# Extract task_index from episode_index if not provided
if task_idx is None:
    # Episode format: episode_TTTTIIIT where TTTT = task index (first 4 digits)
    task_idx = episode_idx // 10000  # Integer division to get first 4 digits
    logger.debug(f"Derived task_index={task_idx} from episode_index={episode_idx}")
```

### Key Changes

1. **Made `task_index` optional** in the data dictionary
2. **Automatic derivation** using `episode_idx // 10000`
3. **Backward compatible**: If `task_index` is provided, it will still be used
4. **Debug logging**: Added log message showing derived task_index for verification

## Verification

After applying the fix, training should proceed without warnings. You can verify by:

1. Checking logs for the debug message:
   ```
   [D] Derived task_index=0 from episode_index=10
   ```

2. Verifying that skill annotations are loaded:
   ```
   [I] EOS_SKILL token ID: 257153
   [I] Using HierarchicalTokenizer with <EOS_SKILL> token support
   ```

3. Confirming that losses are computed:
   ```
   Step 0: total=1.3462, skill=0.0000, action=1.3462
   ```

## Impact

This fix ensures that:
- ✅ Skill annotations are loaded for ALL training frames
- ✅ Dense prediction works correctly (skills predicted at every frame)
- ✅ Memory integration works (past skills are cached)
- ✅ EOS token supervision is applied at skill boundaries

Without this fix, the model would train WITHOUT skill supervision, becoming a standard action-only VLA (no hierarchical capabilities).

## Files Modified

- [hierarchical_transforms.py](../src/openpi/training/hierarchical_transforms.py)
  - `AddSkillAnnotation.__call__()` (lines 93-211)
  - `AddPastSkillsCache.__call__()` (lines 290-360)
  - `CreateDynamicMemoryBatch.__call__()` (lines 531-650)

## Related Documentation

- [UNIFIED_ARCHITECTURE.md](UNIFIED_ARCHITECTURE.md) - Overview of unified hierarchical training
- [TEMPORAL_SKILL_MAPPING.md](TEMPORAL_SKILL_MAPPING.md) - How frame_duration maps to skills

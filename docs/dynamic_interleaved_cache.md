# Dynamic Interleaved Cache for Hierarchical VLA

Advanced memory system that dynamically manages long-term (text) and short-term (vision) memory blocks for hierarchical vision-language-action models.

## Overview

The Dynamic Interleaved Cache is an advanced Phase 2 extension of the hierarchical VLA system. Unlike Phase 1's static prepended cache, this system:

1. **Separates memory types**: Long-term (text summaries) vs. short-term (visual frames)
2. **Dynamic verbalization**: Converts visual memories to text upon skill completion
3. **Efficient context**: Maintains bounded memory usage while preserving long-horizon context
4. **Inference-aligned training**: Training data format matches inference-time memory state

## Architecture

### Memory Block Types

```python
# Long-term memory: Text summary of completed skills
LongTermBlock:
  skill_idx: int
  summary_text: str  # e.g., "<PAST_SKILL>{'skill':'move to',...}</PAST_SKILL>"

# Short-term memory: Visual frames from current skill
ShortTermBlock:
  skill_idx: int
  frame_timestamp: int
  visual_features: np.ndarray  # Image data
```

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                   ContextMemory Manager                      │
│                                                              │
│  Memory: [LongTermBlock, LongTermBlock, ShortTermBlock, ...] │
│                                                              │
│  Operations:                                                 │
│  • add_visual_frame()     - Add current observation         │
│  • verbalize_skill()      - Convert vision → text           │
│  • assemble_model_input() - Create model input              │
└─────────────────────────────────────────────────────────────┘
         ↓                                         ↑
         ↓ Provides input                          ↑ Adds frames
         ↓                                         ↑
┌─────────────────────────────────────────────────────────────┐
│                    Inference Loop                            │
│                                                              │
│  1. Assemble input from memory                              │
│  2. Model prediction                                        │
│  3. Execute action                                          │
│  4. Add frame to memory                                     │
│  5. Check skill completion → verbalize if done              │
└─────────────────────────────────────────────────────────────┘
         ↓
         ↓ Action
         ↓
┌─────────────────────────────────────────────────────────────┐
│                   Robot/Environment                          │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Training with Dynamic Memory

```bash
# Train with Dynamic Interleaved Cache (Phase 2)
uv run python scripts/train_hierarchical.py pi0_b1k_dynamic_memory
```

### Testing the Implementation

```bash
# Run validation tests
cd b1k-baselines/baselines/openpi
uv run python scripts/test_dynamic_memory.py
```

### Inference with ContextMemory

```python
from openpi.training.dynamic_memory import ContextMemory, summarize_skill_annotation

# Initialize memory manager
memory = ContextMemory(max_short_term_frames=10)

# Inference loop
for frame_idx, observation in enumerate(episode):
    # 1. Assemble model input
    past_text, current_images = memory.assemble_model_input()

    # 2. Model prediction
    action, skill = model.predict(past_text, current_images, task_prompt)

    # 3. Execute action
    robot.execute(action)

    # 4. Add frame to memory
    memory.add_visual_frame(observation.image, current_skill_idx, frame_idx)

    # 5. Check skill completion
    if skill_completed:
        skill_summary = summarize_skill_annotation(skill)
        memory.verbalize_skill(current_skill_idx, skill_summary)
        current_skill_idx += 1
```

## Implementation Details

### 1. Core Data Structures

**File**: `src/openpi/training/dynamic_memory.py`

The memory system uses two block types:

- **LongTermBlock**: Stores text summaries of completed skills
  - Compact representation: ~50-100 characters per skill
  - Format: `<PAST_SKILL>{"skill":"move to","obj":["table"],...}</PAST_SKILL>`
  - Persists for entire episode

- **ShortTermBlock**: Stores visual frames from current skill
  - Raw or encoded image data
  - Limited to `max_short_term_frames` (default: 10)
  - Automatically pruned when limit exceeded

### 2. ContextMemory State Manager

**Class**: `ContextMemory`

Manages the time-ordered list of memory blocks and provides three core operations:

#### add_visual_frame()
Adds current observation to short-term memory:
```python
memory.add_visual_frame(
    frame_data=image,           # np.ndarray of image
    current_skill_idx=skill_idx, # int, current skill index
    timestamp=frame_idx          # int, frame index
)
```

Automatically prunes oldest frames if exceeding `max_short_term_frames`.

#### verbalize_skill()
Converts completed skill from vision to text (the "purge and replace" logic):
```python
memory.verbalize_skill(
    completed_skill_idx=skill_idx,
    summary_json_text='{"skill":"pick up","obj":["cup"],...}'
)
```

This operation:
1. Removes all `ShortTermBlock` for the completed skill
2. Adds a single `LongTermBlock` with the text summary
3. Atomically updates the memory list

#### assemble_model_input()
Creates model input from current memory state:
```python
past_text, current_images = memory.assemble_model_input()
# Returns:
#   past_text: str - Concatenated text from all LongTermBlock
#   current_images: np.ndarray or None - Stacked images from ShortTermBlock
```

### 3. Data Pipeline Transforms

**File**: `src/openpi/training/hierarchical_transforms.py`

#### CreateDynamicMemoryBatch
Transform that simulates the inference-time memory state during training.

For each training frame at time `t` in skill `k`:
1. **Long-term memory**: Load text summaries for skills 0 to k-1
2. **Short-term memory**: Metadata about current skill frames (simplified version)
3. **Output**: `dynamic_memory_text`, `num_past_skills`, `num_current_frames`

```python
transform = CreateDynamicMemoryBatch(
    annotation_root="/path/to/annotations",
    max_short_term_frames=10,
    special_token="<PAST_SKILL>"
)
```

#### TokenizeDynamicMemory
Tokenizes the dynamic memory text for model input:

```python
transform = TokenizeDynamicMemory(max_memory_len=256)
# Adds to batch:
#   dynamic_memory_tokens: [memory_len] token IDs
#   dynamic_memory_mask: [memory_len] boolean mask
```

### 4. Model Integration

**File**: `src/openpi/models/pi0_hierarchical.py`

The `Pi0Hierarchical` model's `compute_loss()` method supports dynamic memory:

```python
loss, loss_dict = model.compute_loss(
    rng=rng,
    observation=observation,
    actions=actions,
    skill_tokens=skill_tokens,
    skill_mask=skill_mask,
    dynamic_memory_tokens=dynamic_memory_tokens,  # NEW
    dynamic_memory_mask=dynamic_memory_mask,      # NEW
    use_dynamic_memory=True,                       # NEW
    train=True
)
```

When `use_dynamic_memory=True`:
- Uses `embed_prefix_with_memory()` with dynamic memory tokens
- Memory tokens are prepended to visual observations
- Full attention within memory, causal attention for actions

### 5. Configuration

**File**: `src/openpi/training/config.py`

Two main configurations available:

#### Phase 1: Static Memory (Original)
```python
TrainConfig(
    name="pi0_b1k_hierarchical",
    data=LeRobotB1KHierarchicalDataConfig(
        enable_memory=True,  # Use static prepended cache
        ...
    )
)
```

#### Phase 2: Dynamic Memory (New)
```python
TrainConfig(
    name="pi0_b1k_dynamic_memory",
    data=LeRobotB1KDynamicMemoryDataConfig(
        max_short_term_frames=10,  # Limit on visual frames
        ...
    )
)
```

## Training Pipeline

### Data Flow

```
Episode Frame
    ↓
AddSkillAnnotation         # Add skill info for current frame
    ↓
CreateDynamicMemoryBatch   # Simulate memory state
    ↓  dynamic_memory_text: "Skill 0 summary, Skill 1 summary, ..."
    ↓  num_past_skills: N
    ↓
TokenizeDynamicMemory      # Tokenize memory text
    ↓  dynamic_memory_tokens: [256]
    ↓  dynamic_memory_mask: [256]
    ↓
Model.compute_loss()       # Train with dynamic memory
    ↓  use_dynamic_memory=True
    ↓
    Loss: skill_loss + action_loss
```

### Training Command

```bash
# Start training
uv run python scripts/train_hierarchical.py pi0_b1k_dynamic_memory

# Monitor in WandB
# Project: B1K_DynamicMemory
# Watch: skill_loss, action_loss, total_loss
```

### Expected Training Behavior

The training dynamics should be similar to Phase 1:

| Metric | Initial | 10K steps | 50K steps |
|--------|---------|-----------|-----------|
| Skill Loss | ~10.0 | ~1.0-2.0 | ~0.5-1.0 |
| Action Loss | ~1.0 | ~0.05-0.1 | ~0.02-0.05 |
| Skill Accuracy | ~10% | >90% | >95% |

Additional metrics to monitor:
- `num_past_skills`: Average number of past skills in memory
- `memory_text_length`: Length of memory text (should be bounded)

## Inference Pipeline

### Runtime Memory Management

```python
from openpi.training.dynamic_memory import ContextMemory
from scripts.inference_with_dynamic_memory import DynamicMemoryInferenceLoop

# Initialize
memory = ContextMemory(max_short_term_frames=10)
annotation = load_skill_annotation(annotation_path)

# Inference loop
current_skill_idx = 0
for frame_idx, observation in enumerate(episode):
    # 1. Assemble input
    past_text, current_images = memory.assemble_model_input()

    # 2. Predict
    action = model.predict(past_text, current_images, task_prompt)

    # 3. Execute
    robot.execute(action)

    # 4. Add to memory
    memory.add_visual_frame(observation.image, current_skill_idx, frame_idx)

    # 5. Check completion
    current_skill = annotation["skill_annotation"][current_skill_idx]
    if frame_idx >= current_skill["frame_duration"][1]:
        # Skill completed - verbalize
        summary = summarize_skill_annotation(current_skill)
        memory.verbalize_skill(current_skill_idx, summary)
        current_skill_idx += 1
```

### Skill Completion Detection

In production (without ground-truth annotations), use:

1. **Model confidence**: When skill prediction probability drops
2. **State change**: Environment state indicates task completion
3. **Time-based**: Heuristics based on typical skill duration
4. **Hybrid**: Combination of above signals

Example:
```python
def detect_skill_completion(
    skill_prediction_confidence: float,
    state_changed: bool,
    frames_in_skill: int,
    typical_skill_length: int
) -> bool:
    # Model is uncertain about current skill
    if skill_prediction_confidence < 0.5:
        return True

    # Environment state changed (e.g., object picked up)
    if state_changed:
        return True

    # Skill taking too long
    if frames_in_skill > 2 * typical_skill_length:
        return True

    return False
```

## Advantages Over Phase 1

### 1. Memory Efficiency

**Phase 1 (Static Cache)**:
```
Memory: [Text_Skill0, Text_Skill1, ..., Image_t, Image_t+1, ...]
Tokens: ~50 * N_skills + 256 * N_images
```

**Phase 2 (Dynamic Cache)**:
```
Memory: [Text_Skill0, ..., Image_current1, Image_current2, ...]
Tokens: ~50 * N_past_skills + 256 * 10  (bounded!)
```

For an episode with 5 skills and frame 1000 of skill 3:
- Phase 1: ~250 text tokens + 256*N image tokens (grows with episode length)
- Phase 2: ~150 text tokens + 256*10 image tokens (bounded)

### 2. Inference-Training Alignment

**Phase 1**: Training uses all past skill text. Inference might use different heuristics.

**Phase 2**: Training exactly simulates inference memory state:
- Same memory assembly logic
- Same verbalization trigger points
- Same token limits

This reduces train-inference mismatch.

### 3. Semantic Clarity

**Phase 1**: All past context as text, all current context as images

**Phase 2**: Explicit separation:
- Long-term memory = "What was accomplished"
- Short-term memory = "What is happening now"

This clearer separation may help the model learn better temporal abstractions.

## Performance Tuning

### Hyperparameters

```python
# Memory capacity
max_short_term_frames = 10      # Visual frames to keep (default: 10)
max_memory_len = 256            # Text token limit (default: 256)

# Loss weights
skill_loss_weight = 10.0        # Weight for skill prediction
action_loss_weight = 1.0        # Weight for action prediction

# Skill prediction
skill_prediction_window = 10    # Frames to predict at skill start
```

### Memory Size Trade-offs

**Smaller `max_short_term_frames` (5-8)**:
- ✓ Faster training (less compute)
- ✓ Lower memory usage
- ✗ Less visual context for current skill

**Larger `max_short_term_frames` (15-20)**:
- ✓ More visual context
- ✓ Better temporal understanding
- ✗ Slower training
- ✗ Higher memory usage

**Recommended**: Start with 10, adjust based on:
- Task complexity (simple tasks: 5-8, complex: 15-20)
- Skill duration (short skills: 5, long skills: 15+)
- Available compute (limited: 5-8, abundant: 15-20)

## Troubleshooting

### Issue: "dynamic_memory_tokens not in batch"

**Cause**: Using wrong config or transform pipeline

**Solution**:
```python
# Make sure you're using LeRobotB1KDynamicMemoryDataConfig
data=LeRobotB1KDynamicMemoryDataConfig(...)
# NOT LeRobotB1KHierarchicalDataConfig
```

### Issue: Memory growing unbounded

**Cause**: Not calling `verbalize_skill()` on completion

**Solution**: Ensure skill completion detection is working:
```python
# Add logging
if frame_idx >= skill_end_frame:
    logger.info(f"Verbalizing skill {skill_idx} at frame {frame_idx}")
    memory.verbalize_skill(skill_idx, summary)
```

### Issue: High memory usage during training

**Cause**: Too many past skills or images in cache

**Solution**:
- Reduce `max_short_term_frames` to 5-8
- Reduce `max_memory_len` to 128
- Check that `_purge_oldest_short_term_memory()` is working

### Issue: Skill loss not decreasing

**Cause**: Similar to Phase 1 issues

**Solution**:
- Check `skill_prediction_window` is set correctly
- Verify `predict_skill` flag is True for first N frames
- Try increasing `skill_loss_weight` to 15.0-20.0
- Check that `dynamic_memory_text` is being populated

## Testing

### Unit Tests

```bash
# Run comprehensive test suite
uv run python scripts/test_dynamic_memory.py

# Tests validate:
# 1. ContextMemory add/verbalize/assemble operations
# 2. CreateDynamicMemoryBatch transform
# 3. Skill annotation parsing
# 4. Memory state creation helpers
```

### Integration Test

```bash
# Test full training pipeline (1 step)
uv run python scripts/train_hierarchical.py pi0_b1k_dynamic_memory \
    --num_train_steps 1 \
    --batch_size 2
```

### Inference Test

```bash
# Test inference loop (requires checkpoint)
uv run python scripts/inference_with_dynamic_memory.py \
    --checkpoint_path /path/to/checkpoint \
    --episode_index 170 \
    --task_index 0
```

## Comparison: Phase 0 vs 1 vs 2

| Feature | Phase 0 | Phase 1 | Phase 2 (Dynamic) |
|---------|---------|---------|-------------------|
| Memory Type | None | Static text | Dynamic text + vision |
| Context Window | Current frame | All past skills | Past skills (text) + recent frames (vision) |
| Memory Growth | N/A | Linear in #skills | Bounded |
| Training Alignment | N/A | Approximate | Exact |
| Complexity | Low | Medium | High |
| Best For | Short tasks | Medium tasks | Long, multi-skill tasks |

## References

- **Pi0**: [Physical Intelligence Blog](https://www.physicalintelligence.company/blog/pi0)
- **ProVideLLM**: [arXiv:2401.06465](https://arxiv.org/abs/2401.06465)
- **BEHAVIOR-1K**: [Stanford BEHAVIOR](https://behavior.stanford.edu/)
- **Hierarchical VLA Docs**: [hierarchical_vla.md](hierarchical_vla.md)

## Files Reference

### Core Implementation
- `src/openpi/training/dynamic_memory.py` - Memory blocks and ContextMemory
- `src/openpi/training/hierarchical_transforms.py` - Data transforms
- `src/openpi/models/pi0_hierarchical.py` - Model integration
- `src/openpi/training/config.py` - Training configuration

### Scripts
- `scripts/inference_with_dynamic_memory.py` - Inference loop example
- `scripts/test_dynamic_memory.py` - Test suite
- `scripts/train_hierarchical.py` - Training script

### Documentation
- `docs/hierarchical_vla.md` - Phase 0/1 documentation
- `docs/dynamic_interleaved_cache.md` - This document

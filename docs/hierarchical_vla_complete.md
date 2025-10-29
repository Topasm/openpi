# Hierarchical VLA for BEHAVIOR-1K - Complete Guide

Multi-task learning model that predicts both high-level skills and low-level actions with long-horizon memory support.

## Table of Contents
- [Quick Start](#quick-start)
- [Training Phases](#training-phases)
- [Architecture](#architecture)
- [Implementation](#implementation)
- [Data Format](#data-format)
- [Inference](#inference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Phase 0: Multi-Task Learning (Skills + Actions)
```bash
# 1. Compute normalization statistics
uv run scripts/compute_norm_stats.py --config-name pi0_b1k_hierarchical

# 2. Train Phase 0
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_hierarchical.py pi0_b1k_hierarchical \
  --exp-name="phase0_$(date +%Y%m%d_%H%M%S)" \
  --batch_size=32 \
  --num_train_steps=50000
```

### Phase 1: Long-Horizon Memory
```bash
# Edit config.py line 470: enable_memory=True
# Then train
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_hierarchical.py pi0_b1k_hierarchical \
  --exp-name="phase1_$(date +%Y%m%d_%H%M%S)"
```

### Phase 3: Dense Prediction with EOS Token
```bash
# Edit config.py:
#   use_dense_prediction=True
#   use_eos_token=True
#   use_concise_format=True
#   use_hierarchical_tokenizer=True

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_hierarchical.py pi0_b1k_hierarchical \
  --exp-name="phase3_$(date +%Y%m%d_%H%M%S)"
```

---

## Training Phases

### Phase 0: Multi-Task Learning (Sparse Prediction)
**Goal**: Learn to predict skills and actions jointly

**Config**:
```python
enable_memory = False
use_dense_prediction = False
skill_prediction_window = 10  # Only first 10 frames of each skill
```

**Loss**: `L_total = 10.0 * L_skill + 1.0 * L_action`

**Training signal**:
- Skill prediction: 2-5% of frames (first 10 frames per skill)
- Action prediction: 100% of frames

### Phase 1: Long-Horizon Memory
**Goal**: Handle long episodes by compressing past skills as text

**Config**:
```python
enable_memory = True
use_dense_prediction = False
skill_prediction_window = 10
```

**Memory format**: `<L> {"skill":"move to",...} </L> <L> {"skill":"pick up",...} </L>`

**Memory size**: 256 tokens (~10-15 past skills)

### Phase 3: Dense Prediction with Learned Boundaries
**Goal**: Predict skills at every frame, learn to detect skill boundaries with EOS token

**Config**:
```python
enable_memory = True  # Optional, but recommended
use_dense_prediction = True
use_eos_token = True
use_concise_format = True
use_hierarchical_tokenizer = True
```

**Training signal**:
- Skill prediction: **100% of frames** (dense)
- EOS token: Only at last frame of each skill
- Action prediction: 100% of frames

**Skill format**:
- During skill: `{"skill":"move to","obj":"trash_can"}`
- At end frame: `{"skill":"move to","obj":"trash_can"} <EOS_SKILL>`
- In memory: `<PAST_SKILL>{"skill":"move to","obj":"trash_can"}</PAST_SKILL>`

**Key advantage**: Model learns when skills end, no ground-truth needed at inference!

---

## Architecture

### Phase 0/1: Multi-Task with Memory

```
Vision Encoder (SigLIP)
        ↓
[Past Skills Memory (text)] + [Current Observation (vision)] + [Task Prompt]
        ↓
   PaliGemma Expert
        ↓
   ┌────┴────┐
   ↓         ↓
Skill Head  Action Head
(Language)  (Flow Matching)
```

### Phase 3: Dense Prediction Flow

```
Every frame:
  Input: [Long-term memory (text)] + [Short-term memory (last 10 vision frames)]
  Output: [Skill JSON] + [Actions]

At skill completion (EOS detected):
  1. PURGE: Remove all vision frames from completed skill
  2. REPLACE: Add text summary to long-term memory
  3. Result: Efficient compression (100+ frames → 1 text string)
```

**Memory evolution**:
```
Skill 0 executing:
  Memory: [Vision_t0, Vision_t1, ..., Vision_t100]

Skill 0 completes (EOS detected):
  Memory: [LongTermBlock("<PAST_SKILL>{skill 0}</PAST_SKILL>")]

Skill 1 executing:
  Memory: [
    LongTermBlock("<PAST_SKILL>{skill 0}</PAST_SKILL>"),
    Vision_t200, Vision_t201, ..., Vision_t250
  ]
```

---

## Implementation

### Model (`src/openpi/models/pi0_hierarchical.py`)

**Key classes**:
- `Pi0HierarchicalConfig` - Configuration with phase parameters
- `Pi0Hierarchical` - Model with dual prediction heads

**Key methods**:
```python
# Embedding
embed_prefix(observation)  # Phase 0: vision + prompt
embed_prefix_with_memory(observation, memory_tokens, memory_mask)  # Phase 1/3

# Training
compute_loss(rng, observation, actions, skill_tokens, memory_tokens, ...)

# Inference
sample_actions(rng, observation)  # Action sampling only
infer_with_memory(rng, observation, memory_text, tokenizer, ...)  # Phase 3
```

### Data Pipeline (`src/openpi/training/hierarchical_transforms.py`)

**Transforms**:

1. `AddSkillAnnotation` - Loads skill JSON and determines prediction
   - Phase 0/1: Sets `predict_skill=True` only for first 10 frames
   - Phase 3: Sets `predict_skill=True` for ALL frames, appends EOS at end

2. `TokenizeSkills` - Tokenizes skill text
   - Phase 0/1: Uses `PaligemmaTokenizer`
   - Phase 3: Uses `HierarchicalTokenizer` (with `<EOS_SKILL>` token)

3. `AddPastSkillsCache` - Creates memory (Phase 1 only)

4. `TokenizeMemory` - Tokenizes memory (Phase 1 only)

### Tokenizer (`src/openpi/models/tokenizer.py`)

**Phase 0/1**: `PaligemmaTokenizer` (standard)

**Phase 3**: `HierarchicalTokenizer`
- Adds special token: `<EOS_SKILL>`
- Based on `transformers.AutoTokenizer`
- Stores `eos_skill_token_id` for detection

### Skill Utilities (`src/openpi/training/skill_utils.py`)

**Phase 0/1**:
- `skill_to_text()` - Full JSON format with all fields

**Phase 3**:
- `create_concise_skill_summary_json()` - Concise format
- `_clean_object_id()` - Removes trailing numbers (trash_can_123 → trash_can)

### Dynamic Memory (`src/openpi/training/dynamic_memory.py`)

**Classes**:
- `LongTermBlock` - Text summary of completed skill
- `ShortTermBlock` - Visual frame from current skill
- `ContextMemory` - State manager

**Key methods**:
```python
add_visual_frame(frame_data, skill_idx, timestamp)  # Add to short-term
verbalize_skill(skill_idx, summary_text)  # PURGE vision, REPLACE with text
assemble_model_input()  # Returns (past_text, current_images)
```

---

## Data Format

### Skill Annotations
**Location**: `dataset/2025-challenge-demos/annotations/task-XXXX/episode_XXXXXXXX.json`

**Structure**:
```json
{
  "task_name": "Picking up the object on the floor next to the furniture and placing it in the trash can",
  "skill_annotation": [
    {
      "skill_idx": 0,
      "skill_description": ["move to"],
      "object_id": [["trash_can_123"]],
      "manipulating_object_id": [],
      "skill_type": ["navigation"],
      "frame_duration": [0, 264]
    },
    {
      "skill_idx": 1,
      "skill_description": ["pick up from"],
      "object_id": [["radio_89", "floor"]],
      "manipulating_object_id": ["radio_89"],
      "skill_type": ["uncoordinated"],
      "frame_duration": [265, 1162]
    }
  ]
}
```

### Skill Text Formats

**Phase 0/1** (Full format):
```json
{"skill":"pick up from","obj":["radio_89","floor"],"manip":"radio_89","type":"uncoordinated"}
```

**Phase 3** (Concise format):
```json
{"skill":"pick up from","obj":"radio","type":"uncoordinated"}
```

Note: Object IDs cleaned (radio_89 → radio), only primary object included

**Phase 3 with EOS** (at end frame):
```json
{"skill":"pick up from","obj":"radio","type":"uncoordinated"} <EOS_SKILL>
```

**Long-term memory** (after verbalization):
```json
<PAST_SKILL>{"skill":"pick up from","obj":"radio","type":"uncoordinated"}</PAST_SKILL>
```

---

## Inference

### Phase 0/1: Ground-Truth Boundaries
```bash
python scripts/inference_with_dynamic_memory.py \
    --checkpoint_path /path/to/checkpoint \
    --annotation_root dataset/2025-challenge-demos/annotations \
    --episode_index 170 \
    --task_index 0 \
    --boundary_detection ground_truth
```

### Phase 3: EOS Token Detection
```bash
python scripts/inference_with_dynamic_memory.py \
    --checkpoint_path /path/to/checkpoint \
    --episode_index 170 \
    --task_index 0 \
    --boundary_detection eos_token \
    --use_hierarchical_tokenizer
```

**Inference flow**:
```
For each timestep:
  1. Assemble input from ContextMemory
     - Long-term: Text summaries of past skills
     - Short-term: Last 10 visual frames

  2. Model inference (infer_with_memory)
     Input: observation + memory_text
     Output: actions + skill_text

  3. Execute actions

  4. Add current frame to short-term memory

  5. Check for EOS token
     If skill_text.endswith("<EOS_SKILL>"):
       - Extract JSON (remove <EOS_SKILL>)
       - Verbalize: PURGE vision, REPLACE with text
       - Advance to next skill
```

---

## Configuration

### File: `src/openpi/training/config.py`

### Data Config: `LeRobotB1KHierarchicalDataConfig`

```python
@dataclasses.dataclass(frozen=True)
class LeRobotB1KHierarchicalDataConfig(DataConfigFactory):
    annotation_root: str = "dataset/2025-challenge-demos/annotations"

    # Phase 0: Basic multi-task
    skill_prediction_window: int = 10
    enable_memory: bool = False

    # Phase 3: Dense prediction with EOS
    use_dense_prediction: bool = False  # True for Phase 3
    use_eos_token: bool = False         # True for Phase 3
    use_concise_format: bool = False    # True for Phase 3
    use_hierarchical_tokenizer: bool = False  # True for Phase 3
```

### Model Config: `Pi0HierarchicalConfig`

```python
Pi0HierarchicalConfig(
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_2b",
    action_dim=32,
    action_horizon=50,
    max_token_len=256,

    # Hierarchical-specific
    skill_loss_weight=10.0,
    action_loss_weight=1.0,
    max_skill_tokens=64,

    # Phase 3
    use_hierarchical_tokenizer=False,  # True for Phase 3
    vocab_size_override=None  # Set to len(tokenizer) if using special tokens
)
```

### Training Preset: `pi0_b1k_hierarchical`

Located at line ~980 in `config.py`:

```python
TrainConfig(
    name="pi0_b1k_hierarchical",
    exp_name="openpi_hierarchical",
    data=LeRobotB1KHierarchicalDataConfig(
        enable_memory=True,  # Phase 1: True, Phase 0: False
        # Add Phase 3 flags here
    ),
    model=Pi0HierarchicalConfig(
        skill_loss_weight=10.0,
        action_loss_weight=1.0,
    ),
    ...
)
```

---

## Training Details

### Expected Results

| Phase | Metric | Initial | 10K steps | 50K steps |
|-------|--------|---------|-----------|-----------|
| 0 | Skill Loss | ~10.0 | ~1.0-2.0 | ~0.5-1.0 |
| 0 | Action Loss | ~1.0 | ~0.05-0.1 | ~0.02-0.05 |
| 0 | Skill Accuracy | ~10% | >90% | >95% |
| 3 | Skill Loss | ~12.0 | ~2.0-3.0 | ~0.8-1.5 |
| 3 | EOS Detection | <5% | >80% | >95% |

Note: Phase 3 has higher initial loss due to dense prediction (100% vs 2-5% of frames)

### Monitoring

**WandB project**: `B1K_Hierarchical`

**Key metrics**:
- `skill_loss` - Cross-entropy for skill prediction
- `action_loss` - Flow matching for actions
- `total_loss` - Weighted combination
- `skill_accuracy` - Token-level accuracy (Phase 3: includes EOS)

### Timeline (on 8×A100)
- 10K steps: ~12-24 hours
- 50K steps: ~2-3 days

### Hardware Requirements
- **Minimum**: 1×A100 (40GB), batch_size=8
- **Recommended**: 4×A100 (40GB), batch_size=32
- **Optimal**: 8×A100 (80GB), batch_size=64

---

## Troubleshooting

### Training Issues

**"skill_tokens not in batch"**
- **Cause**: Using wrong config
- **Fix**: Must use `pi0_b1k_hierarchical` config

**"Annotation file not found"**
- **Cause**: Incorrect `annotation_root` path
- **Fix**: Check path in config.py, ensure annotations exist

**Skill loss not decreasing**
- **Cause**: Insufficient signal or wrong loss weight
- **Fix**: Try increasing `skill_loss_weight` to 20.0

**OOM (Out of Memory)**
- **Cause**: Batch size too large
- **Fix**: Reduce `batch_size` to 16 or 8

**Phase 3: Model doesn't generate EOS**
- **Cause**: Not trained long enough, or EOS not in training data
- **Fix**:
  1. Verify `use_eos_token=True` in config
  2. Check training logs that EOS appears in targets
  3. Train longer (EOS detection needs >20K steps typically)

### Inference Issues

**"Tokenizer required for eos_token mode"**
- **Cause**: Missing `--use_hierarchical_tokenizer` flag
- **Fix**: Add flag when using `--boundary_detection eos_token`

**Memory grows too large**
- **Cause**: Too many short-term frames
- **Fix**: Adjust `--max_short_term_frames` (default: 10)

**Actions ignore memory context**
- **Cause**: Memory not properly embedded in prefix
- **Fix**: Verify `memory_text` is not empty, check model logs

---

## File Reference

### Core Implementation
```
src/openpi/models/
  ├── pi0_hierarchical.py      # Model architecture
  ├── tokenizer.py             # HierarchicalTokenizer (Phase 3)
  └── gemma.py                 # Gemma backbone

src/openpi/training/
  ├── hierarchical_transforms.py  # Data transforms
  ├── skill_utils.py              # Skill processing
  ├── dynamic_memory.py           # ContextMemory (Phase 3)
  └── config.py                   # Configuration

scripts/
  ├── train_hierarchical.py                # Training script
  ├── compute_norm_stats.py                # Normalization
  └── inference_with_dynamic_memory.py     # Inference (Phase 3)
```

---

## Development Notes

### Why Sparse Prediction (Phase 0/1)?
Only predict skills at first 10 frames of each skill:
- **Efficient**: 2-5% of frames compute skill loss
- **Sufficient**: 10 frames provide strong training signal
- **Semantic**: Model learns to "plan" at start, "execute" during

### Why Dense Prediction (Phase 3)?
Predict skills at **every** frame:
- **Learned boundaries**: Model learns when skills end via EOS
- **No ground truth needed**: Can deploy without annotations
- **More training signal**: 100% of frames contribute to skill learning
- **Trade-off**: Higher compute cost during training

### Why Text Memory?
Compressed text instead of frames:
- **Efficiency**: 256 tokens vs 100s of image frames
- **Semantic**: Captures "what was done" not "how it looked"
- **Long-term**: Remembers 10-15 skills vs 3-5 frames
- **Scalable**: Can handle episodes with 50+ skills

### Phase 3 Format Design
Simple, consistent format:
- During skill: `{"skill":"...","obj":"..."}`
- At completion: `{"skill":"...","obj":"..."} <EOS_SKILL>`
- In memory: `<PAST_SKILL>{"skill":"...","obj":"..."}</PAST_SKILL>`

Benefits:
- **Clear**: Obvious what each part means
- **Minimal**: Few tokens used
- **Consistent**: Same JSON structure throughout
- **Parseable**: Easy to extract and validate

---

## References

- **Pi0**: https://www.physicalintelligence.company/blog/pi0
- **ProVideLLM**: https://arxiv.org/abs/2401.06465 (memory compression inspiration)
- **BEHAVIOR-1K**: https://behavior.stanford.edu/
- **PaliGemma**: https://github.com/google-research/big_vision

---

## Quick Command Reference

```bash
# Compute norm stats (once before training)
uv run scripts/compute_norm_stats.py --config-name pi0_b1k_hierarchical

# Phase 0 training
uv run scripts/train_hierarchical.py pi0_b1k_hierarchical \
  --exp-name="phase0"

# Phase 1 training (after editing config: enable_memory=True)
uv run scripts/train_hierarchical.py pi0_b1k_hierarchical \
  --exp-name="phase1"

# Phase 3 training (after editing config: all Phase 3 flags=True)
uv run scripts/train_hierarchical.py pi0_b1k_hierarchical \
  --exp-name="phase3"

# Inference with EOS detection
python scripts/inference_with_dynamic_memory.py \
  --checkpoint_path /path/to/ckpt \
  --episode_index 170 \
  --boundary_detection eos_token \
  --use_hierarchical_tokenizer
```

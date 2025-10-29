# Training Clarification: Unified Multi-Task Learning

## What the Model Trains

The `pi0_b1k_dynamic_memory` config trains **both features simultaneously from the start**:

### Feature 1: Skill Prediction (Phase 0 Feature)
- **What**: Predict skill text from current observation
- **Input**: Current visual frames + task prompt
- **Output**: Skill JSON text (e.g., `{"skill":"pick up","obj":["cup"],"manip":"cup"}`)
- **Loss**: Cross-entropy on skill tokens
- **Training signal**: Every frame in skill prediction window (first 10 frames of each skill)

### Feature 2: Dynamic Memory (Advanced Feature)
- **What**: Use past skill summaries as context for current prediction
- **Input**: Past skill texts (long-term) + current visual frames (short-term)
- **Output**: Better action/skill predictions using history
- **Benefit**: Model learns to leverage past context

## How They Work Together

```python
# In compute_loss() - BOTH features are computed every training step

# 1. ACTION LOSS (uses dynamic memory as context)
prefix = [dynamic_memory_tokens, current_images, task_prompt]
action_pred = model(prefix)
action_loss = MSE(action_pred, action_target)

# 2. SKILL LOSS (predicts current skill text)
prefix = [dynamic_memory_tokens, current_images, task_prompt]
skill_pred = model(prefix)
skill_loss = CrossEntropy(skill_pred, skill_target)

# 3. TOTAL LOSS (both trained together)
total_loss = 10.0 * skill_loss + 1.0 * action_loss
```

## Training Pipeline Detail

### Input to Model at Each Training Step

For a frame at time `t` in skill `k`:

```
┌─────────────────────────────────────────────┐
│             MODEL INPUT                      │
├─────────────────────────────────────────────┤
│                                             │
│  [MEMORY TOKENS]                            │  ← Dynamic memory feature
│    Past Skill 0: <PAST_SKILL>{...}         │  ← Text summary of completed skills
│    Past Skill 1: <PAST_SKILL>{...}         │
│    ...                                      │
│    Past Skill k-1: <PAST_SKILL>{...}       │
│                                             │
│  [CURRENT IMAGES]                           │  ← Current observations
│    Image_t-9                                │
│    Image_t-8                                │
│    ...                                      │
│    Image_t (current)                        │
│                                             │
│  [TASK PROMPT]                              │
│    "pick up the cup"                        │
│                                             │
└─────────────────────────────────────────────┘
            ↓
    ┌───────────────┐
    │  PI0 MODEL    │
    └───────────────┘
            ↓
    ┌─────────────────────────────────────┐
    │        PREDICTIONS                   │
    ├─────────────────────────────────────┤
    │                                     │
    │  SKILL PRED:  {"skill":"pick up",   │  ← Phase 0 feature
    │                "obj":["cup"],       │
    │                "manip":"cup"}       │
    │                                     │
    │  ACTION PRED: [a_0, a_1, ..., a_49] │  ← Standard feature
    │                                     │
    └─────────────────────────────────────┘
            ↓
    ┌─────────────────────────────────────┐
    │         LOSS COMPUTATION             │
    ├─────────────────────────────────────┤
    │                                     │
    │  skill_loss = CE(skill_pred, GT)    │  ← Phase 0 loss
    │  action_loss = MSE(action_pred, GT) │  ← Standard loss
    │                                     │
    │  total_loss = 10*skill + 1*action   │  ← Combined
    │                                     │
    └─────────────────────────────────────┘
```

## Data Transform Pipeline

```python
# In LeRobotB1KDynamicMemoryDataConfig (config.py line 600-607)

data_transforms.push(
    inputs=[
        skill_annotation_transform,      # ← Adds skill_tokens (Phase 0 feature)
        dynamic_memory_transform,        # ← Adds dynamic_memory_tokens (Memory feature)
        skill_tokenize_transform,        # ← Tokenizes skill text
        dynamic_memory_tokenize_transform # ← Tokenizes memory text
    ]
)
```

**Each batch contains:**
```python
batch = {
    # Standard Pi0 data
    "observation": {...},
    "actions": [B, 50, 32],

    # Phase 0 feature: Skill prediction
    "skill_tokens": [B, 64],        # Current skill to predict
    "skill_mask": [B, 64],
    "predict_skill": bool,          # Whether to predict at this frame

    # Dynamic memory feature
    "dynamic_memory_tokens": [B, 256],  # Past skills as context
    "dynamic_memory_mask": [B, 256],
    "num_past_skills": int,         # How many past skills in memory
}
```

## Training Steps Breakdown

### Step 1: Early in episode (skill 0, frame 5)
```
Memory: []  (no past skills yet)
Current Skill: "move to radio"
Predict: skill_text="move to radio", action=[...]
Losses: skill_loss=8.5, action_loss=0.9
```

### Step 2: Middle of episode (skill 2, frame 1000)
```
Memory: [
  "<PAST_SKILL>{'skill':'move to','obj':['radio']}</PAST_SKILL>",
  "<PAST_SKILL>{'skill':'pick up','obj':['radio','table']}</PAST_SKILL>"
]
Current Skill: "press radio"
Predict: skill_text="press radio", action=[...]
Losses: skill_loss=1.2, action_loss=0.08  ← Better due to memory context!
```

### Step 3: Late in episode (skill 3, frame 2000)
```
Memory: [
  "<PAST_SKILL>{'skill':'move to',...}</PAST_SKILL>",
  "<PAST_SKILL>{'skill':'pick up',...}</PAST_SKILL>",
  "<PAST_SKILL>{'skill':'press',...}</PAST_SKILL>"
]
Current Skill: "place on table"
Predict: skill_text="place on table", action=[...]
Losses: skill_loss=0.6, action_loss=0.03  ← Even better!
```

## Why Both Features Are Essential

### Skill Prediction (Phase 0) Enables:
1. **Explicit skill understanding** - Model learns task semantics
2. **Training signal for memory** - Creates the skill summaries used in memory
3. **Multi-task learning** - Better representations via auxiliary task

### Dynamic Memory Enables:
1. **Long-horizon reasoning** - Model sees what was accomplished
2. **Context efficiency** - Text summaries << full video history
3. **Better predictions** - Past context improves current skill/action

**They are interdependent:**
- Memory uses skill predictions as summaries
- Skill predictions benefit from memory context
- Both improve action predictions

## Loss Weights Explained

```python
skill_loss_weight = 10.0   # High weight to overcome data imbalance
action_loss_weight = 1.0

total_loss = 10.0 * skill_loss + 1.0 * action_loss
```

**Why skill loss weight is 10x?**
- Skill prediction happens only on ~1-5% of frames (first 10 frames of each skill)
- Action prediction happens on 100% of frames
- 10x weight balances the training signal

## Expected Training Dynamics

| Step | Skill Loss | Action Loss | Total Loss | Memory Usage |
|------|-----------|-------------|------------|--------------|
| 0 | 10.0 | 1.0 | 101.0 | No memory yet |
| 1K | 5.0 | 0.5 | 50.5 | Learning to use memory |
| 10K | 2.0 | 0.1 | 20.1 | Memory helping predictions |
| 25K | 1.0 | 0.05 | 10.05 | Strong memory integration |
| 50K | 0.7 | 0.03 | 7.03 | Converged |

## What Gets Logged to WandB

```python
# Every training step logs:
{
    "skill_loss": 1.2,           # Cross-entropy on skill prediction
    "action_loss": 0.08,         # MSE on action prediction
    "total_loss": 12.08,         # Combined loss

    # Additional metrics (if computed)
    "skill_accuracy": 0.92,      # How often skill prediction is correct
    "num_past_skills": 2,        # Average past skills in memory
    "memory_text_length": 180,   # Average memory token count
}
```

## Single Config Trains Both

The `pi0_b1k_dynamic_memory` config includes **both** features:

```python
# config.py line 1034-1044
data=LeRobotB1KDynamicMemoryDataConfig(
    annotation_root="...",
    skill_prediction_window=10,      # ← Phase 0: When to predict skills
    max_short_term_frames=10,        # ← Memory: How many visual frames
)
```

This config adds transforms for:
1. ✅ `AddSkillAnnotation` - Loads skill labels for prediction
2. ✅ `TokenizeSkills` - Tokenizes skill text
3. ✅ `CreateDynamicMemoryBatch` - Creates past skill summaries
4. ✅ `TokenizeDynamicMemory` - Tokenizes memory text

## Summary

**You don't need separate phases.**

One training run with `pi0_b1k_dynamic_memory` trains:
- ✅ Skill prediction (from annotations)
- ✅ Action prediction (standard Pi0)
- ✅ Dynamic memory usage (past skills as context)

All three objectives are optimized **simultaneously** via multi-task loss:
```
L_total = λ_skill * L_skill + λ_action * L_action
```

The model learns to:
1. Predict current skill from current observations (**Phase 0 feature**)
2. Predict actions from observations + past skill memory (**Memory feature**)
3. Use memory effectively to improve both predictions (**Synergy**)

**Training command (trains everything):**
```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_hierarchical.py pi0_b1k_dynamic_memory \
  --exp-name="full_system_$(date +%Y%m%d_%H%M%S)" \
  --batch_size=32 \
  --num_train_steps=50000
```

This single command trains the complete system with all features enabled from step 0.

# Temporal Skill Mapping: How Frame Duration Enables Skill Generation

This document explains the **critical mechanism** by which the unified hierarchical VLA model learns to map temporal information (frame indices and durations) to skill predictions.

## The Core Question

**How does the model learn that frames 265-1162 correspond to "pick up from" skill?**

The answer lies in the **temporal supervision** provided by the `frame_duration` field in skill annotations, which enables the model to learn implicit timing representations through vision.

---

## Annotation Structure

Each skill annotation contains temporal boundaries:

```json
{
    "skill_idx": 1,
    "skill_description": ["pick up from"],
    "object_id": [["radio_89", "coffee_table_koagbh_0"]],
    "frame_duration": [265, 1162],  // ← CRITICAL: [start_frame, end_frame)
    "skill_type": ["uncoordinated"]
}
```

The `frame_duration` field defines:
- **Start frame**: When the skill begins (inclusive)
- **End frame**: When the skill ends (exclusive, like Python ranges)

For the example above:
- Frames **0 to 264**: Skill 0 ("move to")
- Frames **265 to 1161**: Skill 1 ("pick up from") ← 897 frames
- Frame **1162**: End of skill 1, start of skill 2 ("press")

---

## Training: Temporal Supervision via Dense Prediction

### Step 1: Frame → Skill Lookup

During training, for each sampled frame, the data pipeline:

```python
def get_skill_at_frame(skill_annotations, frame_idx):
    """
    Find which skill is active at a specific frame.

    This is the KEY function that maps frame index → skill.
    """
    for skill in skill_annotations:
        start_frame, end_frame = skill["frame_duration"]
        if start_frame <= frame_idx < end_frame:
            return skill  # Found the active skill!

    return None  # No skill at this frame (gap)
```

**Example lookups for episode_00000010**:
```
Frame 0:    → Skill 0 ("move to", frames 0-264)
Frame 100:  → Skill 0 ("move to", frames 0-264)
Frame 265:  → Skill 1 ("pick up from", frames 265-1161) ← First frame of skill
Frame 500:  → Skill 1 ("pick up from", frames 265-1161) ← Middle of skill
Frame 1000: → Skill 1 ("pick up from", frames 265-1161) ← Still in skill
Frame 1161: → Skill 1 ("pick up from", frames 265-1161) ← Last frame of skill
Frame 1162: → Skill 2 ("press", frames 1162-1433)
```

### Step 2: Dense Skill Annotation

With **dense prediction** enabled (unified model), EVERY frame gets the skill label:

```python
# In unified config: use_dense_prediction = True

# Training batch at frame 500 (middle of "pick up from" skill)
batch = {
    "observation": {
        "images": [...],  # Vision at frame 500
        "state": [...],   # Robot state at frame 500
    },
    "skill_text": '{"skill":"pick up from","obj":"radio","type":"uncoordinated"}',
    "skill_tokens": [token_ids...],
    "skill_mask": [1, 1, 1, ..., 1],  # All valid
    "predict_skill": True,  # ← Dense mode: ALWAYS True
    "is_skill_end_frame": False,  # Not at frame 1161 yet
}

# Training batch at frame 1161 (LAST frame of "pick up from" skill)
batch = {
    "observation": {
        "images": [...],  # Vision at frame 1161 (robot just grasped radio)
        "state": [...],
    },
    "skill_text": '{"skill":"pick up from","obj":"radio","type":"uncoordinated"} <EOS_SKILL>',
    "skill_tokens": [token_ids..., 257153],  # 257153 = <EOS_SKILL> token
    "predict_skill": True,
    "is_skill_end_frame": True,  # ← CRITICAL: Marks skill boundary!
}
```

### Step 3: Model Learns Visual-Temporal Correlation

The model learns to associate **visual observations** with **skill labels** through dense supervision:

```
┌─────────────────────────────────────────────────────────────┐
│                  TEMPORAL LEARNING PROCESS                   │
└─────────────────────────────────────────────────────────────┘

Episode: "turning on radio" (1776 frames, 5 skills)

Skill 0: "move to" radio (frames 0-264)
├─ Frame 0:   Vision shows robot far from radio
│             → Model learns to predict: {"skill":"move to","obj":"radio"}
├─ Frame 100: Vision shows robot navigating towards radio
│             → Model learns to predict: {"skill":"move to","obj":"radio"}
└─ Frame 264: Vision shows robot reached radio
              → Model learns to predict: {"skill":"move to","obj":"radio"} <EOS_SKILL>
              (EOS appears because frame 264 is last frame of skill 0)

Skill 1: "pick up from" (frames 265-1161) ← 897 frames!
├─ Frame 265:  Vision shows robot's hand approaching radio
│              → Model learns to predict: {"skill":"pick up from","obj":"radio"}
├─ Frame 500:  Vision shows hand grasping radio
│              → Model learns to predict: {"skill":"pick up from","obj":"radio"}
├─ Frame 800:  Vision shows hand lifting radio
│              → Model learns to predict: {"skill":"pick up from","obj":"radio"}
└─ Frame 1161: Vision shows radio fully grasped and lifted
               → Model learns to predict: {"skill":"pick up from","obj":"radio"} <EOS_SKILL>
               (EOS appears because frame 1161 is last frame of skill 1)

Skill 2: "press" button (frames 1162-1433)
├─ Frame 1162: Vision shows hand with radio, approaching button
│              → Model learns to predict: {"skill":"press","obj":"radio"}
└─ Frame 1433: Vision shows button pressed
               → Model learns to predict: {"skill":"press","obj":"radio"} <EOS_SKILL>
```

---

## Key Insight: Vision Encodes Implicit Timing

The model **never receives explicit frame numbers** as input! Instead, it learns that:

1. **Visual appearance changes over time** correlate with skill progress
2. **Within a skill**, different visual states (early, middle, late) all map to the same skill
3. **Skill boundaries** are marked by visual transitions + EOS token

```
Visual State                           → Predicted Skill
───────────────────────────────────────────────────────────────
Robot far from radio (frame 0-100)     → "move to"
Robot near radio (frame 200-264)       → "move to"
Hand approaching radio (frame 265-400) → "pick up from"
Hand grasping radio (frame 500-800)    → "pick up from"
Hand lifting radio (frame 900-1161)    → "pick up from"
Hand with radio, near button (1162+)   → "press"
```

The model learns a **visual representation** that implicitly captures:
- Where the robot is in the skill execution
- What objects are being manipulated
- How far along the skill has progressed

---

## Inference: Generating Skills from Vision Alone

At inference, the model has **no access to frame_duration annotations**. Instead:

### Step 1: Initial Frame (Vision Only)

```python
# Frame 0: Robot just spawned
vision = [base_cam, left_wrist, right_wrist]  # Shows robot far from target

# Model forward pass
skill_json, has_eos = model.predict_autoregressive(
    vision=vision,
    task_prompt="Turn on the radio",
    memory_tokens=None  # No past skills yet
)

# Output (generated by skill head):
skill_json = '{"skill":"move to","obj":"radio","type":"navigation"}'
has_eos = False  # Model predicts skill is NOT done yet
```

**How does the model know to predict "move to"?**
- Vision shows robot is far from target
- Training examples taught: "far from object" → "move to"
- Model generalizes this pattern to new episodes

### Step 2: Middle of Skill (Vision + Memory)

```python
# Frame 150: Robot is navigating
vision = [...]  # Shows robot moving towards radio

skill_json, has_eos = model.predict_autoregressive(
    vision=vision,
    task_prompt="Turn on the radio",
    memory_tokens=None  # Still no completed skills
)

# Output:
skill_json = '{"skill":"move to","obj":"radio","type":"navigation"}'
has_eos = False  # Still not done navigating
```

**Why same skill as frame 0?**
- Visual state shows ongoing navigation
- Model learned: "still approaching object" → same "move to" skill
- No EOS because visual cues don't indicate completion

### Step 3: Skill Completion (EOS Prediction)

```python
# Frame 265: Robot reached radio
vision = [...]  # Shows robot AT radio, hand extended

skill_json, has_eos = model.predict_autoregressive(
    vision=vision,
    task_prompt="Turn on the radio",
    memory_tokens=None
)

# Output:
skill_json = '{"skill":"move to","obj":"radio","type":"navigation"}'
has_eos = True  # ← MODEL PREDICTS EOS!
```

**How does the model know the skill ended?**
- Training showed: "reached target object" → append <EOS_SKILL>
- Visual cues: robot stopped moving, hand at object
- Model learned this pattern from 100% of skill boundaries in training

### Step 4: Next Skill (With Memory)

```python
# Frame 266: Robot starts picking up
vision = [...]  # Shows hand approaching radio for grasp

# Memory now has past skill!
memory_tokens = tokenize(
    '<PAST_SKILL>{"skill":"move to","obj":"radio"}</PAST_SKILL>'
)

skill_json, has_eos = model.predict_autoregressive(
    vision=vision,
    task_prompt="Turn on the radio",
    memory_tokens=memory_tokens  # ← Context from past!
)

# Output:
skill_json = '{"skill":"pick up from","obj":"radio","type":"uncoordinated"}'
has_eos = False
```

**How does memory help?**
- Model knows: "just finished 'move to' → likely 'pick up' or 'open' next"
- Task context: "turn on radio" → need to pick up radio first
- Visual state + memory + task → predict "pick up from"

---

## Training vs Inference Comparison

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Frame index source** | Dataset provides frame_idx | Environment step counter |
| **Skill annotation** | Loaded from JSON via frame_duration | Generated by skill head |
| **Temporal mapping** | `get_skill_at_frame(frame_idx)` | Implicit in vision encoder |
| **Skill duration** | Explicit: frames 265-1161 | Implicit: until EOS predicted |
| **EOS supervision** | Annotation says frame 1161 is last | Model predicts when done |
| **Memory** | Extracted from past annotations | Built from generated skills |

---

## Code Implementation

### Training: Frame → Skill Mapping

```python
# In hierarchical_transforms.py: AddSkillAnnotation.__call__()

# 1. Get current frame index
frame_idx = data["index"]  # e.g., 500

# 2. Find which skill is active at this frame
skill = skill_utils.get_skill_at_frame(
    annotation["skill_annotation"],
    frame_idx  # Look up: which skill owns frame 500?
)
# Returns: {"skill_description": ["pick up from"], "frame_duration": [265, 1162], ...}

# 3. Check if this is the LAST frame of the skill
start_frame, end_frame = skill["frame_duration"]  # [265, 1162]
is_last_frame = (frame_idx == end_frame)  # False (500 != 1162)

# 4. Generate skill text
skill_text = '{"skill":"pick up from","obj":"radio","type":"uncoordinated"}'

# 5. Append EOS token ONLY at last frame
if use_eos_token and is_last_frame:
    skill_text = f"{skill_text} <EOS_SKILL>"
    # This only happens at frame 1161 (end_frame - 1, since end_frame is exclusive)

# 6. Add to batch
data["skill_text"] = skill_text
data["predict_skill"] = True  # Dense mode: always True
data["is_skill_end_frame"] = is_last_frame
```

### Inference: Vision → Skill Generation

```python
# In pi0_hierarchical.py: predict_autoregressive()

def predict_autoregressive(self, observation, memory_tokens=None):
    """Generate skill tokens until EOS or max length."""

    # 1. Embed vision + memory + task prompt
    if memory_tokens is not None:
        prefix_tokens = embed_prefix_with_memory(
            observation, memory_tokens
        )
    else:
        prefix_tokens = embed_prefix(observation)

    # 2. Autoregressive generation
    skill_tokens = []
    for i in range(max_skill_tokens):
        # Concatenate prefix + generated tokens
        input_tokens = jnp.concatenate([prefix_tokens, skill_tokens])

        # Forward pass through PaliGemma expert
        logits = self.skill_head(input_tokens)

        # Sample next token
        next_token = jnp.argmax(logits[-1])

        # Check for EOS
        if next_token == 257153:  # <EOS_SKILL>
            has_eos = True
            break

        skill_tokens.append(next_token)

    # 3. Decode to JSON
    skill_json = tokenizer.decode(skill_tokens)

    return skill_json, has_eos

# The model predicts EOS based on:
# - Visual state: "grasp completed"
# - Learned pattern: "completed grasp" → <EOS_SKILL>
# - NOT from explicit frame counting!
```

---

## Why Dense Prediction is Critical

Dense prediction (predicting skill at EVERY frame) is essential for learning robust visual-temporal representations:

### Sparse Prediction (Old Phase 0)
```
Frames 265-274: Predict "pick up from"  ← Only 10 frames
Frames 275-1161: Skip                   ← 886 frames WASTED!
```

**Problems:**
- Model sees "hand approaching" (frame 265) → "pick up from" ✓
- Model NEVER sees "hand grasping" (frame 500) → "pick up from" ✗
- Model NEVER sees "hand lifting" (frame 800) → "pick up from" ✗
- Generalization is poor: can't recognize skill in middle/late stages

### Dense Prediction (Unified Model)
```
Frames 265-1161: Predict "pick up from"  ← ALL 897 frames!
```

**Benefits:**
- Model sees all visual states within skill
- Learns: "approaching" → "pick up from"
- Learns: "grasping" → "pick up from"
- Learns: "lifting" → "pick up from"
- Learns: "fully grasped" + EOS → skill complete
- **Strong generalization**: recognizes skill at any stage

---

## EOS Token Learning

The `<EOS_SKILL>` token is added ONLY at the last frame of each skill:

```python
# Training batch construction
for frame_idx in range(episode_length):
    skill = get_skill_at_frame(annotations, frame_idx)

    start, end = skill["frame_duration"]
    is_last = (frame_idx == end - 1)  # -1 because end is exclusive

    skill_text = create_skill_json(skill)

    if is_last:
        skill_text += " <EOS_SKILL>"  # Only at boundary!

    batch["skill_text"] = skill_text
```

**Training examples:**
```
Frame 264 (last of "move to"):
  → '{"skill":"move to","obj":"radio"} <EOS_SKILL>'

Frame 1161 (last of "pick up from"):
  → '{"skill":"pick up from","obj":"radio"} <EOS_SKILL>'

Frame 1433 (last of "press"):
  → '{"skill":"press","obj":"radio"} <EOS_SKILL>'
```

The model learns: **"Certain visual states (task completion) → predict EOS"**

At inference, the model autonomously predicts EOS when it recognizes completion patterns.

---

## Summary

### The Complete Learning Flow

1. **Annotation provides temporal ground truth**:
   - `frame_duration: [265, 1162]` defines skill boundaries

2. **Data pipeline maps frames → skills**:
   - `get_skill_at_frame(500)` → "pick up from"
   - Dense: ALL frames 265-1161 get this label

3. **Model learns vision → skill mapping**:
   - Frame 265 vision → "pick up from"
   - Frame 500 vision → "pick up from"
   - Frame 1161 vision → "pick up from" + EOS

4. **EOS marks boundaries**:
   - Only frame 1161 (last frame) gets EOS token
   - Model learns: "completion visual state" → EOS

5. **Inference uses learned patterns**:
   - Vision (no frame info!) → predict skill
   - Continues predicting same skill until visual state changes
   - Predicts EOS when recognizes completion
   - Updates memory with completed skill

### Key Takeaway

**The model NEVER sees explicit timing at inference** - it learns an implicit visual representation that captures:
- What skill is being executed (from visual appearance)
- How far along the skill is (from visual state)
- When the skill completes (from visual completion cues → EOS)

This is possible because **dense prediction** provides rich supervision across all visual states within each skill during training.

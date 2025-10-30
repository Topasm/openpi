# Unified Hierarchical VLA - Architecture & Training

Single unified model combining dense prediction, memory, and boundary detection for BEHAVIOR-1K.

## Overview

The Unified Hierarchical VLA is a single-stage training approach that combines the best features from previous multi-phase approaches:

✅ **Dense Prediction**: Predict skills at EVERY frame (not just first N frames)
✅ **Memory Integration**: Use past skills as text context for long-horizon tasks
✅ **EOS Boundary Detection**: Model learns when skills end via `<EOS_SKILL>` tokens
✅ **Single Training**: No need to switch between phases - one continuous training

### Key Advantages

| Feature | Benefit |
|---------|---------|
| **Dense Prediction** | Better skill-action alignment, more training signal |
| **Memory** | Handle episodes longer than context window (~10-15 past skills) |
| **EOS Tokens** | Model autonomously detects skill boundaries at inference |
| **Unified Training** | Simpler pipeline, no phase switching, faster iteration |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED HIERARCHICAL VLA                      │
└─────────────────────────────────────────────────────────────────┘

INPUT:
┌──────────────┐   ┌────────────────┐   ┌─────────────────┐
│ Past Skills  │ + │ Current Vision │ + │  Task Prompt    │
│  (Memory)    │   │  (3 cameras)   │   │ "Put trash..."  │
│<PAST_SKILL>  │   │  224x224x3×3   │   │                 │
│{...}         │   │                │   │                 │
│</PAST_SKILL> │   │                │   │                 │
└──────┬───────┘   └───────┬────────┘   └────────┬────────┘
       │                   │                     │
       └───────────────────┼─────────────────────┘
                           │
                    Tokenize & Embed
                           │
                           ▼
              ┌────────────────────────┐
              │  Multi-Expert LLM      │
              │  (PaliGemma + Action)  │
              └────────┬───────────────┘
                       │
          ┏━━━━━━━━━━━━┻━━━━━━━━━━━━┓
          ▼                          ▼
    ┌──────────────┐         ┌──────────────┐
    │ SKILL HEAD   │         │ ACTION HEAD  │
    │ (Every Frame)│         │ (Flow Match) │
    └──────┬───────┘         └──────┬───────┘
           │                        │
           ▼                        ▼
    {"skill":"X",           [dx,dy,dz,...]
     "obj":"Y"}             × 50 frames
     <EOS_SKILL> ◄─── (at skill end only)


TRAINING SIGNAL (Dense):
┌────────────────────────────────────────────────────┐
│  Episode: 500 frames, 5 skills                     │
├────────────────────────────────────────────────────┤
│  Skill 0 (100 frames)                              │
│  ├─ Frame 0:   Predict {"skill":"move to",...}    │
│  ├─ Frame 1:   Predict {"skill":"move to",...}    │
│  ├─ ...                                            │
│  ├─ Frame 99:  Predict {"skill":"move to",...}    │
│  └─ Frame 100: Predict {...} <EOS_SKILL> ◄─ end   │
│                                                     │
│  Skill 1 (120 frames)                              │
│  ├─ Frame 101: Predict {"skill":"pick up",...}    │
│  ├─ ...                                            │
│  └─ Frame 220: Predict {...} <EOS_SKILL>          │
│                                                     │
│  ... (continues for all frames)                    │
│                                                     │
│  Training signal: 100% of frames predict skills    │
│                   EOS only at skill boundaries     │
└────────────────────────────────────────────────────┘
```

---

## Training Flow

### Single-Stage Training

```
┌─────────────────────────────────────────────────────────┐
│              UNIFIED TRAINING PIPELINE                   │
└─────────────────────────────────────────────────────────┘

1. DATA LOADING
   ├─ Load episode from LeRobot dataset
   ├─ Load skill annotations from JSON
   └─ Extract past skills for memory context

2. DENSE ANNOTATION
   For each frame in episode:
   ├─ Current skill JSON → tokens
   ├─ Append <EOS_SKILL> if last frame of skill
   └─ Past skills → <PAST_SKILL>{...}</PAST_SKILL> tokens

3. FORWARD PASS
   ├─ Embed: [memory + vision + prompt]
   ├─ Multi-expert LLM processes tokens
   ├─ Skill head: Predict skill tokens (dense)
   └─ Action head: Predict actions (flow matching)

4. LOSS COMPUTATION
   L_total = 10.0 × L_skill + 1.0 × L_action

   Where:
   • L_skill: Cross-entropy on skill tokens
   • L_action: MSE on flow matching velocity

5. OPTIMIZATION
   └─ AdamW optimizer, LoRA fine-tuning
```

### Memory Context Example

```python
# Episode frame 250 (in middle of skill 2)
# Past skills compressed as text:
memory = [
    '<PAST_SKILL>{"skill":"navigate to","target":"door"}</PAST_SKILL>',
    '<PAST_SKILL>{"skill":"open","object":"door"}</PAST_SKILL>',
]

# Current observation
vision = [base_camera, left_wrist, right_wrist]  # 224×224×3 each
prompt = "Complete the household task"

# Current skill (frame 250 is NOT end of skill)
current_skill = '{"skill":"pick up","object":"cup"}'
# No <EOS_SKILL> token yet

# At frame 300 (END of skill 2):
current_skill = '{"skill":"pick up","object":"cup"} <EOS_SKILL>'
# Now has EOS token

# Model learns to predict <EOS_SKILL> when skill completes
```

---

## Implementation

### Configuration

All settings are now unified in one config:

```python
# Location: src/openpi/training/config.py
LeRobotB1KHierarchicalDataConfig(
    annotation_root="dataset/2025-challenge-demos/annotations",
    enable_memory=True,              # Always enabled
    use_dense_prediction=True,       # Always enabled
    use_eos_token=True,              # Always enabled
    use_concise_format=True,         # Always enabled
    use_hierarchical_tokenizer=True, # Always enabled
)
```

### Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `enable_memory` | `True` | Use past skills as text context |
| `use_dense_prediction` | `True` | Predict at every frame |
| `use_eos_token` | `True` | Add `<EOS_SKILL>` at boundaries |
| `use_concise_format` | `True` | Efficient JSON format |
| `use_hierarchical_tokenizer` | `True` | Special tokens support |

### Training Command

```bash
# Single unified training - no phase switching needed!
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_hierarchical.py pi0_b1k_hierarchical \
  --exp-name="unified_$(date +%Y%m%d_%H%M%S)" \
  --overwrite \
  --batch_size=32 \
  --num_train_steps=50000 \
  --weight_loader.params_path=gs://openpi-assets/checkpoints/pi0_base/params
```

That's it! No need to change config flags or restart training for different phases.

---

## Data Flow

### Hierarchical Transforms Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                  DATA TRANSFORMS                         │
└─────────────────────────────────────────────────────────┘

1. AddSkillAnnotation
   ├─ Load skill JSON for episode
   ├─ Dense prediction: ALL frames get skill labels
   ├─ Append <EOS_SKILL> at last frame of each skill
   └─ Output: skill_text, predict_skill=True (all frames)

2. AddPastSkillsCache (Memory)
   ├─ Extract past completed skills from episode
   ├─ Format: <PAST_SKILL>{skill JSON}</PAST_SKILL>
   ├─ Limit: ~256 tokens (~10-15 skills)
   └─ Output: memory_text

3. TokenizeSkills
   ├─ Use HierarchicalTokenizer (with special tokens)
   ├─ Tokenize current skill JSON
   ├─ Add <EOS_SKILL> token at boundaries
   └─ Output: skill_tokens [B, 64], skill_mask [B, 64]

4. TokenizeMemory
   ├─ Tokenize past skills memory
   └─ Output: memory_tokens [B, 256], memory_mask [B, 256]

OUTPUT BATCH:
├─ observation: {images, state, tokenized_prompt}
├─ actions: [B, 50, 32]
└─ batch_dict:
    ├─ skill_tokens: [B, 64]
    ├─ skill_mask: [B, 64]
    ├─ memory_tokens: [B, 256]
    ├─ memory_mask: [B, 256]
    └─ has_skill: [B] (all True in dense mode)
```

---

## Model Architecture

### Components

```python
# Vision Encoder: SigLIP-So400m (frozen)
Input: 3 × [224, 224, 3]
Output: 3 × [256 tokens, 2048 dim]

# Multi-Expert Transformer
Expert 0 (PaliGemma): Gemma 2B with LoRA
├─ 18 layers, 2048 hidden dim
├─ Processes: memory + vision + language
└─ LoRA rank 16

Expert 1 (Action): Gemma 300M with LoRA
├─ 18 layers, 1024 hidden dim
├─ Processes: noisy actions + time
└─ LoRA rank 32

# Dual Prediction Heads
Skill Head:
├─ Decode to vocab (257152 tokens)
├─ Cross-entropy loss
└─ Output: [B, seq_len, vocab_size]

Action Head:
├─ Flow matching
├─ MSE loss on velocity
└─ Output: [B, 50, 32]
```

### Loss Weights

```python
L_total = 10.0 × L_skill + 1.0 × L_action

# Higher weight for skills because:
# - Skills are more abstract (harder to learn)
# - Enables hierarchical reasoning
# - Smaller effective "batch size" per unique skill
```

---

## Inference

At inference time, the unified model autonomously generates skills using **only vision and task prompt** as inputs. The model learns to:

1. **Generate skill JSON autoregressively** via the skill head
2. **Predict `<EOS_SKILL>` token** when skill completes
3. **Build memory dynamically** by caching completed skills
4. **Use past skills as context** for subsequent predictions

### Detailed Inference Logic

#### Starting from Scratch (Frame 0)

```
┌─────────────────────────────────────────────────────────┐
│                   COLD START (Frame 0)                   │
└─────────────────────────────────────────────────────────┘

INPUT (no memory yet):
├─ Vision: [base_cam, left_wrist, right_wrist] (224×224×3)
├─ Task Prompt: "Put the trash in the bin"
└─ Memory: [] (empty!)

FORWARD PASS:
1. Vision Encoder (SigLIP)
   └─ Encode 3 cameras → 3×[256 tokens, 2048 dim]

2. Embed Prefix (NO memory)
   ├─ Memory tokens: None
   ├─ Vision tokens: [768 tokens, 2048 dim]
   └─ Task prompt tokens: [N tokens, 2048 dim]

3. PaliGemma Expert (Skill Head)
   ├─ Autoregressive generation
   ├─ Sample tokens from vocab (257153 tokens total)
   ├─ Generate: '{"skill":"navigate to","target":"trash"}'
   └─ Continue until:
       • <EOS_SKILL> token (ID: 257153) predicted → STOP
       • OR max_skill_tokens (64) reached

OUTPUT:
├─ Skill JSON: {"skill":"navigate to","target":"trash"}
├─ EOS detected: False (skill still ongoing)
└─ Actions: [50, 32] from flow matching
```

#### Middle of Episode (Frame 150, after 2 skills completed)

```
┌─────────────────────────────────────────────────────────┐
│              WITH MEMORY (Frame 150)                     │
└─────────────────────────────────────────────────────────┘

INPUT (memory built from past):
├─ Vision: [base_cam, left_wrist, right_wrist]
├─ Task Prompt: "Put the trash in the bin"
└─ Memory: [
      '<PAST_SKILL>{"skill":"navigate to","target":"trash"}</PAST_SKILL>',
      '<PAST_SKILL>{"skill":"pick up","object":"trash"}</PAST_SKILL>'
    ]

FORWARD PASS:
1. Vision Encoder (SigLIP)
   └─ Encode 3 cameras → 3×[256 tokens, 2048 dim]

2. Embed Prefix WITH Memory
   ├─ Memory tokens: [256 tokens, 2048 dim] ◄─ past skills!
   ├─ Vision tokens: [768 tokens, 2048 dim]
   └─ Task prompt tokens: [N tokens, 2048 dim]
   Total prefix: [256 + 768 + N tokens]

3. PaliGemma Expert (Skill Head)
   ├─ Condition on memory context
   ├─ Generate next skill autoregressively:
   │   • "{"skill":"navigate to","target":"bin"}" (model knows to go to bin now)
   └─ Predict EOS: False (still navigating)

OUTPUT:
├─ Skill JSON: {"skill":"navigate to","target":"bin"}
├─ EOS detected: False
└─ Actions: [50, 32]
```

#### Skill Completion (Frame 200, EOS detected)

```
┌─────────────────────────────────────────────────────────┐
│           EOS DETECTION (Frame 200)                      │
└─────────────────────────────────────────────────────────┘

INPUT:
├─ Vision: Robot reached bin
├─ Memory: [2 past skills...]
└─ Task Prompt: "Put the trash in the bin"

FORWARD PASS:
1-2. Same as before

3. PaliGemma Expert (Skill Head)
   ├─ Generate: '{"skill":"navigate to","target":"bin"}'
   ├─ Predict next token: <EOS_SKILL> ◄─ MODEL PREDICTS END!
   └─ Autoregressive generation STOPS

OUTPUT:
├─ Skill JSON: {"skill":"navigate to","target":"bin"}
├─ EOS detected: True ◄─ SKILL COMPLETE!
└─ Actions: [50, 32]

POST-PROCESSING (Update Memory):
memory.append(
    '<PAST_SKILL>{"skill":"navigate to","target":"bin"}</PAST_SKILL>'
)
# Memory now has 3 skills
```

### Complete Inference Loop

```python
def inference_loop(env, model, task_prompt, max_steps=1000):
    """
    Full inference logic showing autonomous skill generation.

    Key insight: Model generates skills from scratch using only
    vision + task prompt. Memory is built up dynamically.
    """
    # Initialize
    observation = env.reset()
    memory = []  # Empty at start!
    done = False
    step = 0

    while not done and step < max_steps:
        # ============================================
        # STEP 1: Prepare inputs
        # ============================================
        vision = extract_vision(observation)  # [3, 224, 224, 3]

        # Memory tokens (None if no past skills yet)
        if len(memory) > 0:
            memory_text = ''.join(memory)
            memory_tokens, memory_mask = tokenize_memory(memory_text)
        else:
            memory_tokens, memory_mask = None, None

        # ============================================
        # STEP 2: Forward pass - Generate skill
        # ============================================
        # This is where the magic happens!
        skill_json, has_eos, actions = model.predict_autoregressive(
            vision=vision,
            task_prompt=task_prompt,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask
        )

        # Internally, predict_autoregressive does:
        # 1. Embed: [memory_tokens (if any) + vision + task_prompt]
        # 2. PaliGemma expert processes prefix
        # 3. Skill head generates tokens one-by-one:
        #    - Sample from vocab distribution
        #    - Append to sequence
        #    - Continue until <EOS_SKILL> (ID: 257153) or max length
        # 4. Action head predicts via flow matching

        # ============================================
        # STEP 3: Execute actions
        # ============================================
        observation, reward, done = env.step(actions[0])

        # ============================================
        # STEP 4: Update memory if skill completed
        # ============================================
        if has_eos:
            # Skill completed! Add to memory
            past_skill_text = f'<PAST_SKILL>{skill_json}</PAST_SKILL>'
            memory.append(past_skill_text)

            # Keep only last 15 skills (~256 tokens max)
            if len(memory) > 15:
                memory = memory[-15:]

            print(f"[Step {step}] Skill completed: {skill_json}")
            print(f"[Step {step}] Memory size: {len(memory)} skills")

        step += 1

    return observation, step
```

### Autoregressive Skill Generation (Detailed)

The **skill head** generates JSON tokens one at a time:

```python
def predict_autoregressive(model, prefix_tokens, max_tokens=64):
    """
    How the skill head generates skill JSON autoregressively.

    Input: prefix_tokens = [memory + vision + task_prompt]
    Output: skill_json, has_eos
    """
    skill_tokens = []
    has_eos = False

    for i in range(max_tokens):
        # Concatenate prefix + generated tokens so far
        input_tokens = jnp.concatenate([prefix_tokens, skill_tokens])

        # Forward pass through PaliGemma expert
        logits = model.skill_head(input_tokens)  # [seq_len, vocab_size]

        # Sample next token
        next_token_logits = logits[-1]  # [vocab_size]
        next_token = jnp.argmax(next_token_logits)  # Greedy decoding

        # Check if EOS token
        if next_token == 257153:  # <EOS_SKILL>
            has_eos = True
            break

        # Append to sequence
        skill_tokens.append(next_token)

    # Decode tokens to JSON string
    skill_json = tokenizer.decode(skill_tokens)

    return skill_json, has_eos

# Example output:
# skill_json = '{"skill":"pick up","object":"trash"}'
# has_eos = False
```

### Key Insights

1. **No Ground Truth at Inference**: Unlike training (where skill annotations are provided), at inference the model generates skills completely autonomously.

2. **Memory is Self-Built**: The model starts with empty memory and builds it up by:
   - Generating skills autoregressively
   - Detecting EOS boundaries
   - Caching completed skills as `<PAST_SKILL>{...}</PAST_SKILL>`

3. **Context Window Management**: Memory keeps last ~15 skills (~256 tokens) to stay within context window while providing long-horizon context.

4. **Dense Prediction Training Enables This**: Because the model was trained to predict skills at EVERY frame, it learns robust skill representations that generalize to novel observations at inference.

5. **EOS Detection is Learned**: The model learns when to emit `<EOS_SKILL>` by observing skill boundaries in training data. At inference, this allows autonomous skill segmentation.

### Comparison: Training vs Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| Skill source | Ground truth annotations | Generated by skill head |
| Memory source | Extracted from annotations | Built from generated skills |
| EOS supervision | Annotated boundaries | Predicted by model |
| Input | Vision + prompt + GT skill + GT memory | Vision + prompt only |
| Output mode | Teacher forcing | Autoregressive generation |

### How Temporal Information Works

**Critical question: How does the model learn that frames 265-1162 correspond to "pick up from" skill?**

The answer is **temporal supervision via dense prediction**:

1. **Annotation Structure**: Each skill has `frame_duration: [start, end]`
   ```json
   {
     "skill_description": ["pick up from"],
     "frame_duration": [265, 1162],  // 897 frames
     "object_id": [["radio_89", "coffee_table_koagbh_0"]]
   }
   ```

2. **Training Pipeline**: For each frame, look up which skill is active
   ```python
   # Frame 500 (middle of skill)
   skill = get_skill_at_frame(annotations, frame_idx=500)
   # Returns: "pick up from" skill (frames 265-1162)

   # Frame 1161 (last frame of skill)
   skill = get_skill_at_frame(annotations, frame_idx=1161)
   # Returns: "pick up from" + <EOS_SKILL> appended
   ```

3. **Dense Supervision**: ALL 897 frames (265-1161) get labeled as "pick up from"
   - Frame 265: Vision shows hand approaching → "pick up from"
   - Frame 500: Vision shows hand grasping → "pick up from"
   - Frame 800: Vision shows hand lifting → "pick up from"
   - Frame 1161: Vision shows grasp complete → "pick up from" + `<EOS_SKILL>`

4. **Model Learns Visual-Temporal Patterns**:
   - NOT explicit frame counting
   - Learns that different visual states (approach, grasp, lift) → same skill
   - Learns that completion visual cues → `<EOS_SKILL>` token

5. **Inference (No Frame Info)**:
   - Model only sees: vision + task prompt + past skills memory
   - Predicts skill based on learned visual patterns
   - Continues same skill until visual state indicates completion
   - Predicts `<EOS_SKILL>` when recognizes completion cues

**See [TEMPORAL_SKILL_MAPPING.md](TEMPORAL_SKILL_MAPPING.md) for detailed explanation with examples.**

---

## Performance

### Expected Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Training time | ~14 hours | 50K steps, batch_size=32, A100 80GB |
| GPU memory | ~60GB | With XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 |
| Step time | 10-25s | Depends on sequence length with memory |
| Skill accuracy | TBD | Measured by exact match on JSON |
| Action MSE | TBD | Measured on normalized actions |

### Advantages over Multi-Phase

| Aspect | Multi-Phase | Unified |
|--------|-------------|---------|
| Training steps | 150K (3×50K) | 50K |
| Wall-clock time | ~42 hours | ~14 hours |
| Complexity | High (3 configs) | Low (1 config) |
| Skill signal | Sparse→Dense | Dense from start |
| Memory | Added later | Integrated from start |

---

## Files Modified

```
b1k-baselines/baselines/openpi/
├── src/openpi/training/
│   └── config.py
│       ├── LeRobotB1KHierarchicalDataConfig (updated defaults)
│       └── pi0_b1k_hierarchical config (unified settings)
│
└── docs/
    └── UNIFIED_ARCHITECTURE.md (this file)
```

---

## Migration from Multi-Phase

If you have checkpoints from the old multi-phase training:

```bash
# Old Phase 0 checkpoint → Can continue training
# Will now use dense prediction + memory + EOS

# Old Phase 1 checkpoint → Can continue training
# Will now also use dense prediction + EOS

# Old Phase 3 checkpoint → Fully compatible!
# Already uses dense + memory + EOS
```

No special migration needed - just start training with the new config!

---

## Troubleshooting

### Issue: "Too many skill tokens"
**Solution**: The model has max_skill_tokens=64. If skills are very long, they'll be truncated. Consider increasing this or using more concise skill descriptions.

### Issue: "Memory tokens not found in batch"
**Solution**: Memory is only added when there are past skills. Early in an episode, memory_tokens may be None - this is expected and handled automatically.

### Issue: "EOS token not in vocabulary"
**Solution**: Make sure `use_hierarchical_tokenizer=True` in config. The base PaligemmaTokenizer doesn't have the `<EOS_SKILL>` token.

---

## Next Steps

1. **Train the unified model**: Run the training command above
2. **Monitor metrics**: Check WandB for skill_loss, action_loss, and EOS prediction accuracy
3. **Evaluate on validation set**: Use val_episodes_index=list(range(190, 200))
4. **Deploy for inference**: Use the trained model with memory management

For more implementation details, see:
- [pi0_hierarchical.py](../src/openpi/models/pi0_hierarchical.py) - Model implementation
- [hierarchical_transforms.py](../src/openpi/training/hierarchical_transforms.py) - Data transforms
- [train_hierarchical.py](../scripts/train_hierarchical.py) - Training script
- [TEMPORAL_SKILL_MAPPING.md](TEMPORAL_SKILL_MAPPING.md) - **How frame_duration enables skill learning**

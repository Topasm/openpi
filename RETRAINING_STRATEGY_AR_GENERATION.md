# 🎓 Retraining Strategy: Proper Autoregressive Skill Generation

## Goal

Train the model to generate skill JSON text autoregressively during inference, fixing the exposure bias problem that causes "cococo..." loops.

## Why Retraining is Needed

### Current Training (Teacher Forcing Only)

The model is currently trained with **100% teacher forcing**:

```python
# Training: Ground truth skills always provided
(prefix_hidden, _), _ = llm([prefix_tokens, None], ...)  # Vision + prompt
vocab_logits = project_to_vocab(prefix_hidden)
loss = cross_entropy(vocab_logits, ground_truth_skill_tokens)  # ← Always GT!
```

**Problem**: Model never sees its own predictions during training, so during inference:
- Generated token → Never seen before → Model confused → Generates garbage
- This is **exposure bias** / **distribution shift**

### Desired Training (Mixed Teacher Forcing + Autoregressive)

Mix ground truth with model predictions during training:

```python
# Training: Sometimes use GT, sometimes use model's own predictions
for token_idx in range(skill_length):
    if random() < teacher_forcing_ratio:  # ← Scheduled sampling!
        prev_token = ground_truth[token_idx]
    else:
        prev_token = model_prediction[token_idx]

    logits = model(prev_token)
    loss = cross_entropy(logits, ground_truth[token_idx + 1])
```

**Benefit**: Model learns to recover from its own mistakes, matches inference distribution.

---

## Retraining Strategy

### Phase 1: Scheduled Sampling (Core Fix)

**Goal**: Gradually expose model to its own predictions during training.

**Implementation**:

```python
class ScheduledSamplingConfig:
    """Config for scheduled sampling during skill generation training."""
    initial_teacher_forcing: float = 1.0  # Start with 100% teacher forcing
    final_teacher_forcing: float = 0.3    # End with 30% teacher forcing
    decay_steps: int = 50000              # Decay over 50K steps
    decay_type: str = "linear"            # or "exponential"

def compute_teacher_forcing_ratio(step: int, config: ScheduledSamplingConfig) -> float:
    """Compute teacher forcing ratio for current training step."""
    if config.decay_type == "linear":
        progress = min(step / config.decay_steps, 1.0)
        ratio = config.initial_teacher_forcing - progress * (
            config.initial_teacher_forcing - config.final_teacher_forcing
        )
    elif config.decay_type == "exponential":
        ratio = config.final_teacher_forcing + (
            config.initial_teacher_forcing - config.final_teacher_forcing
        ) * (0.95 ** (step / 1000))
    else:
        raise ValueError(f"Unknown decay type: {config.decay_type}")

    return ratio
```

**Training Loop Modification**:

```python
def train_step_with_scheduled_sampling(
    model,
    observation,
    ground_truth_skill_tokens,
    skill_mask,
    step: int,
    rng: jax.random.PRNGKey,
    config: ScheduledSamplingConfig
):
    """Training step with scheduled sampling for skill generation."""

    # 1. Embed prefix (vision + prompt)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)

    # 2. Get initial hidden state
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    (prefix_hidden, _), kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=jnp.cumsum(prefix_mask, axis=1) - 1
    )

    # 3. Compute teacher forcing ratio for this step
    tf_ratio = compute_teacher_forcing_ratio(step, config)

    # 4. Autoregressive skill generation with scheduled sampling
    batch_size = observation.state.shape[0]
    skill_length = ground_truth_skill_tokens.shape[1]
    embedding_table = model.PaliGemma.llm.embedder['input_embedding']

    current_hidden = prefix_hidden
    all_logits = []
    sampled_tokens = []

    for token_idx in range(skill_length - 1):  # Predict next token
        # Project to vocabulary
        last_hidden = current_hidden[:, -1, :]
        logits = jnp.dot(last_hidden, embedding_table.value.T)
        all_logits.append(logits)

        # Scheduled sampling: Choose input for next step
        use_teacher_forcing = jax.random.bernoulli(
            rng, tf_ratio, shape=(batch_size,)
        )

        # Get next token: either ground truth or model prediction
        predicted_token = jnp.argmax(logits, axis=-1)  # [B]
        next_token = jnp.where(
            use_teacher_forcing,
            ground_truth_skill_tokens[:, token_idx],  # Use GT
            predicted_token                            # Use prediction
        )
        sampled_tokens.append(next_token)

        # Embed next token and continue (if not last step)
        if token_idx < skill_length - 2:
            next_token_expanded = next_token[:, None]
            next_embedding = model.PaliGemma.llm(next_token_expanded, method="embed")

            # Update KV cache (similar to inference)
            actual_cache_len = kv_cache[0][0].shape[1]
            new_position = jnp.array([[actual_cache_len]], dtype=jnp.int32)

            # Create attention mask
            cache_mask = jnp.ones((batch_size, 1, actual_cache_len), dtype=jnp.bool_)
            new_token_mask = jnp.ones((batch_size, 1, 1), dtype=jnp.bool_)
            full_mask = jnp.concatenate([cache_mask, new_token_mask], axis=-1)

            # Forward pass
            (next_hidden, _), kv_cache = model.PaliGemma.llm(
                [next_embedding, None],
                mask=full_mask,
                positions=new_position,
                kv_cache=kv_cache,
                adarms_cond=[None, None]
            )

            current_hidden = jnp.concatenate([current_hidden, next_hidden], axis=1)

        rng, _ = jax.random.split(rng)

    # 5. Compute loss
    stacked_logits = jnp.stack(all_logits, axis=1)  # [B, skill_len-1, vocab_size]
    targets = ground_truth_skill_tokens[:, 1:]       # [B, skill_len-1]
    mask = skill_mask[:, 1:]                         # [B, skill_len-1]

    import optax
    token_losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=stacked_logits,
        labels=targets
    )

    # Masked average
    masked_loss = token_losses * mask
    skill_loss = jnp.sum(masked_loss) / jnp.maximum(jnp.sum(mask), 1.0)

    return skill_loss, {
        "skill_loss": skill_loss,
        "teacher_forcing_ratio": tf_ratio,
        "sampled_tokens": sampled_tokens  # For logging
    }
```

**Key Changes**:
1. ✅ **Scheduled sampling**: Gradually reduce teacher forcing from 100% → 30%
2. ✅ **Bernoulli sampling**: Randomly choose GT or prediction per batch item
3. ✅ **KV cache updates**: Proper mask and position handling (using our fixed logic!)
4. ✅ **Matches inference**: Same AR loop as `generate_skill_autoregressive()`

---

### Phase 2: Curriculum Learning (Optional Enhancement)

**Goal**: Start with easier tasks, gradually increase difficulty.

**Curriculum Stages**:

1. **Stage 1 (Steps 0-10K)**: Short skills only (< 20 tokens)
   - Filter dataset to simple skills: `{"skill":"pick","obj":"cup"}`
   - tf_ratio: 1.0 → 0.7

2. **Stage 2 (Steps 10K-30K)**: Medium skills (20-40 tokens)
   - Include more complex skills: `{"skill":"move to","obj":"kitchen table"}`
   - tf_ratio: 0.7 → 0.4

3. **Stage 3 (Steps 30K+)**: All skills (any length)
   - Full dataset
   - tf_ratio: 0.4 → 0.3

**Implementation**:
```python
def get_curriculum_stage(step: int) -> dict:
    """Get curriculum parameters for current training step."""
    if step < 10000:
        return {
            "max_skill_tokens": 20,
            "tf_start": 1.0,
            "tf_end": 0.7,
            "stage": "short_skills"
        }
    elif step < 30000:
        return {
            "max_skill_tokens": 40,
            "tf_start": 0.7,
            "tf_end": 0.4,
            "stage": "medium_skills"
        }
    else:
        return {
            "max_skill_tokens": 64,
            "tf_start": 0.4,
            "tf_end": 0.3,
            "stage": "all_skills"
        }
```

---

### Phase 3: Improved Training Signal

**Goal**: Add auxiliary losses to improve AR generation quality.

#### 3.1 Next-Token Prediction Loss (Already exists)

```python
# Standard cross-entropy loss
loss = cross_entropy(predicted_tokens, ground_truth_tokens)
```

#### 3.2 Sequence-Level Loss (NEW)

**Reward complete valid sequences**:

```python
def sequence_level_loss(
    generated_tokens: jnp.ndarray,
    ground_truth: jnp.ndarray,
    tokenizer
) -> float:
    """Reward sequences that match GT exactly or are valid JSON."""

    # Decode generated sequence
    gen_text = tokenizer.decode(generated_tokens)
    gt_text = tokenizer.decode(ground_truth)

    # Exact match bonus
    exact_match = (gen_text == gt_text).astype(jnp.float32)

    # Valid JSON bonus
    try:
        import json
        json.loads(gen_text)
        valid_json = 1.0
    except:
        valid_json = 0.0

    # Sequence reward (use REINFORCE or SCST)
    reward = exact_match * 2.0 + valid_json * 1.0
    return -reward  # Negative because we minimize loss
```

#### 3.3 Contrastive Loss for Plan Embedding (NEW)

**Ensure plan embeddings are consistent across frames of same skill**:

```python
def plan_embedding_contrastive_loss(
    plan_embedding_t0: jnp.ndarray,  # Embedding at frame 0
    plan_embedding_t1: jnp.ndarray,  # Embedding at frame 1 (same skill)
    plan_embedding_neg: jnp.ndarray, # Embedding from different skill
    temperature: float = 0.1
) -> float:
    """Contrastive loss to make plan embeddings consistent within a skill."""

    # Positive pair: Same skill, different frames
    pos_sim = jnp.dot(plan_embedding_t0, plan_embedding_t1.T) / temperature

    # Negative pair: Different skills
    neg_sim = jnp.dot(plan_embedding_t0, plan_embedding_neg.T) / temperature

    # InfoNCE loss
    logits = jnp.concatenate([pos_sim, neg_sim], axis=-1)
    labels = jnp.zeros(plan_embedding_t0.shape[0], dtype=jnp.int32)  # First is positive

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(loss)
```

#### 3.4 Combined Loss

```python
total_loss = (
    1.0 * next_token_loss +           # Standard AR loss
    0.5 * sequence_level_loss +       # Sequence reward
    0.3 * plan_embedding_loss +       # Contrastive loss
    1.0 * action_loss                 # Action flow matching (unchanged)
)
```

---

### Phase 4: Data Augmentation

**Goal**: Increase diversity of skill texts for better generalization.

#### 4.1 Paraphrasing Skills

```python
# Original: {"skill":"pick up","obj":"cup"}
# Augmented:
#   {"skill":"grasp","obj":"cup"}
#   {"skill":"pick","obj":"cup"}
#   {"skill":"take","obj":"cup"}
```

**Implementation**: Use GPT-4/Claude to generate paraphrases offline, add to dataset.

#### 4.2 Object Synonyms

```python
# Original: {"skill":"move to","obj":"table"}
# Augmented:
#   {"skill":"move to","obj":"desk"}
#   {"skill":"move to","obj":"counter"}
```

#### 4.3 Varying JSON Format

```python
# Original: {"skill":"pick","obj":"cup"}
# Augmented:
#   {"obj":"cup","skill":"pick"}         # Different order
#   {"skill": "pick", "obj": "cup"}      # Extra spaces
#   {"skill":"pick","obj":"cup"}         # Compact (no spaces)
```

**Why**: Makes model robust to different JSON formatting during inference.

---

## Training Configuration

### Hyperparameters

```python
@dataclass
class ARTrainingConfig:
    # Scheduled Sampling
    initial_teacher_forcing: float = 1.0
    final_teacher_forcing: float = 0.3
    tf_decay_steps: int = 50000
    tf_decay_type: str = "linear"  # or "exponential"

    # Curriculum Learning
    use_curriculum: bool = True
    curriculum_stages: list = field(default_factory=lambda: [
        {"step": 0, "max_tokens": 20, "tf": 1.0},
        {"step": 10000, "max_tokens": 40, "tf": 0.7},
        {"step": 30000, "max_tokens": 64, "tf": 0.3},
    ])

    # Loss Weights
    next_token_loss_weight: float = 1.0
    sequence_loss_weight: float = 0.5
    plan_embedding_loss_weight: float = 0.3
    action_loss_weight: float = 1.0

    # Regularization
    skill_dropout: float = 0.1  # Dropout in skill prediction head
    label_smoothing: float = 0.05  # Prevent overconfidence

    # Optimization
    learning_rate: float = 3e-5  # Lower than initial training
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Data
    skill_augmentation: bool = True
    augmentation_prob: float = 0.3
```

### Training Schedule

**Recommended**: Fine-tune existing checkpoint (don't train from scratch!)

```python
# Load existing checkpoint
params = load_checkpoint("checkpoints/5000/params")

# Fine-tune with AR objective
trainer = Trainer(
    model=model,
    params=params,  # Start from existing weights!
    config=ARTrainingConfig(),
    dataset=hierarchical_dataset,
)

# Train for 50K-100K steps
trainer.train(
    num_steps=100000,
    eval_every=2000,
    save_every=5000,
)
```

**Why fine-tune**: Existing model already learned vision-language grounding. We just need to fix AR generation!

---

## Implementation Steps

### Step 1: Modify Training Loop

**File**: `src/openpi/training/train.py`

Add scheduled sampling to the training loop:

```python
# In train_step() function
if config.use_scheduled_sampling and skill_tokens is not None:
    # Use scheduled sampling for skill generation
    skill_loss, metrics = train_step_with_scheduled_sampling(
        model, observation, skill_tokens, skill_mask,
        step=state.step, rng=rng, config=config.scheduled_sampling
    )
else:
    # Original teacher forcing
    skill_loss = original_skill_loss(...)

# Add to metrics
metrics["teacher_forcing_ratio"] = compute_teacher_forcing_ratio(
    state.step, config.scheduled_sampling
)
```

### Step 2: Update Config

**File**: `src/openpi/training/config.py`

```python
@dataclass
class LeRobotB1KHierarchicalDataConfig(LeRobotB1KDataConfig):
    # ... existing fields ...

    # NEW: Scheduled Sampling Config
    use_scheduled_sampling: bool = True
    initial_teacher_forcing: float = 1.0
    final_teacher_forcing: float = 0.3
    tf_decay_steps: int = 50000

    # NEW: Curriculum Learning
    use_curriculum: bool = True

    # NEW: Auxiliary Losses
    use_sequence_level_loss: bool = True
    use_plan_contrastive_loss: bool = True
    sequence_loss_weight: float = 0.5
    plan_loss_weight: float = 0.3
```

### Step 3: Create Training Script

**File**: `scripts/retrain_ar_generation.py`

```python
#!/usr/bin/env python3
"""Retrain model with proper autoregressive skill generation."""

import jax
from openpi.training import config, train
from openpi.models import pi0_hierarchical

def main():
    # Load config
    cfg = config.LeRobotB1KHierarchicalDataConfig(
        # Base settings (same as before)
        use_eos_token=True,
        use_dense_prediction=True,
        use_hierarchical_tokenizer=True,
        use_concise_format=True,

        # NEW: AR training settings
        use_scheduled_sampling=True,
        initial_teacher_forcing=1.0,
        final_teacher_forcing=0.3,
        tf_decay_steps=50000,

        use_curriculum=True,
        use_sequence_level_loss=True,
        use_plan_contrastive_loss=True,
    )

    # Load existing checkpoint as starting point
    params = train.load_checkpoint(
        "dataset/openpi/checkpoints/5000/params"
    )

    # Fine-tune with AR objective
    train.run_training(
        config=cfg,
        initial_params=params,  # Start from existing!
        output_dir="outputs/ar_retrain",
        num_steps=100000,
    )

if __name__ == "__main__":
    main()
```

### Step 4: Validation

**Create AR generation benchmark**:

```python
# scripts/eval_ar_generation.py
def evaluate_ar_generation(model, test_dataset):
    """Evaluate AR generation quality."""

    metrics = {
        "exact_match": [],      # Generated skill == ground truth
        "valid_json": [],       # Is valid JSON
        "skill_name_match": [], # Correct skill name
        "object_match": [],     # Correct object
        "eos_detection": [],    # Correctly detects EOS
    }

    for batch in test_dataset:
        # Generate skill autoregressively
        generated_text, _, has_eos = model.generate_skill_autoregressive(
            rng, batch.observation, tokenizer=tokenizer
        )

        # Parse generated and ground truth
        gen_skill = parse_skill_json(generated_text)
        gt_skill = parse_skill_json(batch.skill_text)

        # Compute metrics
        metrics["exact_match"].append(gen_skill == gt_skill)
        metrics["valid_json"].append(is_valid_json(generated_text))
        metrics["skill_name_match"].append(
            gen_skill.get("skill") == gt_skill.get("skill")
        )
        metrics["object_match"].append(
            gen_skill.get("obj") == gt_skill.get("obj")
        )
        metrics["eos_detection"].append(has_eos == batch.is_last_frame)

    # Print results
    for key, values in metrics.items():
        print(f"{key}: {sum(values) / len(values):.2%}")
```

**Success criteria**:
- ✅ Valid JSON: >95%
- ✅ Skill name match: >80%
- ✅ Object match: >80%
- ✅ Exact match: >60%
- ✅ EOS detection: >85%

---

## Expected Training Time

### Computational Requirements

**Hardware**: 1x A100 80GB (same as original training)

**Time Estimate**:
- **50K steps**: ~12-15 hours
- **100K steps**: ~24-30 hours

**Why faster than original training**:
1. Fine-tuning from checkpoint (not from scratch)
2. Only updating skill generation capability
3. Vision encoder already learned

### Checkpointing Strategy

Save checkpoints at:
- Every 5K steps
- Test AR generation at each checkpoint
- Keep best 3 checkpoints based on `exact_match` metric

---

## Fallback: If Retraining Doesn't Work

If scheduled sampling still produces poor results:

### Option A: Separate AR Head

Add a separate decoder head specifically for AR generation:

```python
class Pi0HierarchicalWithARHead(nn.Module):
    # ... existing Pi0Hierarchical ...

    def setup(self):
        super().setup()
        # NEW: Separate AR decoder for skill generation
        self.ar_skill_decoder = nn.TransformerDecoder(
            num_layers=4,
            hidden_dim=2304,
            num_heads=16,
        )

    def generate_skill_ar(self, observation, ...):
        # Use AR decoder instead of LLM head
        vision_context = self.encode_vision(observation)
        skill_text = self.ar_skill_decoder.decode(
            context=vision_context,
            max_length=64,
        )
        return skill_text
```

### Option B: Retrieval-Based Generation

Instead of generating freely, retrieve from a template bank:

```python
# Template bank
SKILL_TEMPLATES = [
    '{"skill":"pick up","obj":"{{object}}"}',
    '{"skill":"move to","obj":"{{object}}"}',
    '{"skill":"place","obj":"{{object}}"}',
    # ... more templates
]

def generate_skill_retrieval(observation):
    # 1. Classify skill type
    skill_logits = model.classify_skill_type(observation)
    skill_type = argmax(skill_logits)  # "pick up", "move to", etc.

    # 2. Classify object
    object_logits = model.classify_object(observation)
    object_name = argmax(object_logits)  # "cup", "table", etc.

    # 3. Fill template
    template = SKILL_TEMPLATES[skill_type]
    skill_text = template.replace("{{object}}", object_name)

    return skill_text
```

---

## Summary

### Immediate Action Items

1. ✅ **Implement scheduled sampling** in training loop
2. ✅ **Add AR evaluation benchmark** to track progress
3. ✅ **Fine-tune from checkpoint 5000** (don't train from scratch!)
4. ✅ **Train for 50K-100K steps** with teacher forcing decay 1.0 → 0.3
5. ✅ **Validate AR generation quality** on held-out set

### Success Metrics

After retraining, you should see:
- **Valid JSON**: >95% (currently 0%)
- **Meaningful skills**: >80% (currently 0% - just "cococo")
- **EOS detection**: >85% via Fast-Loop

### Timeline

- **Week 1**: Implement scheduled sampling + curriculum
- **Week 2**: Run fine-tuning (24-48 hours)
- **Week 3**: Evaluate, iterate if needed

### Expected Outcome

After successful retraining:
```
INFO:openpi:[Slow-Loop] Generated: '{"skill":"move to","obj":"radio"}' (has_eos=False)
INFO:openpi:[Slow-Loop] Generated: '{"skill":"pick up","obj":"radio"} <EOS_SKILL>' (has_eos=True)
```

Instead of:
```
INFO:openpi:[Slow-Loop] Generated: cocococococo... (has_eos=False)
```

---

**Status**: Retraining strategy documented

**Next**: Implement scheduled sampling and start fine-tuning!

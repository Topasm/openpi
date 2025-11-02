# 🎯 Simple Retraining Strategy: Scheduled Sampling Only

## Core Insight

**The Hybrid Dual Loop architecture doesn't need separate training objectives!**

### Why One Loss is Enough

```python
# Training (AR with teacher forcing):
# L_skill_AR: Next-token prediction (including <EOS_SKILL>)
skill_loss = cross_entropy(predicted_tokens, ground_truth_skill_tokens)

# Inference - Slow-Loop (AR generation):
def generate_skill_autoregressive(...):
    for step in range(max_length):
        logits = project_to_vocab(hidden_state)
        next_token = argmax(logits)
        if next_token == EOS_TOKEN:
            break
    # Uses AR-trained head

# Inference - Fast-Loop (Non-AR classification):
def execute_fast_loop(...):
    logits = project_to_vocab(hidden_state)
    eos_probability = sigmoid(logits[EOS_TOKEN_ID])
    # Uses same AR-trained head, just reads EOS logit!
```

**Key point**: Both Slow-Loop and Fast-Loop use the **same Skill Head**, trained with the **same AR loss**. The only difference is:
- Slow-Loop: Generates tokens autoregressively (loop)
- Fast-Loop: Reads EOS probability (single pass)

### The Real Problem

Current training uses **100% teacher forcing**:
```python
# Always use ground truth tokens
prev_token = ground_truth[t]  # ← Always GT!
```

Inference uses **model predictions**:
```python
# Use model's own predictions
prev_token = argmax(logits)  # ← Model output!
```

**Distribution shift** → Model fails when seeing its own outputs → "cococo" loops

---

## Solution: Scheduled Sampling ONLY

No curriculum, no auxiliary losses, no data augmentation. Just fix the exposure bias.

### Modification to `compute_loss`

**File**: `src/openpi/models/pi0_hierarchical.py`

**Current code** (lines 405-452):
```python
# Skill Loss (Language Modeling)
if skill_tokens is not None and skill_mask is not None:
    # Forward pass: prefix only (for skill prediction)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1

    (prefix_hidden, _), _ = self.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=prefix_positions,
        adarms_cond=[None, None]
    )

    # Project to vocabulary (TEACHER FORCING - always uses GT context)
    embedding_table = self.PaliGemma.llm.embedder['input_embedding']
    vocab_logits = jnp.dot(prefix_hidden, embedding_table.value.T)

    # Extract logits for skill positions
    skill_len = skill_tokens.shape[1]
    skill_logits = vocab_logits[:, -skill_len:, :]

    # Compute loss
    targets = skill_tokens[:, 1:]
    logits = skill_logits[:, :-1, :]
    mask = skill_mask[:, 1:]

    token_losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=targets
    )

    masked_loss = token_losses * mask
    skill_loss = jnp.sum(masked_loss) / jnp.maximum(jnp.sum(mask), 1.0)
```

**Problem**: The prefix includes vision + prompt + **full skill_tokens** (teacher forcing). The model predicts `skill_tokens[t+1]` given `skill_tokens[:t]`, but always with ground truth.

**New code with scheduled sampling**:

```python
# Skill Loss with Scheduled Sampling
if skill_tokens is not None and skill_mask is not None:
    batch_size = observation.state.shape[0]
    skill_len = skill_tokens.shape[1]

    # Compute teacher forcing ratio (decays over training)
    teacher_forcing_ratio = self.compute_tf_ratio(train_step)

    # 1. Initial forward pass (vision + prompt, NO skill tokens yet)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1

    (current_hidden, _), kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=prefix_positions,
        adarms_cond=[None, None]
    )

    # 2. Autoregressive loop with scheduled sampling
    embedding_table = self.PaliGemma.llm.embedder['input_embedding']
    all_logits = []
    sampled_tokens = []

    for t in range(skill_len - 1):
        # Project current hidden state to vocabulary
        last_hidden = current_hidden[:, -1, :]  # [B, hidden_dim]
        logits = jnp.dot(last_hidden, embedding_table.value.T)  # [B, vocab_size]
        all_logits.append(logits)

        # Scheduled sampling: choose input for next step
        use_teacher_forcing = jax.random.bernoulli(
            rng, teacher_forcing_ratio, shape=(batch_size,)
        )

        # Get next token
        predicted_token = jnp.argmax(logits, axis=-1)  # [B]
        next_token = jnp.where(
            use_teacher_forcing,
            skill_tokens[:, t],    # Use ground truth
            predicted_token         # Use prediction
        )
        sampled_tokens.append(next_token)

        # Embed and continue (if not last step)
        if t < skill_len - 2:
            next_token_expanded = next_token[:, None]
            next_embedding = self.PaliGemma.llm(next_token_expanded, method="embed")

            # Update KV cache (use fixed logic from generate_skill_autoregressive)
            actual_cache_len = kv_cache[0][0].shape[1]
            new_position = jnp.array([[actual_cache_len]], dtype=jnp.int32)

            # Create mask
            cache_mask = jnp.ones((batch_size, 1, actual_cache_len), dtype=jnp.bool_)
            new_token_mask = jnp.ones((batch_size, 1, 1), dtype=jnp.bool_)
            full_mask = jnp.concatenate([cache_mask, new_token_mask], axis=-1)

            # Forward pass
            (next_hidden, _), kv_cache = self.PaliGemma.llm(
                [next_embedding, None],
                mask=full_mask,
                positions=new_position,
                kv_cache=kv_cache,
                adarms_cond=[None, None]
            )

            # Concatenate hidden states
            current_hidden = jnp.concatenate([current_hidden, next_hidden], axis=1)

        rng, _ = jax.random.split(rng)

    # 3. Compute loss (same as before)
    stacked_logits = jnp.stack(all_logits, axis=1)  # [B, skill_len-1, vocab_size]
    targets = skill_tokens[:, 1:]
    mask = skill_mask[:, 1:]

    token_losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=stacked_logits,
        labels=targets
    )

    masked_loss = token_losses * mask
    skill_loss = jnp.sum(masked_loss) / jnp.maximum(jnp.sum(mask), 1.0)
else:
    skill_loss = jnp.array(0.0)
```

### Helper Method: Teacher Forcing Ratio

Add this method to `Pi0Hierarchical` class:

```python
def compute_tf_ratio(self, step: int) -> float:
    """Compute teacher forcing ratio with linear decay.

    Args:
        step: Current training step

    Returns:
        Teacher forcing ratio (1.0 to 0.3)
    """
    # Config (can be made configurable)
    initial_tf = 1.0
    final_tf = 0.3
    decay_steps = 50000

    # Linear decay
    progress = jnp.minimum(step / decay_steps, 1.0)
    tf_ratio = initial_tf - progress * (initial_tf - final_tf)

    return tf_ratio
```

---

## Configuration Changes

**File**: `src/openpi/models/pi0_hierarchical.py`

Add to `Pi0HierarchicalConfig`:

```python
@dataclass
class Pi0HierarchicalConfig(_model.ModelConfig):
    # ... existing fields ...

    # Scheduled Sampling Config
    use_scheduled_sampling: bool = True
    initial_teacher_forcing: float = 1.0
    final_teacher_forcing: float = 0.3
    tf_decay_steps: int = 50000
```

Update `compute_loss` signature:

```python
def compute_loss(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
    skill_tokens: Optional[at.Int[at.Array, "b skill_len"]] = None,
    skill_mask: Optional[at.Bool[at.Array, "b skill_len"]] = None,
    memory_tokens: Optional[at.Int[at.Array, "b memory_len"]] = None,
    memory_mask: Optional[at.Bool[at.Array, "b memory_len"]] = None,
    *,
    train: bool = False,
    train_step: int = 0,  # ← NEW: Pass current step for TF ratio
) -> tuple[at.Float[at.Array, ""], dict]:
```

---

## Training Script

**File**: `scripts/retrain_ar_simple.py`

```python
#!/usr/bin/env python3
"""Simple retraining with scheduled sampling only."""

from openpi.training import config, train
from openpi.models import pi0_hierarchical

def main():
    # Load config (minimal changes)
    cfg = config.LeRobotB1KHierarchicalDataConfig(
        # Base settings (unchanged)
        use_eos_token=True,
        use_dense_prediction=True,
        use_hierarchical_tokenizer=True,
        use_concise_format=True,

        # NEW: Enable scheduled sampling
        use_scheduled_sampling=True,
        initial_teacher_forcing=1.0,
        final_teacher_forcing=0.3,
        tf_decay_steps=50000,
    )

    # Load existing checkpoint
    params = train.load_checkpoint(
        "dataset/openpi/checkpoints/5000/params"
    )

    # Fine-tune (NOT train from scratch!)
    train.run_training(
        config=cfg,
        initial_params=params,
        output_dir="outputs/ar_simple",
        num_steps=50000,  # Just 50K steps!
        learning_rate=1e-5,  # Lower LR for fine-tuning
    )

if __name__ == "__main__":
    main()
```

---

## Why This is Enough

### 1. Dual-Loop Works with Single AR Head

**Slow-Loop** (AR generation):
```python
for t in range(max_length):
    logits = skill_head(hidden_state)  # ← AR-trained head
    next_token = argmax(logits)
```

**Fast-Loop** (EOS classification):
```python
logits = skill_head(hidden_state)  # ← Same AR-trained head!
eos_prob = sigmoid(logits[EOS_TOKEN_ID])  # Just read EOS logit
```

Both use the **same head**, trained with **same loss** (AR next-token prediction).

### 2. No Need for Separate Classifier Loss

The AR loss **already trains the model to predict `<EOS_SKILL>` tokens**:
- During training: `ground_truth = [..., <EOS_SKILL>]`
- Loss: `cross_entropy(logits, ground_truth)`
- Model learns: "When skill is complete, output <EOS_SKILL>"

Fast-Loop just **reads this probability** without AR sampling!

### 3. Scheduled Sampling Fixes Distribution Shift

**Problem**: Model sees GT during training, own outputs during inference.

**Solution**: Mix GT and predictions during training:
- Step 0: 100% GT (like original training)
- Step 50K: 30% GT, 70% predictions
- Model learns to recover from mistakes

---

## Expected Results

### Training Metrics

Monitor these during training:

```python
# Metrics to log
metrics = {
    "skill_loss": skill_loss,
    "teacher_forcing_ratio": tf_ratio,
    "action_loss": action_loss,
    "total_loss": total_loss,
}
```

**Expected progression**:
- Steps 0-10K: `skill_loss` may increase slightly (model adjusts to seeing predictions)
- Steps 10K-30K: `skill_loss` stabilizes
- Steps 30K-50K: `skill_loss` converges (similar to original training)

### Inference Quality

After 50K steps:

**Before**:
```
INFO:openpi:[Slow-Loop] Generated: cocococococo...
```

**After**:
```
INFO:openpi:[Slow-Loop] Generated: '{"skill":"move to","obj":"radio"}'
INFO:openpi:[Slow-Loop] Generated: '{"skill":"pick up","obj":"radio"} <EOS_SKILL>'
```

**Success criteria**:
- ✅ Valid JSON: >90%
- ✅ Correct skill name: >70%
- ✅ Correct object: >70%
- ✅ No "cococo" loops: >95%

---

## Implementation Checklist

- [ ] Add `train_step` parameter to `compute_loss` signature
- [ ] Add `compute_tf_ratio()` method to `Pi0Hierarchical`
- [ ] Replace teacher-forcing skill loss with scheduled sampling loop
- [ ] Update config with scheduled sampling flags
- [ ] Create training script `scripts/retrain_ar_simple.py`
- [ ] Run fine-tuning for 50K steps
- [ ] Evaluate AR generation quality
- [ ] Test Hybrid Dual Loop end-to-end

---

## Timeline

- **Day 1**: Implement code changes (~4 hours)
- **Day 2-3**: Fine-tune model (50K steps, ~12-24 hours on A100)
- **Day 4**: Evaluate and iterate if needed

**Total**: ~3-4 days including implementation and training

---

## Fallback Plan

If scheduled sampling alone doesn't work (e.g., still generates "cococo"):

### Option A: Increase Sampling Ratio

Try more aggressive scheduled sampling:
```python
final_teacher_forcing: float = 0.1  # Down from 0.3
```

### Option B: Add Entropy Regularization

Penalize overly confident (peaky) distributions:
```python
# Entropy penalty
entropy = -jnp.sum(softmax(logits) * log_softmax(logits), axis=-1)
loss = cross_entropy_loss - 0.01 * entropy  # Encourage diversity
```

### Option C: Use the "Placeholder" Solution

If retraining fails, fall back to the quick fix:
```python
def generate_skill_autoregressive(...):
    # Skip AR generation, use placeholder
    skill_text = '{"skill":"navigate","obj":"target"}'
    plan_embedding = extract_from_vision(observation)
    has_eos = False
    return skill_text, plan_embedding, has_eos
```

The Fast-Loop EOS detection will still work, and plan-conditioned actions will still work!

---

## Summary

### What We're Changing
- ✅ Add scheduled sampling to skill loss (only change!)
- ❌ No curriculum learning
- ❌ No auxiliary losses
- ❌ No data augmentation

### Why It's Enough
1. **Single AR head serves both Slow-Loop and Fast-Loop**
2. **AR loss trains EOS prediction** (Fast-Loop uses it)
3. **Scheduled sampling fixes exposure bias** (Slow-Loop works)

### Cost
- **Time**: 3-4 days (including implementation)
- **Compute**: ~12-24 hours on A100 (50K steps)
- **Risk**: Low (fine-tuning from good checkpoint)

---

**Status**: Simplified strategy ready

**Next**: Implement scheduled sampling in `compute_loss`!

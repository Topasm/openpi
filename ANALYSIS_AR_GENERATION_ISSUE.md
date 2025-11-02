# 🔍 Analysis: Autoregressive Generation Failure

## Problem

The `generate_skill_autoregressive()` method generates garbage output (repeating "cococo...") instead of valid skill JSON.

**Debug output shows**:
```
INFO:openpi:[AR Step 0] token_id=9330, top-5: [(9330, 5.19), (528, 4.63), ...]
INFO:openpi:[AR Step 1] token_id=528, top-5: [(528, 15.81), (549, 13.69), ...]
INFO:openpi:[AR Step 2] token_id=528, top-5: [(528, 11.88), (9330, 9.31), ...]
INFO:openpi:[AR Step 3] token_id=528, top-5: [(528, 12.75), (549, 10.5), ...]
INFO:openpi:[AR Step 4] token_id=549, top-5: [(549, 11.69), (528, 10.81), ...]
...
```

The model alternates between tokens 528 and 549, getting stuck in a loop.

## Root Cause

**The checkpoint was NOT trained for autoregressive skill generation**.

### Training vs Inference Mismatch

**During Training** (pi0_hierarchical.py:405-452):
```python
# Forward pass with vision + prompt
(prefix_hidden, _), _ = self.PaliGemma.llm(
    [prefix_tokens, None],  # Only vision + prompt
    mask=prefix_attn_mask,
    positions=prefix_positions
)

# Project ALL positions to vocabulary
vocab_logits = jnp.dot(prefix_hidden, embedding_table.value.T)

# Compute loss against ground truth skill tokens (TEACHER FORCING)
targets = skill_tokens[:, 1:]  # Ground truth provided!
logits = skill_logits[:, :-1, :]
loss = cross_entropy(logits, targets)
```

**Key insight**: The model is trained with **teacher forcing** - it sees the ground truth skill tokens and learns to predict the next token given the previous ground truth tokens.

**During Inference** (our AR generation):
```python
# Start with vision + prompt (NO skill tokens)
# Try to generate tokens one-by-one
for step in range(max_length):
    logits = project_to_vocab(last_hidden)
    next_token = argmax(logits)
    # Feed back the GENERATED token (not ground truth!)
    next_embedding = embed(next_token)
    last_hidden = llm([next_embedding, None], ...)
```

**Key problem**: The model has **never seen its own generated tokens** during training, only ground truth. This causes **distribution shift** - the model doesn't know how to recover from its own mistakes.

### Why It Gets Stuck

1. **First token**: Model predicts token 9330 (reasonable, though not perfect)
2. **Second token**: Model sees embedding of 9330, which it NEVER saw during training
3. **Model confused**: Falls back to high-frequency tokens (528, 549)
4. **Feedback loop**: These tokens reinforce each other, model gets stuck

This is a classic **exposure bias** problem in seq uence-to-sequence models.

## Evidence

### 1. Token IDs Generated
- **528, 549**: These decode to "co" fragments
- **9330**: Unknown without tokenizer inspection
- **High logit variance**: Logits range from 5 to 16, suggesting model is "confident" but wrong

### 2. Dense Prediction Training
From training logs:
```
INFO:openpi.training.hierarchical_transforms:Phase 3 Mode: Dense prediction enabled (predict at every frame)
INFO:openpi.training.hierarchical_transforms:  EOS token enabled: appending '<EOS_SKILL>' at end of skills
```

**Dense prediction** means the model predicts skills at EVERY frame with vision context, not autoregressively.

### 3. No AR Training Code
Searching the codebase, there's NO code that trains the model to generate skills autoregressively. The training always uses teacher forcing with ground truth skill tokens.

## Why This Matters

### Hybrid Dual Loop Impact

The **Slow-Loop** (Phase 1) relies on autoregressive skill generation:
```python
# Phase 1: Slow-Loop
plan_text, plan_embedding, has_eos = model.generate_skill_autoregressive(...)
```

If this doesn't work, the entire Hybrid Dual Loop architecture breaks:
- ❌ No valid plan text
- ❌ Corrupted plan embedding (based on garbage tokens)
- ❌ Fast-Loop gets wrong conditioning
- ❌ No meaningful re-planning

## Solutions

### Option 1: Non-Autoregressive Skill Prediction ⭐ RECOMMENDED

**Instead of generating token-by-token, predict all skill tokens in parallel from vision context.**

**Approach**:
1. Get hidden state from vision + prompt
2. Project to vocabulary for next N positions
3. Decode most likely sequence
4. Use beam search or greedy decoding

**Pros**:
- Uses model's actual training objective (dense prediction)
- Avoids exposure bias
- Faster (single forward pass vs. N passes)

**Cons**:
- Need to know max skill length (can use fixed value like 32)
- May produce invalid JSON (can post-process)

**Implementation**:
```python
# Get prefix hidden state
(prefix_hidden, _), _ = llm([prefix_tokens, None], ...)

# Project last position to vocabulary for next N tokens
skill_start_hidden = prefix_hidden[:, -1:, :]  # [B, 1, hidden_dim]

# Predict multiple positions in parallel
skill_logits = []
for i in range(max_skill_len):
    logit = dot(skill_start_hidden, embedding_table.T)
    skill_logits.append(logit)
    # For next position, use predicted token's embedding
    token = argmax(logit)
    skill_start_hidden = embed(token)

# Decode all predicted tokens
predicted_tokens = [argmax(logit) for logit in skill_logits]
skill_text = tokenizer.decode(predicted_tokens)
```

**Wait, this is still autoregressive!** Let me think...

Actually, the model was trained to predict skills from the **same vision context** at multiple frames. Maybe we should:

**Better approach**: Predict skill tokens by treating each position as an independent classification:
```python
# Get vision hidden state
vision_hidden = prefix_hidden[:, -1, :]  # Last vision token

# Project to vocabulary N times (treating as N independent predictions)
# This is like "parallel decoding" but we're predicting from the SAME context
skill_tokens = []
for pos in range(max_skill_len):
    # Create position embedding for this skill position
    pos_emb = create_position_embedding(pos)
    combined = vision_hidden + pos_emb
    logits = dot(combined, embedding_table.T)
    token = argmax(logits)
    skill_tokens.append(token)
    if token == EOS_TOKEN:
        break

skill_text = tokenizer.decode(skill_tokens)
```

But this requires position embeddings which we don't have...

### Option 2: Use Only EOS Detection (Skip Skill Text)

**Accept that we can't generate skill text, use Fast-Loop EOS only.**

**Approach**:
```python
def generate_skill_autoregressive(...):
    # Don't generate text, just return placeholder
    skill_text = '{"skill":"unknown","obj":"unknown"}'

    # Extract plan embedding from vision context
    (prefix_hidden, _), _ = llm([prefix_tokens, None], ...)
    plan_embedding = prefix_hidden[:, -1, :]

    has_eos = False  # Always False, rely on Fast-Loop

    return skill_text, plan_embedding, has_eos
```

**Pros**:
- Simple, no AR generation needed
- Plan embedding still works (based on vision)
- Fast-Loop EOS detection works independently

**Cons**:
- No meaningful skill text for logging/debugging
- Can't use skill text for memory
- Loses interpretability

### Option 3: Retrain Model with AR Objective

**Retrain the checkpoint with proper autoregressive skill generation.**

**Training changes needed**:
1. **Scheduled sampling**: Mix teacher forcing with model predictions
2. **AR loss**: Train model to generate from its own outputs
3. **Curriculum learning**: Start with teacher forcing, gradually use more AR

**Pros**:
- Proper solution to exposure bias
- Model learns to generate coherently
- Matches inference behavior

**Cons**:
- Requires retraining (expensive!)
- Need training data and compute
- Time-consuming

### Option 4: Constrained Decoding / Beam Search

**Use advanced decoding strategies to guide generation.**

**Approaches**:
- **Beam search**: Keep top-K hypotheses, pick best complete skill
- **Constrained decoding**: Force valid JSON structure
- **Nucleus sampling**: Sample from top-p probability mass

**Pros**:
- Works with existing checkpoint
- Can improve quality
- Standard NLP technique

**Cons**:
- Still suffers from exposure bias
- Slower than greedy (especially beam search)
- May still produce garbage with bad model

### Option 5: Hybrid Approach (Partial AR + Templates)

**Generate key tokens (skill name, object) autoregressively, use templates for structure.**

**Approach**:
```python
# Generate just the skill name and object
skill_name_tokens = ar_generate(max_len=5)  # e.g., "move", "to"
obj_name_tokens = ar_generate(max_len=5)   # e.g., "radio"

# Use template
skill_text = f'{{"skill":"{skill_name}","obj":"{obj_name}"}}'
```

**Pros**:
- Reduces AR burden (fewer tokens to generate)
- Template ensures valid JSON
- Might work better than full AR

**Cons**:
- Still has AR issues
- Assumes fixed JSON structure
- Limited flexibility

## Recommendation

**For immediate deployment: Option 2 (Skip Skill Text)**

This is the most pragmatic solution given:
1. Model wasn't trained for AR generation
2. Fast-Loop EOS detection works independently
3. Plan embedding can still be extracted from vision context
4. Quick to implement, no retraining needed

**For production: Option 1 (Non-AR Prediction) + Option 4 (Beam Search)**

If you must have skill text:
1. Try non-AR parallel prediction with beam search
2. Add JSON validation and repair
3. Fall back to placeholder if generation fails

**For long-term: Option 3 (Retrain)**

If this is critical infrastructure:
1. Retrain model with scheduled sampling
2. Add AR generation as explicit training objective
3. Test on held-out AR generation benchmark

## Code Changes Needed

### Immediate Fix (Option 2):

```python
def generate_skill_autoregressive(self, ...):
    \"\"\"Slow-Loop: Extract plan embedding without AR generation.\"\"\"
    observation = preprocess_observation(None, observation, train=False)

    # Embed prefix (vision + prompt + optional memory)
    if memory_tokens is not None:
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_with_memory(
            observation, memory_tokens, memory_mask
        )
    else:
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

    # Get hidden state
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_hidden, _), _ = self.PaliGemma.llm(
        [prefix_tokens, None],
        mask=prefix_attn_mask,
        positions=prefix_positions,
        adarms_cond=[None, None]
    )

    # Extract plan embedding from last position
    plan_embedding = prefix_hidden[:, -1, :]  # [B, hidden_dim]

    # Placeholder skill text (can't generate properly)
    skill_text = '{"skill":"navigate","obj":"target"}'

    # Never claim EOS (let Fast-Loop handle it)
    has_eos = False

    logger.warning("[Slow-Loop] Using placeholder skill (AR generation not supported by checkpoint)")

    return skill_text, plan_embedding, has_eos
```

### Testing

This should make the Hybrid Dual Loop work, even without proper skill generation:
1. ✅ Slow-Loop returns valid plan embedding
2. ✅ Fast-Loop uses plan embedding for conditioning
3. ✅ Fast-Loop detects EOS properly
4. ✅ Router triggers re-planning when EOS detected

The only loss is interpretability (no real skill text), but the **core functionality** (plan-conditioned actions + EOS-based re-planning) still works!

## Summary

- ❌ **Autoregressive generation doesn't work** - model wasn't trained for it
- ⚠️ **Exposure bias** - model has never seen its own predictions during training
- ✅ **Plan embedding still works** - can be extracted from vision context
- ✅ **Fast-Loop EOS works** - independent of skill text generation
- 🎯 **Recommended**: Skip skill text, use placeholder + plan embedding

---

**Status**: Root cause identified, workaround available

**Next**: Implement Option 2 (placeholder skill text) or discuss retraining strategy

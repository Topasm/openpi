# Hybrid Dual Loop VLA - Visual Architecture Diagrams

## Complete System Architecture

```
╔════════════════════════════════════════════════════════════════════════╗
║                    HYBRID DUAL LOOP VLA SYSTEM                         ║
╚════════════════════════════════════════════════════════════════════════╝

┌────────────────────────────────────────────────────────────────────────┐
│  INITIALIZATION: Observe-then-Plan (Phase 3)                           │
│  ────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  Robot Environment                                                      │
│  ┌──────┐  ┌──────┐  ┌──────┐       ┌──────┐                         │
│  │Frame1│→ │Frame2│→ │ ...  │→ ... →│Frame10│                         │
│  └──────┘  └──────┘  └──────┘       └───┬───┘                         │
│                                          │                              │
│                                          ▼                              │
│                               ┌──────────────────────┐                 │
│                               │  Slow-Loop (Phase 1) │                 │
│                               │  Generate initial    │                 │
│                               │  plan with context   │                 │
│                               └──────────┬───────────┘                 │
│                                          │                              │
│                                          ▼                              │
│                              plan_text + plan_embedding                │
└────────────────────────────────────────────────────────────────────────┘
                                           │
                                           │ Store for main loop
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│  MAIN EXECUTION LOOP (Phase 4: Explicit Router)                        │
│  Runs until episode complete or max steps                              │
└────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
              ┌────────────────────────────────────────────┐
              │  Current Observation (Every Frame)         │
              │  ───────────────────────────────────────  │
              │  • Images: [base, left_wrist, right_wrist]│
              │  • State: [32-dim proprioception]         │
              │  • Task prompt tokens                     │
              └────────────────┬───────────────────────────┘
                               │
                               ▼
╔══════════════════════════════════════════════════════════════════════╗
║  FAST-LOOP (Phase 2) - Runs EVERY frame (~100ms)                     ║
╚══════════════════════════════════════════════════════════════════════╝
                               │
           ┌───────────────────┴────────────────────┐
           │   Embed Prefix with Plan Conditioning  │
           │   ────────────────────────────────────│
           │   Input Tokens:                        │
           │   [plan_embedding] +                   │
           │   [vision_tokens] +                    │
           │   [task_prompt_tokens]                 │
           └───────────────────┬────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Single Forward Pass│
                    │  (PaliGemma Expert) │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
      ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
      │ Vision      │  │ Skill Head  │  │ Action Head │
      │ Hidden      │  │ (Classifier)│  │ (ODE Flow)  │
      │ States      │  │             │  │             │
      └─────┬───────┘  └──────┬──────┘  └──────┬──────┘
            │                 │                │
            │                 ▼                ▼
            │     Project to Vocabulary   Flow Matching
            │     Extract EOS logit       10 ODE steps
            │                 │                │
            │                 ▼                ▼
            │      eos_probability      actions [50,32]
            │          (0.0-1.0)
            │                 │                │
            └─────────────────┴────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────┐
              │  Fast-Loop Output (Every Frame)   │
              │  ────────────────────────────────│
              │  • actions: [50, 32] ← Execute   │
              │  • eos_probability: 0.734        │
              │  • skill_logits: [vocab_size]    │
              └────────────┬──────────────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │  Execute Action on Robot   │
              │  action[0] → Robot Control │
              └────────────┬───────────────┘
                           │
                           ▼
              ┌────────────────────────────────────┐
              │  EXPLICIT ROUTER (Decision Point)  │
              │  ────────────────────────────────│
              │  if eos_probability > threshold:   │
              │      trigger = SLOW_LOOP           │
              │  else:                             │
              │      continue FAST_LOOP            │
              └────────┬───────────────────────────┘
                       │
         ┌─────────────┴──────────────┐
         │                            │
      No │                         Yes│ (eos_prob > 0.7)
         │                            │
         ▼                            ▼
    Continue Loop      ╔═════════════════════════════════════════════╗
         │             ║  SLOW-LOOP (Phase 1) - Intermittent (~1s)   ║
         │             ╚═════════════════════════════════════════════╝
         │                            │
         │                            ▼
         │             ┌──────────────────────────────────┐
         │             │  Update Memory                    │
         │             │  ───────────────────────────────│
         │             │  Append completed skill:         │
         │             │  memory += "<PAST_SKILL>         │
         │             │            {old_plan}            │
         │             │            </PAST_SKILL>"        │
         │             └──────────────┬───────────────────┘
         │                            │
         │                            ▼
         │             ┌──────────────────────────────────┐
         │             │  Autoregressive Generation       │
         │             │  ───────────────────────────────│
         │             │  Input:                          │
         │             │  • memory_tokens [1, mem_len]    │
         │             │  • vision_tokens [1, vis_len]    │
         │             │  • task_prompt [1, prompt_len]   │
         │             │                                  │
         │             │  Loop (max 64 tokens):           │
         │             │    1. Project hidden → vocab     │
         │             │    2. Sample next token          │
         │             │    3. Check if <EOS_SKILL>       │
         │             │    4. If EOS: break              │
         │             │    5. Else: embed & continue     │
         │             │       (with proper KV cache!)    │
         │             └──────────────┬───────────────────┘
         │                            │
         │                            ▼
         │             ┌──────────────────────────────────┐
         │             │  Slow-Loop Output                │
         │             │  ───────────────────────────────│
         │             │  • new_plan_text:                │
         │             │    '{"skill":"place in",         │
         │             │      "obj":"bin"}'               │
         │             │  • new_plan_embedding:           │
         │             │    [1, hidden_dim]               │
         │             │  • has_eos: True/False           │
         │             └──────────────┬───────────────────┘
         │                            │
         │                            ▼
         │             ┌──────────────────────────────────┐
         │             │  Update State                    │
         │             │  ───────────────────────────────│
         │             │  plan_text ← new_plan_text       │
         │             │  plan_embedding ← new_plan_emb   │
         │             └──────────────┬───────────────────┘
         │                            │
         └────────────────────────────┘
                      │
                      ▼
              [Continue Main Loop]
```

---

## Component Breakdown

### 🐇 Fast-Loop (Phase 2)

```
┌─────────────────────────────────────────────┐
│  FAST-LOOP: Real-time Execution             │
├─────────────────────────────────────────────┤
│  Frequency: Every frame                     │
│  Latency: ~100ms                            │
│  Purpose: Action generation + EOS detection │
└─────────────────────────────────────────────┘

Input:
┌─────────────────┐
│ observation     │ ← Current camera frames + state
│ plan_embedding  │ ← From Slow-Loop (goal)
└────────┬────────┘
         │
         ▼
Process:
┌────────────────────────────────┐
│ 1. Embed with plan injection:  │
│    tokens = [plan_emb] +       │
│             [vision] +         │
│             [prompt]           │
│                                │
│ 2. Single forward pass:        │
│    hidden = PaliGemma(tokens)  │
│                                │
│ 3. Dual predictions:           │
│    ┌─────────────────┐         │
│    │ Skill Head:     │         │
│    │ eos_prob =      │         │
│    │   sigmoid(      │         │
│    │     logits[EOS] │         │
│    │   )             │         │
│    └─────────────────┘         │
│    ┌─────────────────┐         │
│    │ Action Head:    │         │
│    │ for t in ODE:   │         │
│    │   x_t = x_t +   │         │
│    │         v_t*dt  │         │
│    │ actions = x_0   │         │
│    └─────────────────┘         │
└────────┬───────────────────────┘
         │
         ▼
Output:
┌─────────────────────────┐
│ actions: [50, 32]       │ ← Action chunk
│ eos_probability: 0.734  │ ← Skill completion signal
│ skill_logits: [257153]  │ ← For deviation detection
└─────────────────────────┘

Key Innovation: Non-AR EOS classification!
```

### 🐢 Slow-Loop (Phase 1)

```
┌─────────────────────────────────────────────┐
│  SLOW-LOOP: Autoregressive Planning         │
├─────────────────────────────────────────────┤
│  Frequency: Intermittent (~every 50-100 st) │
│  Latency: ~500-1000ms                       │
│  Purpose: Generate next skill plan          │
└─────────────────────────────────────────────┘

Input:
┌──────────────────┐
│ observation      │ ← Current state
│ memory_tokens    │ ← Past completed skills
└────────┬─────────┘
         │
         ▼
Process:
┌────────────────────────────────────────────┐
│ 1. Embed prefix:                           │
│    prefix = [memory] + [vision] + [prompt] │
│                                            │
│ 2. Fill KV cache:                          │
│    hidden, kv = LLM(prefix)                │
│                                            │
│ 3. Autoregressive loop:                    │
│    tokens = []                             │
│    for step in range(64):                  │
│      logits = dot(hidden, vocab_emb)       │
│      next_token = argmax(logits)           │
│                                            │
│      if next_token == EOS_SKILL_ID:        │
│        has_eos = True                      │
│        break                               │
│                                            │
│      # CRITICAL: Proper KV cache           │
│      cache_len = prefix_len + step         │
│      mask = [B, 1, cache_len + 1]  ✅      │
│      hidden, kv = LLM(                     │
│        embed(next_token),                  │
│        kv_cache=kv,                        │
│        mask=mask                           │
│      )                                     │
│      tokens.append(next_token)             │
│                                            │
│ 4. Decode:                                 │
│    plan_text = tokenizer.decode(tokens)    │
│                                            │
│ 5. Extract embedding:                      │
│    plan_emb = hidden[:, -1, :]             │
└────────┬───────────────────────────────────┘
         │
         ▼
Output:
┌─────────────────────────────────────┐
│ plan_text: '{"skill":"pick up",...}'│ ← For memory
│ plan_embedding: [1, hidden_dim]    │ ← For Fast-Loop
│ has_eos: True                       │ ← Completion flag
└─────────────────────────────────────┘

Key Fix: Proper KV cache mask construction!
```

### 👀 Observe-then-Plan (Phase 3)

```
┌─────────────────────────────────────┐
│  OBSERVE-THEN-PLAN: Cold Start Fix  │
├─────────────────────────────────────┤
│  Frequency: Once (at startup)       │
│  Latency: ~1-2 seconds              │
│  Purpose: Robust initial planning   │
└─────────────────────────────────────┘

Timeline:
t=0s ──────────────────────────► t=1.5s
  │                                 │
  │  Observation Collection:        │
  │  ┌──┐ ┌──┐     ┌───┐           │
  │  │F1│→│F2│→...→│F10│           │
  │  └──┘ └──┘     └─┬─┘           │
  │                   │             │
  │  Execute no-op    ▼             │
  │  actions      Use last obs      │
  │                   │             │
  │                   ▼             │
  │         ┌─────────────────┐    │
  │         │  Slow-Loop      │    │
  │         │  Generate plan  │    │
  │         │  with 10 frames │    │
  │         │  of context     │    │
  │         └────────┬────────┘    │
  │                  │              │
  └──────────────────┼──────────────┘
                     │
                     ▼
          Initial plan + embedding
                     │
                     ▼
          Enter Main Execution Loop

Why? Single-frame planning is unreliable!
10 frames ≈ 1 second of visual context
```

### 🔀 Explicit Router (Phase 4)

```
┌─────────────────────────────────┐
│  EXPLICIT ROUTER: Loop Control  │
├─────────────────────────────────┤
│  Runs: Every frame              │
│  Purpose: Coordinate Fast/Slow  │
└─────────────────────────────────┘

Decision Logic:
                 │
                 ▼
        ┌────────────────┐
        │ Fast-Loop runs │
        │ eos_prob = ?   │
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────────────┐
        │ if eos_prob > threshold│
        │    (default: 0.7)      │
        └────┬──────────────┬────┘
             │              │
      No     │              │ Yes
             │              │
             ▼              ▼
    ┌───────────┐   ┌─────────────┐
    │ Continue  │   │ Trigger     │
    │ Fast-Loop │   │ Slow-Loop   │
    └─────┬─────┘   └──────┬──────┘
          │                │
          │                ▼
          │         Update memory
          │         Re-plan
          │         Update plan_emb
          │                │
          └────────────────┘
                   │
                   ▼
            Next iteration

Tuning threshold:
• Higher (0.8): Re-plan less → Faster
• Lower (0.6): Re-plan more → Accurate
```

---

## Memory Management

```
┌────────────────────────────────────────────┐
│  MEMORY: Long-Horizon Skill Cache          │
└────────────────────────────────────────────┘

Format:
┌─────────────────────────────────────────────────┐
│ <PAST_SKILL>{"skill":"move to","obj":"A"}      │
│ </PAST_SKILL>                                   │
│ <PAST_SKILL>{"skill":"pick up","obj":"A"}      │
│ </PAST_SKILL>                                   │
│ <PAST_SKILL>{"skill":"move to","obj":"B"}      │
│ </PAST_SKILL>                                   │
│ ...                                             │
└─────────────────────────────────────────────────┘

Lifecycle:
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Skill    │     │ Tokenize │     │ Append   │
│ Complete │ ──► │ to max   │ ──► │ to       │
│ (EOS)    │     │ 256 tok  │     │ memory   │
└──────────┘     └──────────┘     └────┬─────┘
                                        │
                              ┌─────────▼────────┐
                              │ If len > 256:    │
                              │   Truncate FIFO  │
                              │   (keep recent)  │
                              └──────────────────┘

Capacity: ~12-15 skills (avg 20 tokens each)
```

---

## Training vs. Inference

```
┌────────────────────────────────────────────────────────────┐
│  TRAINING: Dense Prediction with EOS Supervision           │
└────────────────────────────────────────────────────────────┘

Episode: 1200 frames, 3 skills

Skill 0: Frames 0-399
├─ Frame 0:   Target = '{"skill":"move to",...}'
├─ Frame 1:   Target = '{"skill":"move to",...}'
├─ ...
├─ Frame 398: Target = '{"skill":"move to",...}'
└─ Frame 399: Target = '{"skill":"move to",...} <EOS_SKILL>' ← EOS!

Skill 1: Frames 400-899
├─ Frame 400: Target = '{"skill":"pick up",...}'
├─ ...
└─ Frame 899: Target = '{"skill":"pick up",...} <EOS_SKILL>' ← EOS!

Skill 2: Frames 900-1199
├─ ...
└─ Frame 1199: Target = '{"skill":"place in",...} <EOS_SKILL>' ← EOS!

Model learns:
• Visual patterns for each skill
• When to predict <EOS_SKILL> (completion visual cues)
• Skill-action alignment at every frame

┌────────────────────────────────────────────────────────────┐
│  INFERENCE: Dual Usage of Trained Model                    │
└────────────────────────────────────────────────────────────┘

Fast-Loop (Classifier Mode):
┌────────────────────────────┐
│ Input: Current vision      │
│ Output: Single prediction  │
│                            │
│ skill_logits = LLM(vision) │
│ eos_prob = sigmoid(        │
│   skill_logits[EOS_ID]     │
│ )                          │
│                            │
│ No loop! One pass! Fast!   │
└────────────────────────────┘

Slow-Loop (Generator Mode):
┌────────────────────────────┐
│ Input: Vision + memory     │
│ Output: Token sequence     │
│                            │
│ for step in range(64):     │
│   next_token = LLM(...)    │
│   if next_token == EOS:    │
│     break                  │
│                            │
│ Autoregressive! Slow!      │
└────────────────────────────┘

Same model, dual usage! 🎯
```

---

## Performance Timeline

```
Episode: 500 steps, 5 skills, ~60 seconds total

t=0s ─────────────────────────────────────────────► t=60s
│
├─ Observe-then-Plan (1.5s)
│  └─ Collect 10 frames + initial plan
│
├─ Fast-Loop × 100 (10s)
│  └─ Real-time action generation
│
├─ Slow-Loop (0.8s) ← EOS detected
│  └─ Re-plan with memory
│
├─ Fast-Loop × 100 (10s)
│
├─ Slow-Loop (0.8s) ← EOS detected
│
├─ Fast-Loop × 100 (10s)
│
├─ Slow-Loop (0.8s) ← EOS detected
│
├─ Fast-Loop × 100 (10s)
│
├─ Slow-Loop (0.8s) ← EOS detected
│
└─ Fast-Loop × 100 (10s)

Breakdown:
• Observe: 1.5s (2.5%)
• Fast: 50s (83%)    ← Majority!
• Slow: 4s (7%)
• Other: 4.5s (7.5%)
```

---

## File Structure

```
b1k-baselines/baselines/openpi/
│
├─ src/openpi/models/
│  └─ pi0_hierarchical.py
│     ├─ generate_skill_autoregressive() [737-872]  ← Phase 1
│     └─ execute_fast_loop()             [874-999]  ← Phase 2
│
├─ scripts/
│  └─ hybrid_dual_loop_inference.py
│     ├─ observe_then_plan()             [53-132]   ← Phase 3
│     └─ hybrid_dual_loop_inference()    [135-382]  ← Phase 4
│
├─ docs/
│  ├─ HYBRID_DUAL_LOOP_ARCHITECTURE.md   ← Full details
│  ├─ HYBRID_DUAL_LOOP_QUICKSTART.md     ← Quick start
│  └─ HYBRID_DUAL_LOOP_DIAGRAM.md        ← This file
│
├─ HYBRID_DUAL_LOOP_README.md            ← Main entry
└─ IMPLEMENTATION_SUMMARY.md             ← Summary
```

---

**Quick Reference**: See [HYBRID_DUAL_LOOP_README.md](../HYBRID_DUAL_LOOP_README.md)

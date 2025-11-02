# 🚀 Quick Start: Retrain for AR Generation

## TL;DR

Fix the "cococo..." autoregressive generation problem in 3 steps:

```bash
# 1. Start retraining (12-24 hours)
cd b1k-baselines/baselines/openpi
uv run scripts/retrain_ar_generation.py \
    --checkpoint_path=../../../dataset/openpi/checkpoints/5000 \
    --output_dir=./outputs/ar_retrain \
    --num_steps=50000

# 2. Test the results
uv run scripts/serve_b1k.py \
    --task_name=turning_on_radio \
    policy:checkpoint \
    --policy.config=pi0_b1k_hierarchical \
    --policy.dir=./outputs/ar_retrain/checkpoints/50000

# 3. See valid JSON instead of garbled output!
# Before: cocococococo...
# After: '{"skill":"move to","obj":"radio"}'
```

---

## What This Does

**Problem**: Model generates "cococo..." instead of valid skill JSON

**Root cause**: Trained with teacher forcing (only sees ground truth), fails when seeing its own predictions

**Solution**: Scheduled sampling - gradually expose model to its own predictions during training

---

## Step-by-Step Guide

### Prerequisites

- [x] CUDA-capable GPU (A100 recommended)
- [x] Existing checkpoint at `dataset/openpi/checkpoints/5000`
- [x] Python environment with dependencies installed

### Step 1: Start Retraining

```bash
cd b1k-baselines/baselines/openpi

# Basic command
uv run scripts/retrain_ar_generation.py \
    --checkpoint_path=../../../dataset/openpi/checkpoints/5000 \
    --output_dir=./outputs/ar_retrain \
    --num_steps=50000
```

**What happens**:
- Loads checkpoint 5000
- Fine-tunes with scheduled sampling
- Saves checkpoints every 5K steps to `./outputs/ar_retrain/checkpoints/`

**Time**: 12-24 hours on A100 80GB

### Step 2: Monitor Progress

The training will log:
```
INFO: teacher_forcing_ratio: 1.000 (step 0)
INFO: teacher_forcing_ratio: 0.650 (step 25000)
INFO: teacher_forcing_ratio: 0.300 (step 50000)
```

This shows the model gradually learning to use its own predictions.

### Step 3: Test Results

```bash
# Test with dummy environment
uv run scripts/serve_b1k.py \
    --task_name=turning_on_radio \
    policy:checkpoint \
    --policy.config=pi0_b1k_hierarchical \
    --policy.dir=./outputs/ar_retrain/checkpoints/50000
```

**Expected output**:
```
INFO:openpi:[Slow-Loop] Generated: '{"skill":"move to","obj":"radio"}' (has_eos=False)
```

✅ **Success!** Valid JSON instead of "cococo..."

---

## Advanced Options

### Resume Interrupted Training

```bash
uv run scripts/retrain_ar_generation.py \
    --output_dir=./outputs/ar_retrain \
    --resume=True \
    --num_steps=50000
```

### Train Longer for Better Results

```bash
uv run scripts/retrain_ar_generation.py \
    --checkpoint_path=../../../dataset/openpi/checkpoints/5000 \
    --output_dir=./outputs/ar_retrain \
    --num_steps=100000 \  # 2x longer
    --tf_decay_steps=100000
```

### More Conservative Sampling

```bash
uv run scripts/retrain_ar_generation.py \
    --checkpoint_path=../../../dataset/openpi/checkpoints/5000 \
    --output_dir=./outputs/ar_retrain \
    --num_steps=50000 \
    --final_teacher_forcing=0.5  # Keep more ground truth
```

---

## Troubleshooting

### "Out of memory" error

**Solution**: Reduce batch size in the training script

### Still generating garbled output

**Solutions**:
1. Train longer: `--num_steps=100000`
2. More conservative: `--final_teacher_forcing=0.4`
3. Check checkpoint loaded correctly

### Training too slow

**Expected**: ~12-24 hours on A100
- If much slower, check GPU utilization
- Consider using fewer validation steps: `--val_log_interval=5000`

---

## Files Created

After training, you'll have:
```
outputs/ar_retrain/
├── checkpoints/
│   ├── 5000/       # Intermediate checkpoint
│   ├── 10000/
│   ├── ...
│   └── 50000/      # Final checkpoint (use this!)
└── assets/         # Norm stats, tokenizer
```

---

## What Changed in the Code

All changes are backward compatible:

1. **Model config**: Added 4 scheduled sampling parameters (default: disabled)
2. **Training**: New `compute_loss()` mode when scheduled sampling enabled
3. **Scripts**: Two new scripts (retrain + eval)

**No changes to inference code!** The fixed `generate_skill_autoregressive()` method works the same.

---

## Next Steps After Retraining

### 1. Validate Quality

```bash
# Run evaluation metrics
uv run scripts/eval_ar_generation.py \
    --checkpoint_path=./outputs/ar_retrain/checkpoints/50000 \
    --num_samples=100
```

**Target metrics**:
- Valid JSON: >95%
- Garbled: <5%
- Skill/object match: >80%

### 2. Deploy to Robot

Use the retrained checkpoint with your Hybrid Dual Loop:

```bash
uv run scripts/hybrid_dual_loop_inference.py \
    --checkpoint_path=./outputs/ar_retrain/checkpoints/50000 \
    --task_prompt="Pick up the radio and turn it on" \
    --max_steps=500
```

### 3. Enable Long-Term Memory

Now that skill generation works, enable memory compression:

```python
# In your deployment script
memory_text = f"<L> {completed_skill_1} </L> <L> {completed_skill_2} </L>"
skill_json, plan_emb, has_eos = model.generate_skill_autoregressive(
    rng, obs,
    memory_tokens=tokenize(memory_text),
    tokenizer=tokenizer
)
```

---

## Detailed Documentation

For more information, see:

- **[IMPLEMENTATION_COMPLETE_SCHEDULED_SAMPLING.md](IMPLEMENTATION_COMPLETE_SCHEDULED_SAMPLING.md)** - Full implementation details
- **[RETRAINING_SIMPLE_SCHEDULED_SAMPLING.md](RETRAINING_SIMPLE_SCHEDULED_SAMPLING.md)** - Strategy explanation
- **[ANALYSIS_AR_GENERATION_ISSUE.md](ANALYSIS_AR_GENERATION_ISSUE.md)** - Root cause analysis

---

## Summary

✅ **All code is implemented and ready**

✅ **Just run the retraining script to fix AR generation**

✅ **Expected time: 3-4 days** (including 12-24h training)

✅ **Result: Valid JSON instead of "cococo..."**

**Start now**: `uv run scripts/retrain_ar_generation.py --checkpoint_path=...`

🎯 **Good luck with retraining!** 🎯

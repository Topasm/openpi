# BEHAVIOR-1K Training Guide

Quick reference for training models on the BEHAVIOR-1K challenge.

---

## Task Specialization (Dual-Head Model)

For **navigation + manipulation** task specialization, see:

📖 **[TASK_SPECIALIZATION_GUIDE.md](TASK_SPECIALIZATION_GUIDE.md)**

This comprehensive guide covers:
- Dual-head architecture (navigation + manipulation)
- Automatic movement labeling
- Hybrid router training
- Complete training instructions
- Troubleshooting

### Quick Start

```bash
cd /path/to/openpi

# 1. Compute normalization stats (one-time)
uv run scripts/compute_norm_stats.py --config-name pi0_b1k_dual_head

# 2. Start training
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_val.py pi0_b1k_dual_head \
  --exp_name="dual_head_$(date +%Y%m%d_%H%M%S)" \
  --batch_size=32 \
  --num_train_steps=50000
```

---

## Standard Pi0 Training

For **standard Pi0** training (no task specialization):

```bash
# 1. Compute normalization stats
uv run scripts/compute_norm_stats.py --config-name pi0_b1k

# 2. Start training
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_val.py pi0_b1k \
  --exp_name="pi0_$(date +%Y%m%d_%H%M%S)" \
  --batch_size=32 \
  --num_train_steps=50000
```

---

## Available Configs

| Config Name | Description | Use Case |
|-------------|-------------|----------|
| `pi0_b1k_dual_head` | Dual-head with task routing | **Recommended** for nav+manip |
| `pi0_b1k` | Standard Pi0 | Baseline comparison |
| `pi0_b1k_moe` | MoE in transformer | Advanced (high memory) |

---

## Documentation

- **Task Specialization**: [TASK_SPECIALIZATION_GUIDE.md](TASK_SPECIALIZATION_GUIDE.md) ⭐
- **General OpenPI**: [README.md](README.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Support

For issues or questions:
1. Check [TASK_SPECIALIZATION_GUIDE.md](TASK_SPECIALIZATION_GUIDE.md) Troubleshooting section
2. Review training logs for error messages
3. Verify dataset paths and normalization stats

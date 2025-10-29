# OpenPI Documentation

## Hierarchical VLA for BEHAVIOR-1K

📖 **Main Guide**: [hierarchical_vla_complete.md](hierarchical_vla_complete.md)

Complete documentation covering:
- All training phases (0, 1, 3)
- Architecture and implementation details
- Data formats and configuration
- Inference with EOS token detection
- Troubleshooting guide

## Other Docs

- [docker.md](docker.md) - Docker setup instructions
- [norm_stats.md](norm_stats.md) - Normalization statistics guide
- [remote_inference.md](remote_inference.md) - Remote inference setup

## Quick Start

```bash
# Phase 3 training (dense prediction with EOS)
# 1. Edit config.py: set all Phase 3 flags to True
# 2. Run training
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_hierarchical.py pi0_b1k_hierarchical \
  --exp-name="phase3_$(date +%Y%m%d_%H%M%S)"
```

See [hierarchical_vla_complete.md](hierarchical_vla_complete.md) for full details.

# Hierarchical VLA for BEHAVIOR-1K

Multi-task learning model that predicts both high-level skills and low-level actions.

## Quick Start

```bash
# Phase 0: Multi-task learning (skills + actions)
uv run python scripts/train_hierarchical.py pi0_b1k_hierarchical

# Phase 1: With long-horizon memory (edit config.py line 1105: enable_memory=True)
uv run python scripts/train_hierarchical.py pi0_b1k_hierarchical
```

## Architecture

### Phase 0: Multi-Task Learning

```
Vision Encoder (SigLIP) → PaliGemma → ┬→ Skill Head (language model)
                                       └→ Action Head (flow matching)
```

**Loss**: `L_total = 10.0 * L_skill + 1.0 * L_action`

**Components**:
- Skill prediction: Cross-entropy loss on skill text (first 10 frames of each skill)
- Action prediction: Flow matching loss (all frames)

### Phase 1: Long-Horizon Memory

Prepends compressed past skills as memory tokens:

```
Memory: "<L> skill0 </L> <L> skill1 </L>" → Embed → Concatenate with current observation
```

**Memory size**: 256 tokens max (~10-15 past skills)

## Implementation

### Model
- **File**: `src/openpi/models/pi0_hierarchical.py`
- **Config**: `Pi0HierarchicalConfig`
- **Key methods**:
  - `embed_prefix()` - Phase 0
  - `embed_prefix_with_memory()` - Phase 1
  - `compute_loss()` - Multi-task loss

### Data Pipeline
- **File**: `src/openpi/training/hierarchical_transforms.py`
- **Transforms**:
  - `AddSkillAnnotation` - Loads skill JSON
  - `TokenizeSkills` - Tokenizes skill text
  - `AddPastSkillsCache` - Creates memory (Phase 1)
  - `TokenizeMemory` - Tokenizes memory (Phase 1)

### Configuration
- **File**: `src/openpi/training/config.py`
- **Data config**: `LeRobotB1KHierarchicalDataConfig` (lines 502-588)
- **Train preset**: `pi0_b1k_hierarchical` (lines 1083-1121)

### Training Script
- **File**: `scripts/train_hierarchical.py`
- Handles skill + memory tokens
- Enhanced logging for hierarchical losses

## Data Format

### Skill Annotations
Location: `/home/seonghyeon/Postech-Behavior-Challenge/dataset/2025-challenge-demos/annotations`

Format: `task-XXXX/episode_XXXXXXXX.json`

```json
{
  "skill_annotation": [
    {
      "skill_idx": 0,
      "skill_description": ["move to"],
      "start_frame": 0,
      "end_frame": 264,
      "obj": ["radio_89"],
      "manip": null,
      "type": "uncoordinated"
    }
  ]
}
```

### Skill Text (tokenized)
```json
{"skill":"move to","obj":["radio_89"],"manip":null,"type":"uncoordinated"}
```

## Training

### Expected Results

| Metric | Initial | 10K steps | 50K steps |
|--------|---------|-----------|-----------|
| Skill Loss | ~10.0 | ~1.0-2.0 | ~0.5-1.0 |
| Action Loss | ~1.0 | ~0.05-0.1 | ~0.02-0.05 |
| Skill Accuracy | ~10% | >90% | >95% |

### Monitoring
- Project: `B1K_Hierarchical` in WandB
- Watch: `skill_loss`, `action_loss`, `total_loss`

### Timeline
- 10K steps: ~12-24 hours on 8×A100
- 50K steps: ~2-3 days on 8×A100

## Configuration

### Key Parameters

```python
# Model
skill_loss_weight = 10.0  # Weight for skill prediction
action_loss_weight = 1.0   # Weight for action prediction
max_skill_tokens = 64      # Max tokens per skill

# Data
skill_prediction_window = 10  # Frames to predict at skill start
enable_memory = False          # Phase 0: False, Phase 1: True
annotation_root = "/home/seonghyeon/Postech-Behavior-Challenge/dataset/2025-challenge-demos/annotations"
```

### Compute Norm Stats

Before training, compute normalization statistics:

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_b1k_hierarchical
```

## Troubleshooting

**"skill_tokens not in batch"**: Using wrong config - must use `pi0_b1k_hierarchical`

**"Annotation not found"**: Check `annotation_root` path in config

**Skill loss not decreasing**: Try increasing `skill_loss_weight` to 20.0

**OOM**: Reduce `batch_size` to 16 or 8

## Development Notes

### Skill Prediction Window
Only predict skills at first 10 frames of each skill (not every frame):
- Efficient: Only 2-5% of frames compute skill loss
- Sufficient: 10 frames provide strong training signal
- Semantic: Model learns to "plan" at start, "execute" during

### Phase 1 Memory Design
Compressed text instead of frames:
- Efficiency: 256 tokens vs full video frames
- Semantic: Captures "what was done" not "how it looked"
- Long-term: Remembers 10-15 skills vs 3-5 frames

## Note on MoE Files

The codebase contains MoE (Mixture of Experts) related files that are NOT used by the hierarchical VLA implementation. These can be ignored:
- `src/openpi/models/pi0_moe.py` - Not used
- `src/openpi/models/moe.py` - Not used
- `src/openpi/models/gemma_moe.py` - Not used
- Configs: `pi0_b1k_moe`, `pi0_b1k_dual_head` - Different approach

The hierarchical VLA uses skill prediction (language modeling), not expert routing.

## Related Files

### Core Implementation
- `src/openpi/models/pi0_hierarchical.py` - Model architecture
- `src/openpi/training/hierarchical_transforms.py` - Data transforms
- `src/openpi/training/skill_utils.py` - Skill utilities
- `src/openpi/training/config.py` - Configuration
- `scripts/train_hierarchical.py` - Training script

### Testing
- `scripts/test_hierarchical_pipeline.py` - Test script

## References

- Pi0: https://www.physicalintelligence.company/blog/pi0
- ProVideLLM: https://arxiv.org/abs/2401.06465
- BEHAVIOR-1K: https://behavior.stanford.edu/

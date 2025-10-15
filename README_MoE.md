# Movement Label Collection for Future MoE Implementation

This branch implements **online movement labeling** for the B1K dataset as preparation for future Mixture of Experts (MoE) implementation.

## What's Implemented

### 1. Online Movement Labeling ([src/openpi/transforms.py](src/openpi/transforms.py))

The `OnlineMovementLabeler` transform computes movement labels on-the-fly during training:

- **Classification**: Based on base velocity magnitude from proprioception
  - **Manipulation (label 0)**: Base not moving (`velocity <= 0.01`)
  - **Navigation (label 1)**: Base moving (`velocity > 0.01`)

- **Data Source**: Uses `base_qvel` [0:3] from the 23-dim compact state (after B1kInputs transform)

- **Standard Track Compatible**: Only uses base joint velocities, no global position information

- **Statistics Logging**: Logs label distribution every 1000 samples

### 2. MoE Configuration Support

#### Pi0MoEConfig ([src/openpi/models/pi0_moe.py](src/openpi/models/pi0_moe.py))
- Extends `Pi0Config` with MoE parameters
- Currently creates standard Pi0 models while collecting movement label data
- Configures 2 experts with supervised routing for future implementation

#### MoE Training Config ([src/openpi/training/config.py](src/openpi/training/config.py))
- `LeRobotB1KDataConfigMoE`: Data config with online movement labeling enabled
- `pi0_b1k_moe`: Training config using Pi0MoEConfig with MoE parameters

### 3. MoE Layer Infrastructure

#### MoE Core ([src/openpi/models/moe.py](src/openpi/models/moe.py))
Complete MoE layer implementation ready for future use:
- `MoEConfig`: Configuration for experts, routing, and auxiliary losses
- `Router`: Supports supervised, learned, and top-k routing
- `MoEFeedForward`: MoE FFN layer with expert routing and load balancing

#### Gemma MoE ([src/openpi/models/gemma_moe.py](src/openpi/models/gemma_moe.py))
Gemma transformer blocks with MoE FFN layers:
- `MoEBlock`: Transformer block with optional MoE in FFN
- `MoEModule`: Full Gemma model with MoE support
- Auxiliary loss aggregation across layers

### 4. Data Pipeline Updates

#### Data Loader ([src/openpi/training/data_loader.py](src/openpi/training/data_loader.py))
- Yields 3-tuple: `(observation, actions, batch_dict)`
- `batch_dict` contains `movement_label` for each sample

#### Training Loop ([scripts/train_val.py](scripts/train_val.py))
- Handles 3-tuple batch format
- Movement labels available but not yet used for routing
- Ready for future MoE loss integration

## Usage

### Training with Movement Label Collection

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_val.py pi0_b1k_moe \
  --exp_name="b1k_moe_$(date +%Y%m%d_%H%M%S)" \
  --batch_size=64 \
  --num_train_steps=50000
```

### Monitoring Movement Labels

Training logs show:
- First 10 samples: velocity values and labels
- Every 1000 samples: distribution statistics (% Manipulation vs Navigation)

Expected distribution: ~60% Manipulation, ~40% Navigation

## Configuration

### Movement Labeling Parameters

```python
LeRobotB1KDataConfigMoE(
    enable_movement_labels=True,
    velocity_threshold=0.01,  # L2 norm threshold for base velocity
)
```

### MoE Parameters

```python
Pi0MoEConfig(
    moe_config=MoEConfig(
        num_experts=2,
        router_type="supervised",
        load_balancing_loss_coef=0.01,
        router_z_loss_coef=0.001,
    ),
    moe_layers="all",
)
```

## Next Steps (TODO)

1. **Full MoE Model Implementation**
   - Complete `Pi0MoE` class with MoE-enabled Gemma
   - Integrate `gemma_moe.MoEModule` into Pi0
   - Implement `compute_loss_with_moe()` method

2. **MoE Training Integration**
   - Pass movement labels to MoE router
   - Aggregate and log MoE auxiliary losses
   - Monitor expert usage statistics

3. **MoE Evaluation**
   - Compare MoE vs baseline performance
   - Analyze expert specialization
   - Validate routing decisions

## File Structure

```
src/openpi/
├── models/
│   ├── moe.py              # MoE core implementation
│   ├── gemma_moe.py        # Gemma with MoE layers
│   └── pi0_moe.py          # Pi0 with MoE config
├── training/
│   ├── config.py           # MoE training configs
│   └── data_loader.py      # 3-tuple batch format
└── transforms.py           # OnlineMovementLabeler

scripts/
└── train_val.py            # Training with movement labels

.gitignore                  # Excludes temporary/draft files
```

## Notes

- Movement labels are computed efficiently during data loading (no preprocessing required)
- MoE infrastructure is complete but not yet activated in the model
- Standard Pi0 model is used for now while collecting movement label statistics
- All changes are compatible with standard track requirements (no global position data)

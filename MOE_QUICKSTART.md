# MoE Quick Start Guide

**Quick reference for implementing and using MoE in Pi0 for B1K tasks.**

See [MOE_IMPLEMENTATION_PLAN.md](MOE_IMPLEMENTATION_PLAN.md) for full details.

---

## Current Status: Phase 1 Complete ✅

### What Works Now
- ✅ Movement labels computed on-the-fly from base velocity
- ✅ Training pipeline yields `(observation, actions, batch_dict)` with `movement_label`
- ✅ MoE infrastructure ready (moe.py, gemma_moe.py)
- ✅ Pi0MoEConfig defined with MoE parameters

### What's Missing
- ❌ Pi0MoE model class (creates standard Pi0 for now)
- ❌ Training loop doesn't use movement_labels yet
- ❌ No MoE losses computed

---

## Training with Current Setup

```bash
# Standard training with movement label collection
cd /home/seonghyeon/Postech-Behavior-Challenge/b1k-baselines/baselines/openpi

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_val.py pi0_b1k_moe \
  --exp_name="b1k_$(date +%Y%m%d_%H%M%S)" \
  --batch_size=64 \
  --num_train_steps=50000
```

**What happens**:
- OnlineMovementLabeler computes movement labels every batch
- Labels added to batch_dict but not used by model yet
- Trains standard Pi0 model (not MoE)
- Logs movement label distribution every 1000 samples

**Check logs for**:
```
Sample 0: base_vel=0.000560, threshold=0.01
...
Online movement labels (after 1000 samples): Manipulation=600 (60.0%), Navigation=400 (40.0%)
```

---

## Next Steps: Implementing Pi0MoE

### Step 1: Complete Pi0MoE Class

**File**: [src/openpi/models/pi0_moe.py](src/openpi/models/pi0_moe.py:52)

**Uncomment and implement**:
```python
class Pi0MoE(pi0.Pi0):
    """Pi0 model with Mixture of Experts FFN layers."""

    def __init__(self, config: Pi0MoEConfig, *, rngs: nnx.Rngs):
        # TODO: Implement initialization
        # 1. Call super().__init__() with base config
        # 2. Replace self.PaliGemma.llm with MoE Gemma
        # 3. Use gemma_moe.convert_to_moe_config() to create MoE configs
        pass

    def compute_loss_with_moe(
        self, rng, observation, actions, movement_labels, *, train=False
    ) -> tuple[Array, dict]:
        # TODO: Implement MoE forward pass
        # 1. Encode vision/state (same as Pi0)
        # 2. Run MoE Gemma with movement_labels
        # 3. Decode actions
        # 4. Return loss + MoE aux dict
        pass
```

**Update create() method**:
```python
def create(self, rng: at.KeyArrayLike) -> Pi0MoE:
    return Pi0MoE(self, rngs=nnx.Rngs(rng))
```

### Step 2: Update Training Loop

**File**: [scripts/train_val.py](scripts/train_val.py:360)

**Replace loss_fn**:
```python
@at.typecheck
def loss_fn(
    model: _model.BaseModel,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
    movement_labels: jnp.ndarray | None,
):
    # Check if model has MoE
    if hasattr(model, "compute_loss_with_moe") and movement_labels is not None:
        chunked_loss, moe_aux = model.compute_loss_with_moe(
            rng, observation, actions, movement_labels, train=True
        )
        # Add auxiliary losses
        total_loss = jnp.mean(chunked_loss)
        if "load_balance_loss" in moe_aux:
            total_loss = total_loss + moe_aux["load_balance_loss"]
        if "router_z_loss" in moe_aux:
            total_loss = total_loss + moe_aux["router_z_loss"]
        return total_loss, moe_aux
    else:
        # Standard loss
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), {}
```

**Extract movement_labels**:
```python
# Already done, just uncomment/update:
if len(batch) == 3:
    observation, actions, batch_dict = batch
    movement_labels = batch_dict.get("movement_label", None)
else:
    observation, actions = batch
    movement_labels = None
```

**Update grad computation**:
```python
(loss, moe_aux), grads = nnx.value_and_grad(loss_fn, has_aux=True, argnums=diff_state)(
    model, train_rng, observation, actions, movement_labels
)
```

**Log MoE metrics**:
```python
info = {
    "loss": loss,
    "grad_norm": optax.global_norm(grads),
    "param_norm": optax.global_norm(kernel_params),
}

# Add MoE metrics
if moe_aux:
    if "load_balance_loss" in moe_aux:
        info["moe_load_balance_loss"] = moe_aux["load_balance_loss"]
    if "router_z_loss" in moe_aux:
        info["moe_router_z_loss"] = moe_aux["router_z_loss"]
    if "expert_usage" in moe_aux:
        for i, usage in enumerate(moe_aux["expert_usage"]):
            info[f"moe_expert_{i}_usage"] = usage
```

---

## Testing

### Unit Test: Movement Labeler
```python
from openpi.transforms import OnlineMovementLabeler
import numpy as np

labeler = OnlineMovementLabeler(velocity_threshold=0.01)

# Test navigation (high velocity)
sample_nav = {"state": np.random.randn(23)}
sample_nav["state"][0:3] = [0.05, 0.02, 0.01]  # High base velocity
result = labeler(sample_nav)
assert result["movement_label"] == 1

# Test manipulation (low velocity)
sample_manip = {"state": np.random.randn(23)}
sample_manip["state"][0:3] = [0.001, 0.0005, 0.0002]  # Low base velocity
result = labeler(sample_manip)
assert result["movement_label"] == 0
```

### Integration Test: Model Creation
```python
from openpi.training import config
import jax

# Get MoE config
cfg = config.get_config('pi0_b1k_moe')

# Create model
model = cfg.model.create(jax.random.key(42))

# Check type (will be Pi0 until implementation, then Pi0MoE)
print(f"Model type: {type(model).__name__}")
```

### Smoke Test: Training
```bash
# Quick 10-step training test
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_val.py pi0_b1k_moe \
  --num_train_steps=10 \
  --batch_size=4 \
  --overwrite \
  --exp_name="test_smoke"
```

---

## File Structure

```
src/openpi/
├── models/
│   ├── moe.py                 # Router, MoEFeedForward (complete)
│   ├── gemma_moe.py           # MoE Gemma blocks (complete)
│   └── pi0_moe.py             # Pi0MoEConfig + Pi0MoE class (TODO)
├── training/
│   ├── config.py              # pi0_b1k_moe config, LeRobotB1KDataConfigMoE
│   └── data_loader.py         # 3-tuple batch format
└── transforms.py              # OnlineMovementLabeler (complete)

scripts/
└── train_val.py               # Training loop (needs MoE integration)

docs/
├── README_MoE.md              # Current status
├── MOE_IMPLEMENTATION_PLAN.md # Full roadmap
└── MOE_QUICKSTART.md          # This file
```

---

## Configuration Reference

### Movement Labeling
```python
LeRobotB1KDataConfigMoE(
    enable_movement_labels=True,
    velocity_threshold=0.01,  # Tune if expert usage imbalanced
)
```

### MoE Parameters
```python
Pi0MoEConfig(
    moe_config=MoEConfig(
        num_experts=2,                      # Manipulation + Navigation
        router_type="supervised",           # Use movement labels
        load_balancing_loss_coef=0.01,      # Prevent expert collapse
        router_z_loss_coef=0.001,           # Encourage lower logits
    ),
    moe_layers="all",  # Apply MoE to all transformer layers
)
```

### Training
```python
TrainConfig(
    name="pi0_b1k_moe",
    batch_size=64,
    num_train_steps=50_000,
    # ... other params same as pi0_b1k
)
```

---

## Common Issues & Solutions

### Issue: 100% Manipulation or 100% Navigation
**Cause**: Velocity threshold too high/low
**Fix**: Adjust `velocity_threshold` in config
```python
velocity_threshold=0.005  # Lower = more navigation
velocity_threshold=0.02   # Higher = more manipulation
```

### Issue: Expert collapse (one expert unused)
**Cause**: Load balancing loss too weak
**Fix**: Increase `load_balancing_loss_coef`
```python
load_balancing_loss_coef=0.1  # Stronger penalty
```

### Issue: Training instability
**Cause**: Auxiliary losses too strong
**Fix**: Reduce coefficients
```python
load_balancing_loss_coef=0.001
router_z_loss_coef=0.0001
```

### Issue: No performance improvement
**Cause**: Experts not specializing
**Fix**:
1. Check expert usage is balanced (~60/40)
2. Try learned routing instead of supervised
3. Increase model capacity per expert
4. Analyze expert activations for specialization

---

## Key Metrics to Monitor

### During Training
- `moe_load_balance_loss`: Should decrease, stay low
- `moe_router_z_loss`: Should stay low
- `moe_expert_0_usage`: Should be ~60% (manipulation)
- `moe_expert_1_usage`: Should be ~40% (navigation)

### During Evaluation
- Success rate by task type (manipulation vs navigation)
- Action prediction MSE by movement type
- Expert routing accuracy (matches true movement label)

---

## References

- **Full Plan**: [MOE_IMPLEMENTATION_PLAN.md](MOE_IMPLEMENTATION_PLAN.md)
- **Current Status**: [README_MoE.md](README_MoE.md)
- **MoE Core**: [src/openpi/models/moe.py](src/openpi/models/moe.py)
- **Gemma MoE**: [src/openpi/models/gemma_moe.py](src/openpi/models/gemma_moe.py)
- **Pi0 MoE**: [src/openpi/models/pi0_moe.py](src/openpi/models/pi0_moe.py)

---

**Last Updated**: 2025-01-15
**Next Action**: Implement Pi0MoE class (Phase 2.1)

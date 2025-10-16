# MoE Implementation Plan

**Purpose**: Complete roadmap for implementing Mixture of Experts (MoE) in Pi0 model for B1K robot learning, with movement-based expert routing to improve performance on manipulation and navigation tasks.

**Status**: Phase 1 (Data Collection) Complete ✅

---

## Overview

### Goal
Implement a 2-expert MoE system that routes between:
- **Expert 0 (Manipulation)**: Specializes in arm-only manipulation tasks (base stationary)
- **Expert 1 (Navigation)**: Specializes in base movement and navigation tasks

### Hypothesis
By specializing different experts for manipulation vs navigation, the model can:
1. Learn better task-specific representations
2. Reduce interference between different movement types
3. Improve overall performance on both task types

### Key Design Decisions
- **2 experts**: One for manipulation, one for navigation (simple, interpretable)
- **Supervised routing**: Use velocity-based movement labels (no learned router initially)
- **Standard track compatible**: Only uses base velocity, no global position

---

## Phase 1: Data Collection ✅ COMPLETE

### Implemented
- [x] OnlineMovementLabeler transform ([src/openpi/transforms.py](src/openpi/transforms.py))
- [x] Velocity-based classification (threshold=0.01)
- [x] 3-tuple batch format with movement_label ([src/openpi/training/data_loader.py](src/openpi/training/data_loader.py))
- [x] MoE config infrastructure ([src/openpi/training/config.py](src/openpi/training/config.py))

### Results
- Movement label distribution: ~60% Manipulation, ~40% Navigation
- Labels computed on-the-fly during training (no preprocessing needed)
- Standard track compatible (uses only base_qvel)

---

## Phase 2: MoE Model Implementation 🚧 TODO

### 2.1 Complete Pi0MoE Class

**File**: [src/openpi/models/pi0_moe.py](src/openpi/models/pi0_moe.py)

**Tasks**:
1. **Implement Pi0MoE.__init__**
   ```python
   class Pi0MoE(pi0.Pi0):
       def __init__(self, config: Pi0MoEConfig, *, rngs: nnx.Rngs):
           # Initialize base Pi0 structure
           # Replace PaliGemma.llm with MoE-enabled Gemma
           # Keep vision encoder, state encoder, action decoder same
   ```
   - Get base Gemma configs (paligemma_variant, action_expert_variant)
   - Convert to MoE configs using `gemma_moe.convert_to_moe_config()`
   - Initialize MoE Gemma module
   - Handle LoRA parameters correctly

2. **Implement compute_loss_with_moe**
   ```python
   def compute_loss_with_moe(
       self, rng, observation, actions, movement_labels, *, train=False
   ) -> tuple[Array, dict]:
       # Encode vision and state (same as Pi0)
       # Run MoE Gemma with movement_labels
       # Decode actions
       # Return loss + MoE auxiliary outputs
   ```
   - Forward pass through MoE Gemma
   - Pass movement_labels to router
   - Collect MoE auxiliary losses (load balance, router z-loss)
   - Return per-sample loss + aux dict

3. **Update Pi0MoEConfig.create()**
   ```python
   def create(self, rng) -> Pi0MoE:
       return Pi0MoE(self, rngs=nnx.Rngs(rng))
   ```

**Dependencies**:
- [src/openpi/models/gemma_moe.py](src/openpi/models/gemma_moe.py) (already implemented)
- [src/openpi/models/moe.py](src/openpi/models/moe.py) (already implemented)

**Testing**:
```python
# Test model creation
config = config.get_config('pi0_b1k_moe')
model = config.model.create(jax.random.key(42))
assert isinstance(model, Pi0MoE)

# Test forward pass
loss, moe_aux = model.compute_loss_with_moe(
    rng, observation, actions, movement_labels, train=True
)
assert 'load_balance_loss' in moe_aux
assert 'router_z_loss' in moe_aux
```

### 2.2 Integrate MoE Gemma

**File**: [src/openpi/models/gemma_moe.py](src/openpi/models/gemma_moe.py)

**Tasks**:
1. **Verify MoEBlock implementation**
   - Ensure MoE FFN can replace standard FFN
   - Check movement_labels broadcasting to sequence length
   - Validate auxiliary loss aggregation

2. **Test MoEModule standalone**
   ```python
   # Create test MoE Gemma
   moe_config = convert_to_moe_config(
       base_gemma_config,
       moe_config=MoEConfig(num_experts=2, router_type="supervised"),
       moe_layers="all",
   )
   moe_module = MoEModule(configs=[moe_config], ...)

   # Test forward pass with movement labels
   outputs, moe_aux = moe_module(
       embedded=embeddings,
       movement_labels=labels,
   )
   ```

3. **Integration checklist**:
   - [ ] MoE FFN layers replace standard FFN in transformer blocks
   - [ ] Router receives movement_labels correctly
   - [ ] Supervised routing maps labels to experts (0→expert_0, 1→expert_1)
   - [ ] Auxiliary losses computed per layer and aggregated
   - [ ] Expert usage statistics tracked

---

## Phase 3: Training Integration 🚧 TODO

### 3.1 Update Training Loop

**File**: [scripts/train_val.py](scripts/train_val.py)

**Current state**: Movement labels extracted but not used

**Tasks**:
1. **Enable MoE loss computation**
   ```python
   def loss_fn(model, rng, observation, actions, movement_labels):
       if isinstance(model, pi0_moe.Pi0MoE):
           chunked_loss, moe_aux = model.compute_loss_with_moe(
               rng, observation, actions, movement_labels, train=True
           )
           # Add auxiliary losses
           total_loss = jnp.mean(chunked_loss)
           if "load_balance_loss" in moe_aux:
               total_loss += moe_aux["load_balance_loss"]
           if "router_z_loss" in moe_aux:
               total_loss += moe_aux["router_z_loss"]
           return total_loss, moe_aux
       else:
           # Standard Pi0
           chunked_loss = model.compute_loss(rng, observation, actions, train=True)
           return jnp.mean(chunked_loss), {}
   ```

2. **Log MoE metrics**
   ```python
   info = {
       "loss": loss,
       "grad_norm": optax.global_norm(grads),
       "param_norm": optax.global_norm(kernel_params),
   }

   # Add MoE metrics
   if moe_aux:
       info["moe_load_balance_loss"] = moe_aux["load_balance_loss"]
       info["moe_router_z_loss"] = moe_aux["router_z_loss"]
       if "expert_usage" in moe_aux:
           for i, usage in enumerate(moe_aux["expert_usage"]):
               info[f"moe_expert_{i}_usage"] = usage
       if "routing_probs" in moe_aux:
           info["moe_avg_routing_confidence"] = jnp.mean(jnp.max(moe_aux["routing_probs"], axis=-1))
   ```

3. **Update validation**
   - Handle MoE models in validation loop
   - Log separate metrics for manipulation vs navigation samples
   - Track expert usage distribution

### 3.2 Hyperparameter Tuning

**Config**: [src/openpi/training/config.py](src/openpi/training/config.py) - `pi0_b1k_moe`

**Parameters to tune**:
1. **Movement labeling**:
   - `velocity_threshold`: 0.01 (current), try [0.005, 0.01, 0.02]
   - May need adjustment based on expert usage balance

2. **MoE auxiliary losses**:
   - `load_balancing_loss_coef`: 0.01 (current), try [0.001, 0.01, 0.1]
   - `router_z_loss_coef`: 0.001 (current), try [0.0001, 0.001, 0.01]
   - Balance: prevent load imbalance without hurting task performance

3. **Expert configuration**:
   - `num_experts`: 2 (fixed for manipulation + navigation)
   - `moe_layers`: "all" (current), could try specific layers [6, 12, 18, ...]
   - Trade-off: more MoE layers = more specialization but higher complexity

4. **Training dynamics**:
   - May need to adjust learning rate (experts might need different rates)
   - May need warmup period for router (though supervised routing is deterministic)
   - Monitor for expert collapse (one expert unused)

---

## Phase 4: Evaluation & Analysis 🚧 TODO

### 4.1 Performance Metrics

**Compare**: Pi0 baseline vs Pi0MoE

**Metrics**:
1. **Overall performance**:
   - Success rate on validation set
   - Action prediction MSE
   - Training convergence speed

2. **Task-specific performance**:
   - Manipulation tasks (base stationary): MSE, success rate
   - Navigation tasks (base moving): MSE, success rate
   - Check if MoE improves performance on each type

3. **MoE-specific metrics**:
   - Expert usage balance (should be ~60/40 matching label distribution)
   - Routing confidence (higher = better specialization)
   - Load balancing loss over training
   - Expert parameter divergence (measure specialization)

### 4.2 Expert Analysis

**Goals**: Understand what each expert learned

**Analysis**:
1. **Routing analysis**:
   ```python
   # For each validation sample
   - True movement label (from velocity)
   - Router decision (which expert was used)
   - Routing confidence
   - Task outcome (success/fail)
   ```

2. **Activation analysis**:
   - Visualize expert hidden states (PCA/t-SNE)
   - Compare manipulation vs navigation representations
   - Check if experts have distinct activation patterns

3. **Parameter analysis**:
   - Compare expert FFN weights
   - Measure parameter divergence (L2 distance)
   - Identify specialized vs shared features

### 4.3 Ablation Studies

**Questions to answer**:
1. **Is supervised routing optimal?**
   - Compare: supervised vs learned vs top-k routing
   - Hypothesis: Supervised is good baseline, learned may improve

2. **Which layers need MoE?**
   - Compare: all layers vs bottom/middle/top layers only
   - Hypothesis: Middle-to-top layers benefit most (higher-level features)

3. **Does velocity threshold matter?**
   - Compare: different thresholds (0.005, 0.01, 0.02)
   - Check expert usage balance and performance

4. **Are 2 experts enough?**
   - Try 3 experts: manipulation + navigation + combined?
   - May be overkill for current task distribution

---

## Phase 5: Optimization & Deployment 🚧 TODO

### 5.1 Efficiency Optimization

**Goals**: Reduce computational overhead

**Tasks**:
1. **Expert capacity optimization**:
   - Adjust `expert_capacity_factor` (currently 1.25)
   - Lower = faster but may drop tokens

2. **MoE layer selection**:
   - If some layers don't benefit from MoE, use standard FFN
   - Reduces parameters and computation

3. **Inference optimization**:
   - For deterministic routing, can precompute expert assignments
   - Batch samples by expert to maximize throughput

### 5.2 Deployment Considerations

**File**: [src/openpi/policies/b1k_policy.py](src/openpi/policies/b1k_policy.py)

**Tasks**:
1. **Export MoE policy**:
   - Ensure movement labels computed during inference
   - Handle variable-length episodes
   - Test on real robot (if available)

2. **Checkpoint compatibility**:
   - Ensure MoE checkpoints can be loaded
   - Handle LoRA parameter merging
   - Document checkpoint format

3. **Documentation**:
   - Update README with MoE usage
   - Document hyperparameters and tuning guide
   - Provide example scripts

---

## Implementation Timeline

### Sprint 1 (Week 1-2): Core MoE Model
- [ ] Complete Pi0MoE class implementation
- [ ] Test model creation and forward pass
- [ ] Verify MoE Gemma integration
- [ ] Unit tests for all components

### Sprint 2 (Week 3): Training Integration
- [ ] Update training loop for MoE loss
- [ ] Add MoE metric logging
- [ ] Test training run (small scale)
- [ ] Debug any issues

### Sprint 3 (Week 4-5): Full Training & Tuning
- [ ] Train full MoE model (50k steps)
- [ ] Hyperparameter tuning experiments
- [ ] Compare against baseline Pi0
- [ ] Document results

### Sprint 4 (Week 6): Evaluation & Analysis
- [ ] Expert usage analysis
- [ ] Task-specific performance breakdown
- [ ] Ablation studies
- [ ] Write evaluation report

### Sprint 5 (Week 7): Optimization & Polish
- [ ] Efficiency optimizations
- [ ] Deployment preparation
- [ ] Documentation updates
- [ ] Code cleanup

---

## Testing Strategy

### Unit Tests
```python
# Test OnlineMovementLabeler
def test_movement_labeler():
    labeler = OnlineMovementLabeler(velocity_threshold=0.01)
    sample = {"state": np.array([0.02, 0.01, 0.0, ...])}  # base_qvel > threshold
    result = labeler(sample)
    assert result["movement_label"] == 1  # Navigation

# Test Pi0MoE forward pass
def test_pi0_moe_forward():
    model = create_test_pi0_moe()
    loss, moe_aux = model.compute_loss_with_moe(rng, obs, actions, labels)
    assert loss.shape == (batch_size,)
    assert "load_balance_loss" in moe_aux

# Test MoE router
def test_moe_router():
    router = Router(num_experts=2, router_type="supervised")
    movement_labels = jnp.array([0, 1, 0, 1])  # batch=4
    probs, aux = router(tokens, movement_labels)
    assert probs.shape == (4, seq_len, 2)
    assert jnp.allclose(probs[0, :, 0], 1.0)  # Expert 0 for label 0
```

### Integration Tests
```python
# Test end-to-end training step
def test_training_step():
    config = get_config("pi0_b1k_moe")
    state, batch = setup_test_training()
    new_state, info = train_step(config, rng, state, batch)
    assert "loss" in info
    assert "moe_load_balance_loss" in info

# Test data pipeline
def test_data_pipeline():
    data_loader = create_data_loader(config)
    batch = next(iter(data_loader))
    obs, actions, batch_dict = batch
    assert "movement_label" in batch_dict
```

### Smoke Tests
```bash
# Quick training test (10 steps)
python scripts/train_val.py pi0_b1k_moe \
  --num_train_steps=10 \
  --batch_size=4 \
  --overwrite

# Check logs for MoE metrics
# Should see: moe_load_balance_loss, moe_router_z_loss, moe_expert_*_usage
```

---

## Success Criteria

### Phase 2 (Model Implementation)
- [x] Pi0MoE model creates successfully
- [x] Forward pass works with movement_labels
- [x] MoE auxiliary losses computed correctly
- [x] Unit tests pass

### Phase 3 (Training)
- [x] Training runs without errors
- [x] MoE metrics logged to W&B
- [x] Expert usage balanced (~60/40)
- [x] Model converges (loss decreases)

### Phase 4 (Evaluation)
- [x] MoE matches or exceeds baseline performance
- [x] Experts show task-specific specialization
- [x] Interpretable routing decisions
- [x] No expert collapse

### Phase 5 (Deployment)
- [x] Inference latency acceptable
- [x] Policy exports and loads correctly
- [x] Documentation complete
- [x] Ready for real robot testing

---

## Risk Mitigation

### Risk 1: Expert Collapse
**Symptom**: One expert handles all samples
**Mitigation**:
- Tune load_balancing_loss_coef (increase)
- Check movement label distribution
- Try learned router instead of supervised

### Risk 2: No Performance Improvement
**Symptom**: MoE performs same as baseline
**Mitigation**:
- Analyze expert specialization (may not be learning distinct features)
- Try more layers with MoE
- Increase model capacity per expert
- Consider 3-expert setup

### Risk 3: Training Instability
**Symptom**: Loss spikes, NaNs, divergence
**Mitigation**:
- Lower learning rate
- Add gradient clipping
- Reduce auxiliary loss coefficients
- Use mixed precision carefully

### Risk 4: Increased Computation Cost
**Symptom**: Training 2x slower
**Mitigation**:
- Reduce expert capacity factor
- Use MoE in fewer layers
- Optimize implementation (JIT compilation)
- Accept trade-off if performance gains are significant

---

## References

### Papers
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
- [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538)
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)

### Code
- [src/openpi/models/moe.py](src/openpi/models/moe.py) - MoE core implementation
- [src/openpi/models/gemma_moe.py](src/openpi/models/gemma_moe.py) - MoE Gemma
- [src/openpi/models/pi0_moe.py](src/openpi/models/pi0_moe.py) - Pi0 with MoE
- [src/openpi/transforms.py](src/openpi/transforms.py) - OnlineMovementLabeler

### Docs
- [README_MoE.md](README_MoE.md) - Current implementation status
- [BEHAVIOR Challenge Rules](https://behavior.stanford.edu/challenge) - Standard track requirements

---

## Notes

### Design Choices Rationale

**Why 2 experts (manipulation + navigation)?**
- Matches natural task decomposition in B1K dataset
- Simpler than 3+ experts (easier to interpret)
- Distribution is ~60/40 (good balance, not too skewed)

**Why supervised routing?**
- Ground truth movement labels available from velocity
- Deterministic (easier to debug)
- No router training instability
- Can switch to learned routing later if needed

**Why velocity-based labels?**
- Standard track compatible (no global position)
- Fast to compute (no preprocessing)
- Physically meaningful (base moving vs stationary)
- Empirically validated (~60/40 split)

**Why MoE in all layers?**
- Maximum specialization potential
- Standard in MoE literature
- Can ablate later if needed

### Open Questions

1. **Should we use expert dropout?**
   - Pro: Prevents over-reliance on one expert
   - Con: Adds randomness, may hurt deterministic routing

2. **Should we share some layers between experts?**
   - Pro: Reduces parameters, may improve sample efficiency
   - Con: Less specialization

3. **Should we use learned routing eventually?**
   - Pro: May find better routing strategy than velocity-based
   - Con: More complex, less interpretable

4. **Should we condition on task prompt?**
   - Pro: Could route based on task type (not just movement)
   - Con: More complex, may not align with expert specialization

**Decision**: Start simple (supervised, all layers, no dropout), iterate based on results.

---

**Last Updated**: 2025-01-15
**Status**: Phase 1 complete, ready for Phase 2 implementation

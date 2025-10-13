# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **openpi** repository from Physical Intelligence, containing PyTorch implementations of vision-language-action (VLA) models for robotics:
- **π₀ (pi0)**: Flow-based VLA model
- **π₀-FAST (pi0-fast)**: Autoregressive VLA with FAST action tokenizer (not yet supported in PyTorch)
- **π₀.₅ (pi05)**: Upgraded π₀ with improved generalization via knowledge insulation

This fork uses **PyTorch exclusively**. **All JAX model implementations have been removed** (see [JAX_REMOVAL_SUMMARY.md](JAX_REMOVAL_SUMMARY.md) for details). Base model checkpoints are pre-trained on 10k+ hours of robot data and fine-tuned checkpoints are available for specific robot platforms (ALOHA, DROID, LIBERO, B1K).

## Development Environment

### Dependency Management
This project uses **uv** for Python dependency management (requires Python 3.11+):
```bash
# Initial setup
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Update dependencies
uv sync

# If dependency conflicts occur, remove venv and retry
rm -rf .venv && uv sync
```

### Pre-commit Hooks
Install pre-commit hooks for automatic linting:
```bash
pre-commit install
```

### Linting and Formatting
```bash
ruff check .        # Run linter
ruff format .       # Format code
```

## Common Commands

### Training

**Compute normalization statistics (required before training):**
```bash
uv run scripts/compute_norm_stats.py --config-name <config_name>
```

**Training with PyTorch:**
```bash
# Single GPU
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> [--resume]

# Multi-GPU (single node)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name> [--resume]

# Multi-Node training
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

**Key training flags:**
- `--resume`: Resume training from latest checkpoint
- `--save_interval <n>`: Save checkpoint every n steps (default: 1000)

### Inference

**Serve a policy (starts server on port 8000):**
```bash
# From checkpoint
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_dir>

# Pre-defined environment
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

**Test inference without robot:**
```bash
uv run examples/simple_client/test_b1k_client.py
```

### Testing
```bash
pytest                              # Run all tests
pytest <path/to/test_file.py>      # Run specific test file
```

## PyTorch Setup Requirements

**CRITICAL**: PyTorch requires patching the transformers library for proper model operation.

**1. Verify transformers version:**
```bash
uv pip show transformers  # Must be 4.53.2
```

**2. Apply required patches:**
```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

These patches provide:
- AdaRMSNorm support (required for π₀.₅)
- Correct activation precision control
- KV cache usage without updates

**WARNING**: With uv's default hardlink mode, this permanently affects transformers in the uv cache across all projects. To fully undo: `uv cache clean transformers`

**3. Download or convert model checkpoints:**

Most pre-trained checkpoints are provided in PyTorch format. If you have a JAX checkpoint that needs conversion:
```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir <jax_checkpoint_path> \
    --config_name <config_name> \
    --output_path <pytorch_checkpoint_path>
```

## Architecture Overview

### Model Architecture
The PyTorch implementation provides a hierarchical VLA architecture:
1. **Vision Encoder**: SigLIP-based image encoder (224x224 images)
2. **Language Model**: Gemma-based transformer (2B or 300M variants, with optional LoRA)
3. **Action Decoder**: Flow matching for continuous actions (π₀/π₀.₅ models)

**Note**: π₀-FAST (autoregressive FSQ tokenizer) is not yet supported in PyTorch.

### Code Organization

**Core Model Implementations:**
- `src/openpi/models_pytorch/`: PyTorch model implementations
  - `pi0_pytorch.py`: Main π₀/π₀.₅ model class
  - `gemma_pytorch.py`: Gemma-based language model
  - `preprocessing_pytorch.py`: Image preprocessing and tokenization
  - `transformers_replace/`: Modified transformers library files (AdaRMSNorm, precision control, KV cache)
- `src/openpi/models/`: Shared configuration and utilities (JAX implementations removed)
  - `pi0_config.py`: Model configuration (contains some JAX legacy code for compatibility)
  - `tokenizer.py`: Action tokenization (flow matching)
  - `model.py`: Base model config and shared types

**Policy Layer (Platform-Specific):**
- `src/openpi/policies/`: Robot-specific policy implementations
  - `policy.py`: Base `Policy` class wrapping models with transforms
  - `b1k_policy.py`, `droid_policy.py`, `aloha_policy.py`, `libero_policy.py`: Platform-specific input/output transforms
  - `policy_config.py`: Factory for creating trained policies from checkpoints

**Training Pipeline:**
- `src/openpi/training/`:
  - `config.py`: Training configurations, data configs, and transform factories
    - `TrainConfig`: Top-level training configuration
    - `DataConfigFactory`: Abstract factory for creating data configs (subclasses: `LeRobotAlohaDataConfig`, `LeRobotLiberoDataConfig`, `LeRobotB1KDataConfig`, `RLDSDroidDataConfig`)
    - `ModelTransformFactory`: Creates model-specific transforms (tokenization, image resizing)
  - `data_loader.py`: LeRobot and RLDS dataset loaders
  - `optimizer.py`, `checkpoints.py`: Training utilities
- `scripts/train_pytorch.py`: PyTorch training entry point (supports single/multi-GPU, multi-node)

**Data Transforms (Three-Stage Pipeline):**
- `src/openpi/transforms.py`: Transform primitives
  1. **Repack Transforms**: Map dataset keys to common format (applied only during training)
  2. **Data Transforms**: Robot-specific transformations (e.g., delta actions, coordinate conversions) (applied in training and inference)
  3. **Model Transforms**: Model-specific preprocessing (tokenization, normalization) (applied in training and inference)

**Client/Server Infrastructure:**
- `packages/openpi-client/`: Lightweight client package for robot code
  - `websocket_client_policy.py`: Remote policy client
  - `image_tools.py`: Image preprocessing utilities
  - `runtime/`: Agent-environment runtime framework
- `src/openpi/serving/websocket_policy_server.py`: Policy server implementation

**Examples:**
- `examples/`: Platform-specific examples and data conversion scripts
  - Each subdirectory contains environment-specific code and READMEs
  - `convert_*_data_to_lerobot.py`: Data conversion utilities

### Key Configuration Patterns

**All training configs are defined in `src/openpi/training/config.py`:**

1. **Model Config** (`Pi0Config` or `Pi0FASTConfig`):
   - `action_dim`, `action_horizon`: Action space dimensions
   - `max_token_len`: Maximum sequence length (prompt + state + actions)
   - `paligemma_variant`, `action_expert_variant`: Model size and LoRA settings
   - `pi05=True`: Enable π₀.₅ mode (discrete state input, AdaRMSNorm)

2. **Data Config** (via `DataConfigFactory` subclasses):
   - `repo_id`: HuggingFace/LeRobot dataset identifier
   - `behavior_dataset_root`: Local dataset path (for B1K)
   - `episodes_index`: Subset of episodes to use for training
   - `norm_stats`: Normalization statistics (auto-loaded from assets)
   - Platform-specific factories handle transform setup automatically

3. **Weight Loader**:
   - `CheckpointWeightLoader("gs://...")`: Load pre-trained weights
   - `NoOpWeightLoader()`: Random initialization

4. **Freeze Filter** (for LoRA):
   - `model.get_freeze_filter()`: Automatically configure frozen parameters based on LoRA variant

**Accessing configs programmatically:**
```python
from openpi.training import config
cfg = config.get_config("pi0_libero")  # Get by name
```

### Data Processing Flow

**Training data pipeline:**
```
Raw Dataset (LeRobot/RLDS)
  → data_loader.py: Load episodes with action_horizon lookahead
  → Repack Transform: Standardize key names (training only)
  → Data Transforms: Robot-specific (e.g., delta actions, coordinate conversion)
  → Normalization: Apply norm_stats (quantile or z-score)
  → Model Transforms: Tokenization, image resizing, padding
  → Model input: Observation + Actions
```

**Inference data pipeline:**
```
Raw Observation
  → Data Transforms: Robot-specific
  → Normalization: Apply norm_stats
  → Model Transforms: Tokenization, image resizing
  → Model.sample_actions(): Generate action chunk
  → Output Transforms: Denormalize, convert to robot format
  → Robot actions
```

### Normalization Statistics

**Computing norm stats:**
- Run `scripts/compute_norm_stats.py --config-name <config>` before training
- Saves to `<assets_base_dir>/<config_name>/<asset_id>/norm_stats.json`
- Contains: `q01`, `q50`, `q99`, `std` for each state/action dimension

**Reusing norm stats from pre-training:**
- Use `AssetsConfig(assets_dir=<base_model_assets>, asset_id=<robot_name>)` in your data config
- Beneficial when fine-tuning on robots that were part of pre-training mixture
- See `docs/norm_stats.md` for details

**Troubleshooting diverging loss:**
- Check `q01`, `q99`, `std` values in `norm_stats.json`
- Rarely-used dimensions may have very small values → huge normalized values
- Manually adjust norm stats as workaround

### Checkpoint Management

**Checkpoint structure:**
```
<checkpoint_base_dir>/<config_name>/<exp_name>/<step>/
  ├── params/         # Model parameters
  ├── ema_params/     # EMA weights (if enabled)
  └── assets/         # Normalization stats and other assets
```

**Loading checkpoints:**
- `policy_config.create_trained_policy(config, checkpoint_dir)`: Creates policy from checkpoint
- PyTorch checkpoints contain `.safetensors` or `.pt` files
- Automatically loads associated transforms and normalization stats from the assets directory

## Robot Platform Integration

**Supported platforms:**
- **ALOHA**: Dual-arm manipulation (towel folding, pen uncapping)
- **DROID**: Single-arm Franka (table-top manipulation)
- **LIBERO**: Simulated manipulation benchmark
- **B1K**: Behavior-1K challenge (custom platform)

**Creating a new robot policy:**
1. Define `YourRobotInputs(Transform)` in `src/openpi/policies/your_robot_policy.py`:
   - Convert environment observations to model format (images, state, masks)
2. Define `YourRobotOutputs(Transform)`:
   - Convert model actions back to robot commands
3. Create `LeRobotYourRobotDataConfig(DataConfigFactory)` in `src/openpi/training/config.py`:
   - Define repack transforms (dataset keys → common format)
   - Instantiate data transforms (delta actions, coordinate conversions)
4. Add to `_CONFIGS` list in `config.py`

**Key considerations:**
- Images must be 224x224 RGB uint8
- State/action dimensions must match `model.action_dim`
- Gripper actions typically remain absolute (not delta)
- Consider whether dataset uses absolute vs. delta actions

## Remote Inference Setup

**Server side:**
```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=<config> \
    --policy.dir=<checkpoint_dir>
```

**Client side (in robot code):**
```python
from openpi_client import websocket_client_policy, image_tools

client = websocket_client_policy.WebsocketClientPolicy(host="<server_ip>", port=8000)
observation = {
    "observation/image": image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, 224, 224)
    ),
    "observation/state": state,
    "prompt": task_instruction,
}
action_chunk = client.infer(observation)["actions"]
```

## Common Development Pitfalls

**Transformers patching:**
- ALWAYS apply transformers patches after installation/updates
- Without patches, models will fail with missing AdaRMSNorm or precision errors
- Patches must be reapplied if you run `uv sync` or reinstall transformers

**GPU Memory Issues:**
- PyTorch uses more memory than JAX for the same model
- Use gradient checkpointing if available in config
- For multi-GPU: distributed training splits memory load
- Consider disabling EMA (`ema_decay=None`) for LoRA training

**Import errors:**
- Ensure `uv sync` has completed successfully
- For RLDS datasets, install optional dependencies: `uv sync --extra rlds`

**Dataset loading:**
- RLDS data loader requires `num_workers=0` (handles multiprocessing internally)
- LeRobot data loader: increase `num_workers` for faster loading (default: 2)

**Action dimension mismatches:**
- Verify `model.action_dim` matches your robot's action space
- Check that delta action masks align with joint/gripper structure
- Ensure `action_horizon` matches desired chunk length

**Missing norm stats:**
- Always run `compute_norm_stats.py` before training on new datasets
- For fine-tuning, consider reusing base model's norm stats via `AssetsConfig`

**PyTorch precision:**
- Training defaults to full bfloat16 (lower memory, potentially higher loss)
- Use `pytorch_training_precision="float32"` in config if experiencing training instability
- Mixed precision training not yet supported

**torch.compile issues:**
- First inference step will be slow due to compilation
- Subsequent steps are much faster
- Disable with `torch.compiler.disable()` if encountering errors

#!/bin/bash
# Hierarchical VLA Phase 0 Training Script
# This script sets up the environment and runs Phase 0 training

set -e  # Exit on error

echo "=========================================="
echo "Hierarchical VLA Phase 0 Training"
echo "=========================================="

# Step 1: Find project root (3 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "✓ Project root: $PROJECT_ROOT"

# Step 2: Set DATASET_PATH
export DATASET_PATH="$PROJECT_ROOT/dataset"
echo "✓ DATASET_PATH: $DATASET_PATH"

# Step 3: Verify dataset exists
if [ ! -d "$DATASET_PATH/2025-challenge-demos/annotations" ]; then
    echo "❌ ERROR: Annotations directory not found!"
    echo "   Expected: $DATASET_PATH/2025-challenge-demos/annotations"
    exit 1
fi
echo "✓ Annotations directory exists"

# Step 4: Check if we're in the right directory
if [ ! -f "scripts/train_hierarchical.py" ]; then
    echo "❌ ERROR: Not in openpi directory!"
    echo "   Current: $(pwd)"
    echo "   Expected: $PROJECT_ROOT/b1k-baselines/baselines/openpi"
    exit 1
fi
echo "✓ In correct directory: $(pwd)"

# Step 5: Run training
echo ""
echo "Starting Phase 0 training..."
echo "Press Ctrl+C to cancel in the next 3 seconds..."
sleep 3

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_hierarchical.py pi0_b1k_hierarchical \
  --exp-name="phase0_$(date +%Y%m%d_%H%M%S)" \
  --overwrite \
  --batch_size=32 \
  --num_train_steps=50000 \
  --weight_loader.params_path=gs://openpi-assets/checkpoints/pi0_base/params

#!/bin/bash
#SBATCH --job-name=rf3-inference-test
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%j-inference.out
#SBATCH --error=slurm_logs/%j-inference.err

# RF3 Inference Functional Test on B300
# Runs structure prediction on test CIF files to validate the model works on Blackwell GPUs.
#
# Usage:
#   mkdir -p slurm_logs
#   sbatch scripts/inference_test.sbatch
#
# Prerequisites:
#   - Foundry venv installed at /fsx/ubuntu/venvs/foundry
#   - RF3 checkpoint downloaded: foundry install rf3 --checkpoint-dir /fsx/ubuntu/checkpoints

set -euo pipefail

# Environment
source /fsx/ubuntu/venvs/foundry/bin/activate
cd /fsx/ubuntu/projects/foundry
export FOUNDRY_CHECKPOINT_DIRS=/fsx/ubuntu/checkpoints

# Disable cuequivariance (LLVM backend does not support Blackwell SM 10.3 yet)
export DISABLE_CUEQUIVARIANCE=1

# Print environment info
echo "===== Environment ====="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "======================="
echo ""

# Test 1: Inference on 8vkf CIF file
echo "===== Test 1: 8vkf_from_file.cif ====="
OUT_DIR=/fsx/ubuntu/predictions/8vkf
mkdir -p "$OUT_DIR"
time rf3 fold \
    inputs=models/rf3/tests/data/8vkf_from_file.cif \
    out_dir="$OUT_DIR" \
    ckpt_path=rf3
echo "Output files:"
ls -lh "$OUT_DIR"/
echo ""

# Test 2: Inference on 5vht CIF file
echo "===== Test 2: 5vht_from_file.cif ====="
OUT_DIR=/fsx/ubuntu/predictions/5vht
mkdir -p "$OUT_DIR"
time rf3 fold \
    inputs=models/rf3/tests/data/5vht_from_file.cif \
    out_dir="$OUT_DIR" \
    ckpt_path=rf3
echo "Output files:"
ls -lh "$OUT_DIR"/
echo ""

# Summary
echo "===== Inference Test Complete ====="
echo "Predictions saved to /fsx/ubuntu/predictions/"
echo "==================================="

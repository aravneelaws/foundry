# RF3 on B300: Quickstart Guide

> Clean step-by-step reproduction guide. Just commands that work.
> For detailed rationale and debugging notes, see [B300_SETUP_NOTES.md](B300_SETUP_NOTES.md).

## Prerequisites

- SageMaker HyperPod cluster with p6-b300.48xlarge instances
- FSx for Lustre mounted at `/fsx`
- Slurm configured with GPU GRES (`gpu:b300:8` per node)

## 1. Install uv and Python 3.12

```bash
# Install uv on FSx (visible from all nodes)
curl -LsSf https://astral.sh/uv/install.sh | \
  CARGO_HOME=/fsx/ubuntu/.cargo UV_INSTALL_DIR=/fsx/ubuntu/.local/bin sh
export PATH=/fsx/ubuntu/.local/bin:$PATH

# Install Python 3.12 (DLAMI only has 3.10)
uv python install 3.12
```

## 2. Clone and Install Foundry

```bash
mkdir -p /fsx/ubuntu/projects
cd /fsx/ubuntu/projects
git clone https://github.com/aravneelaws/foundry.git

# Create venv on FSx
uv venv --python 3.12 /fsx/ubuntu/venvs/foundry

# Install foundry with RF3 dependencies
cd /fsx/ubuntu/projects/foundry
uv pip install --python /fsx/ubuntu/venvs/foundry/bin/python -e '.[rf3,dev]'

# IMPORTANT: Reinstall PyTorch with CUDA 12.8 (default install pulls cu126 which doesn't support B300)
uv pip install --python /fsx/ubuntu/venvs/foundry/bin/python \
  torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

## 3. Fix NVRTC for Blackwell

PyTorch bundles NVRTC 12.8 which does not support SM 10.3. Replace with CUDA 13.0's NVRTC:

```bash
source /fsx/ubuntu/venvs/foundry/bin/activate
NVRTC_DIR=$(python -c "import nvidia.cuda_nvrtc; import os; print(os.path.dirname(nvidia.cuda_nvrtc.__file__))")/lib
cp "$NVRTC_DIR/libnvrtc.so.12" "$NVRTC_DIR/libnvrtc.so.12.backup"
cp "$NVRTC_DIR/libnvrtc-builtins.so.12.8" "$NVRTC_DIR/libnvrtc-builtins.so.12.8.backup"
cp /usr/local/cuda-13.0/lib64/libnvrtc.so.13.0.88 "$NVRTC_DIR/libnvrtc.so.12"
cp /usr/local/cuda-13.0/lib64/libnvrtc-builtins.so.13.0.88 "$NVRTC_DIR/libnvrtc-builtins.so.12.8"
```

## 4. Verify Installation

```bash
srun --nodes=1 --ntasks=1 --gres=gpu:1 --partition=dev bash -c '
  source /fsx/ubuntu/venvs/foundry/bin/activate
  python -c "
import torch
print(\"PyTorch:\", torch.__version__)
print(\"CUDA available:\", torch.cuda.is_available())
print(\"GPU:\", torch.cuda.get_device_name(0))
print(\"Compute capability:\", torch.cuda.get_device_capability(0))
from rf3.model.RF3 import RF3
from rf3.trainers.rf3 import RF3Trainer
print(\"RF3 imports: OK\")
"'
```

Expected output:
```
PyTorch: 2.7.1+cu128
CUDA available: True
GPU: NVIDIA B300 SXM6 AC
Compute capability: (10, 3)
RF3 imports: OK
```

## 5. Download RF3 Checkpoint

```bash
srun --nodes=1 --ntasks=1 --gres=gpu:1 --partition=dev bash -c '
  source /fsx/ubuntu/venvs/foundry/bin/activate
  foundry install rf3 --checkpoint-dir /fsx/ubuntu/checkpoints
'
```

## 6. Run Inference (Functional Proof)

### Interactive (single test file)

```bash
srun --nodes=1 --ntasks=1 --gres=gpu:1 --partition=dev bash -c '
  source /fsx/ubuntu/venvs/foundry/bin/activate
  cd /fsx/ubuntu/projects/foundry
  export FOUNDRY_CHECKPOINT_DIRS=/fsx/ubuntu/checkpoints
  rf3 fold \
    inputs=models/rf3/tests/data/8vkf_from_file.cif \
    out_dir=/fsx/ubuntu/predictions/8vkf \
    ckpt_path=rf3
'
```

### Via Slurm job script

```bash
sbatch scripts/inference_test.sbatch
# Check output: cat slurm_logs/<job_id>.out
```

## 7. Run Training Benchmark

### Single-node smoke test (8 GPUs, 50 steps)

```bash
sbatch scripts/benchmark_rf3_single.sbatch
```

### Full 2-node benchmark (16 GPUs, 200 steps)

```bash
mkdir -p slurm_logs
sbatch scripts/benchmark_rf3.sbatch
```

## Key Paths

| Item | Path |
|------|------|
| uv binary | `/fsx/ubuntu/.local/bin/uv` |
| Python 3.12 | `/fsx/ubuntu/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12` |
| Virtual env | `/fsx/ubuntu/venvs/foundry/` |
| Project | `/fsx/ubuntu/projects/foundry/` |
| Checkpoints | `/fsx/ubuntu/checkpoints/` |
| Predictions | `/fsx/ubuntu/predictions/` |
| Slurm logs | `/fsx/ubuntu/projects/foundry/slurm_logs/` |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvidia-smi` fails on head node | GPU drivers only on compute nodes. Use `srun --gres=gpu:1` |
| PyTorch says CUDA not available | Reinstall cu128 wheel (step 2) |
| `nvrtc: error: invalid value for --gpu-architecture` | Replace bundled NVRTC with CUDA 13.0 version (step 3) |
| `LLVM ERROR: Cannot select: intrinsic` | Set `export DISABLE_CUEQUIVARIANCE=1` |
| `ModuleNotFoundError: cuequivariance_ops_cu12` | Module name is `cuequivariance_ops` (no `_cu12`). If import fails, reinstall `cuequivariance-ops-cu12` |
| Slurm job stuck in PENDING | Check `sinfo` -- nodes may be in use. Only 2 nodes available |

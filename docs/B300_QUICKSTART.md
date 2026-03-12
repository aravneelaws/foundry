# RF3 on B300: Quickstart Guide

> Clean step-by-step reproduction guide for running RosettaFold3 on NVIDIA B300 GPUs.
> For detailed rationale and debugging notes, see [B300_SETUP_NOTES.md](B300_SETUP_NOTES.md).

## Prerequisites

- SageMaker HyperPod cluster with p6-b300.48xlarge instances (or similar Blackwell GPU instances)
- FSx for Lustre mounted at `/fsx`
- Slurm configured with GPU GRES (`gpu:b300:8` per node)
- CUDA 13.0 available on compute nodes (typically at `/usr/local/cuda-13.0/`)

## 1. Install uv and Python 3.12

```bash
# Install uv on FSx (visible from all nodes)
curl -LsSf https://astral.sh/uv/install.sh | \
  CARGO_HOME=/fsx/ubuntu/.cargo UV_INSTALL_DIR=/fsx/ubuntu/.local/bin sh
export PATH=/fsx/ubuntu/.local/bin:$PATH

# Install Python 3.12 if not already available
uv python install 3.12
```

## 2. Clone and Install Foundry

```bash
mkdir -p /fsx/ubuntu/projects
cd /fsx/ubuntu/projects
git clone https://github.com/aravneelaws/foundry.git
cd foundry
git checkout production

# Create venv on FSx
uv venv --python 3.12 /fsx/ubuntu/venvs/foundry

# Install foundry with RF3 dependencies
uv pip install --python /fsx/ubuntu/venvs/foundry/bin/python -e '.[rf3,dev]'

# IMPORTANT: Install PyTorch with CUDA 13.0 (default install pulls cu126 which doesn't support Blackwell)
uv pip install --python /fsx/ubuntu/venvs/foundry/bin/python \
  torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130

# Install cuEquivariance cu13 (for Blackwell-optimized fused triangle kernels)
uv pip install --python /fsx/ubuntu/venvs/foundry/bin/python \
  cuequivariance-ops-cu13==0.9.0 cuequivariance-ops-torch-cu13==0.9.0
```

> **Why cu130?** Blackwell GPUs (SM 10.x) require NVRTC 13.0+ for runtime kernel compilation. PyTorch cu128 bundles NVRTC 12.8 which crashes on SM 10.3. PyTorch cu130 bundles NVRTC 13.0 natively, so no manual NVRTC swap is needed. cuEquivariance cu13 builds include Blackwell-optimized fused kernels for triangle attention (v0.8.0+).

## 3. Verify Installation

```bash
srun --nodes=1 --ntasks=1 --gres=gpu:1 --partition=dev bash -c '
  source /fsx/ubuntu/venvs/foundry/bin/activate
  python -c "
import torch
print(\"PyTorch:\", torch.__version__)
print(\"CUDA built with:\", torch.version.cuda)
print(\"CUDA available:\", torch.cuda.is_available())
print(\"GPU:\", torch.cuda.get_device_name(0))
print(\"Compute capability:\", torch.cuda.get_device_capability(0))
from foundry import SHOULD_USE_CUEQUIVARIANCE
print(\"cuEquivariance enabled:\", SHOULD_USE_CUEQUIVARIANCE)
from rf3.model.RF3 import RF3
from rf3.trainers.rf3 import RF3Trainer
print(\"RF3 imports: OK\")
"'
```

Expected output:
```
PyTorch: 2.9.1+cu130
CUDA built with: 13.0
CUDA available: True
GPU: NVIDIA B300 SXM6 AC
Compute capability: (10, 3)
cuEquivariance enabled: True
RF3 imports: OK
```

## 4. Download RF3 Checkpoint (for inference)

```bash
srun --nodes=1 --ntasks=1 --gres=gpu:1 --partition=dev bash -c '
  source /fsx/ubuntu/venvs/foundry/bin/activate
  foundry install rf3 --checkpoint-dir /fsx/ubuntu/checkpoints
'
```

## 5. Run Inference

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

Or via the Slurm job script (runs both test proteins):

```bash
mkdir -p slurm_logs
sbatch scripts/inference_test.sbatch
# Check output: tail -20 slurm_logs/<job_id>-inference.out
```

## 6. Run Training Benchmark

The training benchmark uses synthetic data (random tensors matching RF3 input shapes) to measure GPU throughput without external data dependencies.

### Single-node (8 GPUs, cuEquivariance enabled)

```bash
mkdir -p slurm_logs
sbatch scripts/b300_bench_cu130_1node.sbatch
```

This runs 4 epochs x 50 batches/GPU = 200 optimizer steps with:
- crop_size=384 tokens, n_atoms=3072, diffusion_batch=48, MSA=1024
- bf16-mixed precision, DDP across 8 GPUs
- cuEquivariance **enabled** (Blackwell-optimized fused triangle kernels)

### Multi-node (2 nodes x 8 GPUs = 16 GPUs)

```bash
sbatch scripts/b300_bench_cu130_2node.sbatch
```

### Benchmark without cuEquivariance (for comparison)

To run with cuEquivariance disabled (vanilla PyTorch triangle ops):

```bash
# Single-node
sbatch scripts/benchmark_rf3_single.sbatch

# Multi-node
sbatch scripts/benchmark_rf3.sbatch
```

### Checking results

```bash
# Check job status
squeue -u ubuntu

# View profiling summary (after job completes)
grep -A15 'PROFILING SUMMARY' slurm_logs/<job_id>-*.out

# View epoch timings
grep 'Epoch.*completed' slurm_logs/<job_id>-*.out

# Detailed per-step CSV is written to the Hydra output dir (path printed in job output)
```

## Important Notes

### Blackwell Requirements

These apply to any SM 10.x GPU (B300, B200, etc.):

1. **PyTorch cu130 required**: Default pip install pulls `cu126` which doesn't support Blackwell. PyTorch cu128 partially works but its bundled NVRTC 12.8 causes cuEquivariance to crash. **Use `torch==2.9.1+cu130`** for full Blackwell support including cuEquivariance.
2. **cuEquivariance cu13 required**: The cu12 builds of cuequivariance-ops crash on SM 10.x. Install `cuequivariance-ops-cu13` and `cuequivariance-ops-torch-cu13` for Blackwell-optimized fused triangle kernels.
3. **No NVRTC swap needed with cu130**: PyTorch cu130 bundles NVRTC 13.0 natively, which supports SM 10.x. The manual NVRTC library swap documented in earlier versions of this guide is no longer necessary.

### SLURM + Lightning Fabric Configuration

The benchmark scripts use this DDP launch pattern:
```
#SBATCH --ntasks-per-node=8    (one SLURM task per GPU)
#SBATCH --gres=gpu:8           (all 8 GPUs allocated to the job)
#SBATCH --cpus-per-task=24     (192 CPUs / 8 tasks)

srun python ... trainer.devices_per_node=8 trainer.num_nodes=<N>
```

Key constraints:
- `ntasks-per-node` **must** match `trainer.devices_per_node` (Lightning Fabric validation)
- Do **NOT** use `gpus-per-task=1` -- it restricts `CUDA_VISIBLE_DEVICES` per task, which breaks Lightning's GPU assignment via `SLURM_LOCALID`
- Use `gres=gpu:8` so all tasks see all GPUs

## Key Paths

| Item | Path |
|------|------|
| Virtual env | `/fsx/ubuntu/venvs/foundry/` |
| Project | `/fsx/ubuntu/projects/foundry/` |
| Checkpoints | `/fsx/ubuntu/checkpoints/` |
| Predictions | `/fsx/ubuntu/predictions/` |
| Slurm logs | `/fsx/ubuntu/projects/foundry/slurm_logs/` |
| Profiling CSVs | Hydra output dir (printed in job output) |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvidia-smi` fails on head node | GPU drivers only on compute nodes. Use `srun --gres=gpu:1` |
| PyTorch says CUDA not available | Reinstall cu130 wheel (step 2) |
| `LLVM ERROR: Cannot select: intrinsic` | You're using PyTorch cu128 instead of cu130. Reinstall with `--index-url https://download.pytorch.org/whl/cu130` |
| `'sm_103a' is not a recognized processor` | Same as above -- need PyTorch cu130 for SM 10.3 support |
| `SHOULD_USE_CUEQUIVARIANCE: False` | Install `cuequivariance-ops-cu13` and `cuequivariance-ops-torch-cu13` |
| `devices=8 but ntasks-per-node=1` | Use `#SBATCH --ntasks-per-node=8` with `trainer.devices_per_node=8` |
| `machine only has: [0]` | Remove `--gpus-per-task=1`, use `--gres=gpu:8` instead |
| Slurm job stuck in PENDING | Check `sinfo` -- nodes may be in use |

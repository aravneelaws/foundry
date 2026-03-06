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

# Install Python 3.12 (DLAMI only has 3.10)
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

# IMPORTANT: Reinstall PyTorch with CUDA 12.8 (default install pulls cu126 which doesn't support Blackwell)
uv pip install --python /fsx/ubuntu/venvs/foundry/bin/python \
  torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

## 3. Fix NVRTC for Blackwell

PyTorch bundles NVRTC 12.8 which does not support SM 10.x (Blackwell). Replace with CUDA 13.0's NVRTC:

```bash
source /fsx/ubuntu/venvs/foundry/bin/activate

# Run on a compute node (head node has no GPU driver)
srun --nodes=1 --ntasks=1 --gres=gpu:1 --partition=dev bash -c '
  source /fsx/ubuntu/venvs/foundry/bin/activate
  NVRTC_DIR=$(python -c "import nvidia.cuda_nvrtc; import os; print(os.path.dirname(nvidia.cuda_nvrtc.__file__))")/lib
  cp "$NVRTC_DIR/libnvrtc.so.12" "$NVRTC_DIR/libnvrtc.so.12.backup"
  cp "$NVRTC_DIR/libnvrtc-builtins.so.12.8" "$NVRTC_DIR/libnvrtc-builtins.so.12.8.backup"
  cp /usr/local/cuda-13.0/lib64/libnvrtc.so.13.0.88 "$NVRTC_DIR/libnvrtc.so.12"
  cp /usr/local/cuda-13.0/lib64/libnvrtc-builtins.so.13.0.88 "$NVRTC_DIR/libnvrtc-builtins.so.12.8"
  echo "NVRTC swap complete"
'
```

> **Note**: This only needs to be done once. The venv is on FSx so the fix is visible from all nodes.

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

## 5. Download RF3 Checkpoint (for inference)

```bash
srun --nodes=1 --ntasks=1 --gres=gpu:1 --partition=dev bash -c '
  source /fsx/ubuntu/venvs/foundry/bin/activate
  foundry install rf3 --checkpoint-dir /fsx/ubuntu/checkpoints
'
```

## 6. Run Inference

```bash
srun --nodes=1 --ntasks=1 --gres=gpu:1 --partition=dev bash -c '
  source /fsx/ubuntu/venvs/foundry/bin/activate
  cd /fsx/ubuntu/projects/foundry
  export DISABLE_CUEQUIVARIANCE=1
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

**Expected results:**
- `8vkf_from_file.cif` (Cytochrome P450, ~407 residues): ~1m 13s (cold start, includes model load)
- `5vht_from_file.cif` (Chorismate Mutase, ~184 residues): ~46s (warm start)

## 7. Run Training Benchmark

The training benchmark uses synthetic data (random tensors matching RF3 input shapes) to measure GPU throughput without external data dependencies.

### Single-node (8 GPUs)

```bash
mkdir -p slurm_logs
sbatch scripts/benchmark_rf3_single.sbatch
```

This runs 4 epochs x 50 batches/GPU = 200 optimizer steps with:
- crop_size=384 tokens, n_atoms=3072, diffusion_batch=48, MSA=1024
- bf16-mixed precision, DDP across 8 GPUs
- `DISABLE_CUEQUIVARIANCE=1`

**Expected results (8x B300 SXM6 AC):**
- Avg step time: ~9.5s
- Throughput: ~325 tokens/sec, ~0.85 samples/sec
- Peak GPU memory: ~23 GB per GPU
- Total run time: ~35 min (including warmup)

### Multi-node (2 nodes x 8 GPUs = 16 GPUs)

```bash
sbatch scripts/benchmark_rf3.sbatch
```

**Expected results (16x B300 SXM6 AC):**
- Avg step time: ~9.6s (nearly identical to single-node)
- Throughput: ~643 tokens/sec, ~1.68 samples/sec
- Scaling efficiency: ~99% (near-linear)
- Inter-node communication: EFA with GPU Direct RDMA
- Total run time: ~18 min (including warmup)

### Checking results

```bash
# Check job status
squeue -u ubuntu

# View profiling summary (after job completes)
grep -A15 'PROFILING SUMMARY' slurm_logs/<job_id>-bench-*.out

# View epoch timings
grep 'Epoch.*completed' slurm_logs/<job_id>-bench-*.out

# Detailed per-step CSV is written to the Hydra output dir (path printed in job output)
```

## Important Notes

### Blackwell-Specific Workarounds

These are required on any SM 10.x GPU (B300, B200, etc.) due to software toolchain gaps:

1. **PyTorch CUDA 12.8 wheels**: Default pip install pulls `cu126` which doesn't support Blackwell. Must use `cu128` or later.
2. **NVRTC library swap**: PyTorch bundles NVRTC 12.8 which can't compile kernels for SM 10.x. Replace with CUDA 13.0's NVRTC (step 3 above).
3. **cuEquivariance disabled**: The `cuequivariance_ops` LLVM backend doesn't support SM 10.x. Set `DISABLE_CUEQUIVARIANCE=1`. The model falls back to vanilla PyTorch for triangle attention/multiplication ops. This makes throughput numbers pessimistic compared to what's achievable once cuEquivariance adds Blackwell support.

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
| PyTorch says CUDA not available | Reinstall cu128 wheel (step 2) |
| `nvrtc: error: invalid value for --gpu-architecture` | Replace bundled NVRTC with CUDA 13.0 version (step 3) |
| `LLVM ERROR: Cannot select: intrinsic` | Set `export DISABLE_CUEQUIVARIANCE=1` |
| `devices=8 but ntasks-per-node=1` | Use `#SBATCH --ntasks-per-node=8` with `trainer.devices_per_node=8` |
| `machine only has: [0]` | Remove `--gpus-per-task=1`, use `--gres=gpu:8` instead |
| Slurm job stuck in PENDING | Check `sinfo` -- nodes may be in use. Only 2 nodes available |
| `ModuleNotFoundError: cuequivariance_ops_cu12` | Module name is `cuequivariance_ops` (no `_cu12`). If import fails, reinstall `cuequivariance-ops-cu12` |

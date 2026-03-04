# RF3 on SageMaker HyperPod p6-B300: Setup & Benchmark Notes

> Detailed writeup covering decisions, problems, debugging, and solutions.
> For a clean step-by-step reproduction guide, see [B300_QUICKSTART.md](B300_QUICKSTART.md).

## Project Goal

Test RosettaFold3 (RF3) training on NVIDIA B300 GPUs (SageMaker HyperPod p6-b300 instances) and compare training performance against H200 GPUs for TCO analysis.

## Cluster Configuration

| Component | Details |
|-----------|---------|
| **Cluster type** | SageMaker HyperPod Slurm |
| **Instance type** | p6-b300.48xlarge (2 nodes) |
| **GPUs per node** | 8x NVIDIA B300 SXM6 AC |
| **GPU memory** | 275 GB HBM3e per GPU (287 GB total reported by torch) |
| **GPU compute capability** | SM 10.3 (Blackwell) |
| **GPU TDP** | 1100W per GPU |
| **CPUs per node** | 192 vCPUs (2x48-core, HT enabled) |
| **System RAM** | 4 TB per node |
| **OS** | Ubuntu 22.04.5 LTS, kernel 6.8.0-1044-aws |
| **NVIDIA driver** | 580.126.09 |
| **CUDA toolkit** | 12.6, 12.8, 12.9, 13.0 (all available; default symlink `/usr/local/cuda`) |
| **NCCL** | 2.26.2 (bundled with PyTorch); OFI plugin at `/opt/amazon/ofi-nccl/lib/` |
| **EFA** | v2.17.3, with nvidia peermem (v1.2.2) |
| **Shared storage** | FSx for Lustre at `/fsx` (1.2 TB) |
| **Slurm** | v24.11.0 |
| **Slurm partitions** | `dev` (default, EXCLUSIVE), `ml-p6-b300-48xlarge` |
| **Node hostnames** | `ip-10-2-141-204`, `ip-10-2-210-184` |

---

## Phase 1: Environment Setup

### The Python Problem

The HyperPod DLAMI ships only Python 3.10.12. Foundry's `pyproject.toml` requires `python>=3.12,<3.13`. There's no system package manager route to get 3.12 on this AMI, and conda environments are fragile on shared FSx. We opted for **uv** -- it installs a standalone Python build to a user directory and works well on shared filesystems.

Key decision: install both `uv` and the Python build on `/fsx` (not `/home`) so both compute nodes see the same tools without any NFS home-dir issues.

### The PyTorch CUDA Problem

This was the most subtle issue. Foundry's `pyproject.toml` specifies `torch>=2.2.0,<3` with no CUDA variant pin. When you run `uv pip install -e '.[rf3,dev]'`, the resolver grabs `torch==2.7.1+cu126` from PyPI's default index (cu126 is the current default wheel). That's fine for H100/A100 but B300 requires SM 10.3 support which is only in `cu128+` wheels.

The fix is simple but easy to forget: **always reinstall the cu128 wheel after the main install**:
```bash
uv pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

We considered pinning in `pyproject.toml` but decided against it -- the upstream repo supports many GPU architectures and we don't want to break that.

### The nvidia-smi Problem

Running `nvidia-smi` on the head/controller node fails because HyperPod doesn't load GPU drivers there. This is by design (the controller runs Slurm management, not GPU workloads). Every GPU command must go through `srun`:
```bash
srun --nodes=1 --ntasks=1 --gres=gpu:1 --partition=dev bash -c 'nvidia-smi'
```

This applies to all verification scripts, not just nvidia-smi. Any Python script importing `torch.cuda` must run via `srun` on a compute node.

### The cuequivariance Naming Confusion

The pip package is called `cuequivariance-ops-cu12` but the Python module is `cuequivariance_ops` (no `_cu12` suffix). This initially looked like an installation bug but is the intended behavior. The `_cu12` suffix is purely a pip distribution name to distinguish CUDA version variants; the importable module is always `cuequivariance_ops`.

### What Went Smoothly

- **NVLink P2P**: All 8 B300 GPUs within a node can do direct P2P memory access via NVLink. No special configuration needed.
- **EFA networking**: The OFI-NCCL plugin is pre-installed at `/opt/amazon/ofi-nccl/lib/`. Cross-node NCCL communication should work out of the box with the right environment variables.

### The NVRTC JIT Problem (Blackwell-Specific)

This was the most significant Blackwell compatibility issue. PyTorch uses NVRTC (NVIDIA Runtime Compiler) to JIT-compile certain CUDA kernels at runtime. The NVRTC version bundled with PyTorch (via the `nvidia-cuda-nvrtc-cu12` pip package) ships NVRTC 12.8, which does not know about SM 10.3 (Blackwell). This causes any op using JIT compilation to fail:

```
nvrtc: error: invalid value for --gpu-architecture (-arch)
```

Affected ops we discovered: `torch.erfinv` (used in weight init), `torch.prod` (used in inference pipeline). Most PyTorch ops use pre-compiled kernels and work fine; only the JIT-compiled ops fail.

We tried several approaches:
1. **PyTorch nightly (2.12.0.dev+cu128)**: Same NVRTC, same failure
2. **`LD_PRELOAD` with CUDA 13.0 NVRTC**: PyTorch loads its bundled version first, doesn't pick up the system one
3. **Python monkey-patching** (`rf3/blackwell_compat.py`): Works for Python-dispatched ops like `erfinv`, but `prod` is dispatched entirely in C++ and can't be intercepted

**What worked**: Replacing the bundled NVRTC libraries with CUDA 13.0 versions:
```bash
NVRTC_DIR=$(python -c "import nvidia.cuda_nvrtc; import os; print(os.path.dirname(nvidia.cuda_nvrtc.__file__))")/lib
cp "$NVRTC_DIR/libnvrtc.so.12" "$NVRTC_DIR/libnvrtc.so.12.backup"
cp "$NVRTC_DIR/libnvrtc-builtins.so.12.8" "$NVRTC_DIR/libnvrtc-builtins.so.12.8.backup"
cp /usr/local/cuda-13.0/lib64/libnvrtc.so.13.0.88 "$NVRTC_DIR/libnvrtc.so.12"
cp /usr/local/cuda-13.0/lib64/libnvrtc-builtins.so.13.0.88 "$NVRTC_DIR/libnvrtc-builtins.so.12.8"
```

CUDA 13.0 is pre-installed on the HyperPod DLAMI at `/usr/local/cuda-13.0/`. Its NVRTC recognizes SM 10.3 and all JIT-compiled ops work correctly after the swap.

### The cuequivariance LLVM Problem

The `cuequivariance_ops` library (v0.9.0) uses an LLVM-based CUDA backend. This LLVM version does not know about SM 10.3, causing:
```
'sm_103a' is not a recognized processor for this target (ignoring processor)
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.i32
```

The fix is simple: set `DISABLE_CUEQUIVARIANCE=1`. The model falls back to vanilla PyTorch implementations for triangle attention and triangle multiplication. These are functionally identical but may be slightly slower. All our sbatch scripts set this variable.

### Software Versions (in venv)

| Package | Version |
|---------|---------|
| Python | 3.12.13 |
| PyTorch | 2.7.1+cu128 |
| Lightning | 2.6.1 |
| Hydra | 1.3.2 |
| NCCL | 2.26.2 |
| cuequivariance | 0.9.0 |
| cuequivariance-ops-cu12 | 0.9.0 |
| cuequivariance-torch | 0.9.0 |
| atomworks | >=2.1.1 |
| wandb | 0.25.0 |
| rc-foundry | 0.0.1.dev1144+g75986084a (editable) |

---

## Phase 2: Functional Proof & Benchmark

### Step 2.2: RF3 Inference on B300

The first functional proof is running RF3 inference on test CIF files already in the repo. This validates the full model forward pass on B300 hardware.

**Checkpoint download:**
```bash
srun --nodes=1 --ntasks=1 --gres=gpu:1 --partition=dev bash -c '
  source /fsx/ubuntu/venvs/foundry/bin/activate
  foundry install rf3 --checkpoint-dir /fsx/ubuntu/checkpoints
'
```

The checkpoint (`rf3_foundry_01_24_latest_remapped.ckpt`) is ~600MB and will be saved to `/fsx/ubuntu/checkpoints/`.

**Running inference:**
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

Or use the sbatch script:
```bash
sbatch scripts/inference_test.sbatch
```

**Results (B300 SXM6 AC, single GPU):**
- `8vkf_from_file.cif`: 1m13s total (includes model load + 200-step diffusion sampling)
- `5vht_from_file.cif`: 46s (model already cached in GPU memory)
- Output: CIF structure prediction files generated successfully

**Test files available** (in `models/rf3/tests/data/`):
- `8vkf_from_file.cif` -- CIF structure file
- `5vht_from_file.cif` -- CIF structure file
- `5vht_from_json.json` -- JSON input format
- `multiple_examples_from_json.json` -- Batch JSON input

**Expected output:** CIF structure prediction files in the output directory.

### Step 2.3: Synthetic Training Benchmark (TODO)

### Step 2.4: Benchmark Configs (TODO)

### Step 2.5: Profiling Callback (TODO)

---

## Phase 3: Training Performance Results (TODO)

## Phase 4: H200 Comparison & TCO Analysis (TODO)

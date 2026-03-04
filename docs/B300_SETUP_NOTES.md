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
- **cuequivariance on Blackwell**: The `cuequivariance-ops-cu12` v0.9.0 package works on B300 without recompilation, despite being built for cu12. The ops are compatible with the cu128 PyTorch runtime.

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

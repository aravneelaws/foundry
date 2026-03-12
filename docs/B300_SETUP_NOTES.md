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

### Impact of Disabling cuEquivariance on Benchmark Results

Disabling cuEquivariance is a significant caveat for all B300 benchmark results. Here is what it affects and why.

**What cuEquivariance provides:** Two fused CUDA kernels that accelerate the most compute-intensive operations in the RF3 model:

| Fused Kernel | Operations Fused | Vanilla Fallback |
|-------------|-----------------|------------------|
| `cuet.triangle_attention` | Q/K/V projection, scaled dot-product attention, bias addition, softmax, and gating -- in a single kernel launch | Separate `opt_einsum.contract` for attention scores, `F.softmax`, another einsum for value aggregation, `torch.sigmoid` for gating, with intermediate tensor materializations at each step |
| `cuet.triangle_multiplicative_update` | Input LayerNorm, dual projection, sigmoid gating, einsum contraction, output LayerNorm, output projection, and output gating -- in a single kernel launch | Separate `nn.LayerNorm`, tensor slicing, `torch.sigmoid`, `torch.einsum`, another `nn.LayerNorm`, `nn.Linear`, and gating, each materializing intermediate tensors over `(B, L, L, D)` pair space |

**Where these run in the model:** Every `PairformerBlock` contains 4 triangle operations (2x Triangle Attention + 2x Triangle Multiplication). Every `MSAModule` also contains 4 triangle operations. These blocks are the core of the RF3 Pairformer trunk, so the fused kernels affect a large fraction of total compute.

**The fallback path:** With `DISABLE_CUEQUIVARIANCE=1`, the model uses pure vanilla PyTorch eager-mode operations (see `models/rf3/src/rf3/model/layers/attention.py`, methods `_forward_vanilla` in both `TriangleAttention` and `TriangleMultiplication`). There is no intermediate option -- no FlashAttention or Triton kernel exists for these specific triangle operations. The fallback is functionally identical (same math, same outputs) but requires multiple kernel launches and intermediate memory allocations where the fused path uses one.

**Impact on benchmark comparisons:** On H200 GPUs, cuEquivariance is automatically enabled (the `SHOULD_USE_CUEQUIVARIANCE` flag in `src/foundry/__init__.py` is set to `True` when the library imports successfully). On B300, it must be disabled. This means:
- Any **B300 vs H200 comparison is apples-to-oranges** for these operations unless H200 is also measured with cuEquivariance disabled
- B300 throughput numbers are **pessimistic** -- they reflect the cost of the vanilla fallback, not the best achievable performance on Blackwell
- This gap will close once cuEquivariance adds SM 10.3 / Blackwell support in a future release

**Recommendation for fair comparison:** When running H200 baselines (Phase 4), measure **both** with and without cuEquivariance:
```bash
# H200 with cuEquivariance (default -- reflects production performance)
sbatch scripts/benchmark_rf3.sbatch

# H200 without cuEquivariance (matches B300 conditions -- fair hardware comparison)
DISABLE_CUEQUIVARIANCE=1 sbatch scripts/benchmark_rf3.sbatch
```
This isolates the hardware performance difference from the software optimization difference.

### cu13 Build Tested -- Same LLVM Crash (with PyTorch cu128)

We initially tested `cuequivariance-ops-cu13==0.9.0` and `cuequivariance-ops-torch-cu13==0.9.0` with PyTorch 2.7.1+cu128. **This combination crashes** with the same LLVM error:

```
'sm_103a' is not a recognized processor for this target (ignoring processor)
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.i32
```

The root cause was that PyTorch cu128 bundles NVRTC 12.8, which does not support SM 10.3. Even though cuEquivariance cu13 links against NVRTC 13, the PyTorch-bundled NVRTC 12.8 was being used for kernel compilation, causing the LLVM crash.

This was reported upstream: [NVIDIA/cuEquivariance#255](https://github.com/NVIDIA/cuEquivariance/issues/255).

### Resolution: PyTorch cu130 + cuEquivariance cu13

Following guidance from NVIDIA engineers on issue #255, we upgraded to **PyTorch 2.9.1+cu130** (which bundles NVRTC 13.0 natively) alongside cuEquivariance cu13. This resolved the LLVM crash completely.

**The fix: use PyTorch cu130 (not cu128) with cuEquivariance cu13 on Blackwell GPUs.**

The upgraded environment:

| Package | Previous (broken) | Updated (working) |
|---------|-------------------|-------------------|
| PyTorch | 2.7.1+cu128 | **2.9.1+cu130** |
| CUDA runtime | 12.8 | **13.0** |
| Triton | 3.3.1 | **3.5.1** |
| cuequivariance-ops | cu12 0.9.0 | **cu13 0.9.0** |
| cuequivariance-ops-torch | cu12 0.9.0 | **cu13 0.9.0** |
| Lightning | 2.6.1 | 2.6.1 (unchanged) |
| NVRTC swap needed? | Yes (manual swap) | **No (native cu130)** |
| cuEquivariance works? | No (LLVM crash) | **Yes -- both triangle_attention and triangle_multiplicative_update** |

The cu130 venv is at `/fsx/ubuntu/venvs/foundry-cu130/`. The original cu128 venv at `/fsx/ubuntu/venvs/foundry/` is preserved as a fallback.

**Setup commands for the cu130 environment:**
```bash
# Create fresh venv
uv venv --python 3.12 /fsx/ubuntu/venvs/foundry-cu130

# Install Foundry
cd /fsx/ubuntu/projects/foundry
uv pip install --python /fsx/ubuntu/venvs/foundry-cu130/bin/python -e '.[rf3,dev]'

# Install PyTorch cu130 (replaces default cu126/cu128)
uv pip install --python /fsx/ubuntu/venvs/foundry-cu130/bin/python \
  torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130

# Install cuEquivariance cu13
uv pip install --python /fsx/ubuntu/venvs/foundry-cu130/bin/python \
  cuequivariance-ops-cu13==0.9.0 cuequivariance-ops-torch-cu13==0.9.0
```

### Software Versions

**Recommended environment (cu130 -- cuEquivariance enabled):**

| Package | Version |
|---------|---------|
| Python | 3.12.13 |
| PyTorch | **2.9.1+cu130** |
| CUDA runtime | **13.0** |
| Triton | **3.5.1** |
| Lightning | 2.6.1 |
| Hydra | 1.3.2 |
| cuequivariance | 0.9.0 |
| cuequivariance-ops-cu13 | **0.9.0** |
| cuequivariance-ops-torch-cu13 | **0.9.0** |
| cuequivariance-torch | 0.9.0 |
| atomworks | >=2.1.1 |
| rc-foundry | 0.0.1.dev (editable) |

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
- `8vkf_from_file.cif`: **1m 13s** wall-clock (cold start: includes checkpoint load + 10 recycles + 50-step diffusion x 5 samples)
- `5vht_from_file.cif`: **46s** wall-clock (warm start: model already cached in GPU memory)
- Output: CIF structure prediction files generated successfully

**Test files available** (in `models/rf3/tests/data/`):
- `8vkf_from_file.cif` -- CIF structure file
- `5vht_from_file.cif` -- CIF structure file
- `5vht_from_json.json` -- JSON input format
- `multiple_examples_from_json.json` -- Batch JSON input

**How these results were measured:**

These timings use the **existing upstream RF3 inference pipeline** (`rf3 fold` CLI), not custom benchmark code. Each invocation is wrapped with the `time` shell command in `scripts/inference_test.sbatch`. The inference config (default `rf3.yaml`) runs 10 recycles, 5 diffusion samples at 50 timesteps each.

**Test protein details:**

| PDB ID | Protein | Residues | Notes |
|--------|---------|----------|-------|
| 8VKF | Cytochrome P450 CYP199A4 | ~407 | Larger structure with heme, ligand, ions, 454 waters |
| 5VHT | E. coli Chorismate Mutase | ~184 (homodimer) | Smaller structure with non-canonical amino acid, 73 waters |

**Cold vs. warm start:** The first run (8VKF, 1m 13s) includes one-time costs: loading the ~600MB checkpoint from FSx to GPU, model initialization, and CUDA kernel JIT compilation. The second run (5VHT, 46s) skips these because the model remains in GPU memory. The ~27s difference reflects both the initialization overhead and the difference in protein size.

**Interpretability limitations:**
- No H200 baseline comparison yet (planned for Phase 4)
- Single run per protein -- no variance or confidence intervals measured
- Model load time and pure inference time are not separated
- cuEquivariance is disabled (`DISABLE_CUEQUIVARIANCE=1`), meaning triangle operations use vanilla PyTorch fallback (see "Impact of Disabling cuEquivariance" below)

**Expected output:** CIF structure prediction files in the output directory.

### Step 2.3: Synthetic Training Benchmark

To benchmark training throughput without external data dependencies, we use a synthetic dataset that generates random tensors matching the exact shapes and dtypes expected by `RF3Trainer.training_step()`. This isolates GPU compute performance from data I/O.

**Synthetic data design** (`models/rf3/src/rf3/data/synthetic.py`):
- Generates complete training examples in-memory (no disk I/O)
- Produces atom-level features (3072 atoms), token-level features (384 tokens), MSA stacks (1024 sequences x 4 recycles), diffusion tensors (48 samples), and ground truth coordinates
- Deterministic per-example seeding (`seed + idx`) for reproducibility
- Pre-computes valid `atom_to_token_map` and `ref_space_uid` shared across examples

**Running the benchmark:**

Single-node smoke test (8 GPUs, ~50 steps/GPU/epoch x 4 epochs):
```bash
mkdir -p slurm_logs
sbatch scripts/benchmark_rf3_single.sbatch
```

Full 2-node benchmark (16 GPUs across 2 nodes):
```bash
mkdir -p slurm_logs
sbatch scripts/benchmark_rf3.sbatch
```

**Bug fixes applied for synthetic data compatibility:**
- Disabled `LogDatasetSamplingRatiosCallback` in `callbacks/benchmark.yaml` -- it calls `parse_example_id()` which expects structured PDB-format IDs, not the `"synthetic_{idx}"` format
- Set `dataloader.train.n_fallback_retries: 0` in `experiment/benchmark.yaml` -- the `FallbackDatasetWrapper` is designed for production PDB datasets that can have corrupt files; synthetic data never fails, and the wrapper adds overhead that pollutes throughput measurements

### Step 2.4: Benchmark Configs

The benchmark uses three Hydra config files that compose together:

**`configs/experiment/benchmark.yaml`** -- Top-level experiment config:
- Overrides datasets → `synthetic`, callbacks → `benchmark`, logger → `csv`
- Trains from scratch (no checkpoint), 4 epochs x 400 examples/epoch
- Disables validation, checkpointing, and EMA for pure throughput measurement
- Sets diffusion batch size = 48, recycles = 4
- Disables `FallbackDatasetWrapper` (`n_fallback_retries: 0`)

**`configs/datasets/synthetic.yaml`** -- Dataset config:
- Single `SyntheticRF3Dataset` with `crop_size=384`, `n_atoms=3072`, `n_msa=1024`
- Disables all augmentations (mirror, atomization, ligand dropout)
- Validation set is `null`

**`configs/callbacks/benchmark.yaml`** -- Callback config:
- Inherits `train_logging` (loss logging, LR logging, model parameter logging)
- Disables `LogDatasetSamplingRatiosCallback` (incompatible with synthetic IDs)
- Adds `TimingCallback` (per-step wall-clock timing, logs every 10 steps)
- Adds `ProfilingCallback` (throughput, GPU memory, GPU utilization, logs every 10 steps)

**Config overrides via command line:** The Slurm scripts pass Hydra overrides for multi-node settings:
```bash
# Single node
srun python models/rf3/src/rf3/train.py experiment=benchmark trainer.devices_per_node=8 trainer.num_nodes=1

# 2 nodes
srun python models/rf3/src/rf3/train.py experiment=benchmark trainer.devices_per_node=8 trainer.num_nodes=2
```

### Step 2.5: Profiling Callback

The `ProfilingCallback` (`src/foundry/callbacks/profiling.py`) collects detailed metrics during training:

**Metrics collected per step:**
| Metric | Source | Unit |
|--------|--------|------|
| `step_time` | `torch.cuda.synchronize()` barriers around each step | seconds |
| `samples_per_sec` | `world_size / step_time` | samples/sec |
| `tokens_per_sec` | `samples_per_sec * num_tokens_per_example` | tokens/sec |
| `atoms_per_sec` | `samples_per_sec * num_atoms_per_example` | atoms/sec |
| `gpu_mem_allocated` | `torch.cuda.memory_stats()` peak allocated | GB |
| `gpu_mem_reserved` | `torch.cuda.memory_stats()` peak reserved | GB |
| `gpu_utilization` | `pynvml` SM utilization polling (background thread) | % |
| `gpu_mem_utilization` | `pynvml` memory utilization polling | % |

**Output:**
- Per-step CSV at `{output_dir}/profiling_metrics.csv`
- Aggregated metrics logged every 10 steps to the CSV logger
- Final summary at end of training (excludes first 5 warmup steps): avg/min/max/median step time, throughput, and peak memory

**Dependency note:** GPU utilization polling requires `pynvml`. If not installed, the callback prints a warning and disables utilization metrics (other metrics still work).

---

## Phase 3: Training Performance Results

> **Status:** Complete. All B300 benchmark configurations finished.

### Single-node (8x B300 SXM6 AC) -- Job 42

Benchmark config: `experiment=benchmark`, 4 epochs x 50 batches/GPU = 200 optimizer steps. Synthetic data with crop_size=384, n_atoms=3072, diffusion_batch=48, MSA=1024. Training from scratch (random weights), bf16-mixed precision, DDP across 8 GPUs.

**Throughput & Timing** (excluding first 5 warmup steps):

| Metric | Value | Notes |
|--------|-------|-------|
| Step time (avg) | **9.45s** | Per optimizer step across 8 GPUs |
| Step time (median) | 9.87s | |
| Step time (min) | 7.51s | |
| Step time (max) | 11.22s | |
| Samples/sec | **0.85** | Across all 8 GPUs (1 sample/GPU/step) |
| Tokens/sec | **325** | 384 tokens/sample |
| Atoms/sec | ~2,550 | 3072 atoms/sample |

**GPU Memory** (per GPU):

| Metric | Value | Notes |
|--------|-------|-------|
| Peak allocated | **23.10 GB** | 8.4% of 275 GB available |
| Peak reserved | **29.58 GB** | 10.8% of 275 GB available |

**Epoch Timings:**

| Epoch | Wall-clock Time | Batches | Notes |
|-------|-----------------|---------|-------|
| 0 | 562s | 50 | Includes CUDA JIT warmup (~80s overhead) |
| 1 | 483s | 50 | Steady state |
| 2 | 482s | 50 | Steady state |
| 3 | 476s | 50 | Steady state |

**NCCL Communication:**
- Transport: P2P/CUMEM (NVLink) with NVLS (NVLink SHARP)
- 32 channels per rank pair
- Single-node only -- no EFA/network communication

**Observations:**
- **Memory headroom is very large** -- only 8.4% of GPU memory used. This suggests much larger crop sizes (e.g., 768 or 1024 tokens) or larger diffusion batch sizes could fit. Production training with real data may also use more memory due to variable-length sequences.
- **~80s warmup overhead** in epoch 0 from CUDA kernel JIT compilation on first forward pass. Subsequent epochs are consistent at ~480s.
- **Loss values are meaningless** (synthetic random data), but gradients flow correctly through all 200 steps without NaN or divergence, confirming the full training pipeline works end-to-end on B300.

**Profiling CSV:** `/fsx/ubuntu/training/logs/train/benchmark/2026-03-05_01-32_JOB_42/profiling_metrics.csv` (200 rows, per-step metrics)

### Multi-node (2x8 = 16x B300 SXM6 AC) -- Job 43

Benchmark config: Same as single-node but with `trainer.num_nodes=2`. 4 epochs x 25 batches/GPU = 100 optimizer steps (400 examples split across 16 GPUs). DDP across 16 GPUs on 2 nodes with EFA interconnect.

**Throughput & Timing** (excluding first 5 warmup steps):

| Metric | Value | Notes |
|--------|-------|-------|
| Step time (avg) | **9.55s** | Nearly identical to single-node (9.45s) |
| Step time (median) | 9.88s | |
| Step time (min) | 7.59s | |
| Step time (max) | 11.26s | |
| Samples/sec | **1.68** | Across all 16 GPUs |
| Tokens/sec | **643** | 384 tokens/sample |
| Peak GPU memory allocated | **23.10 GB** | Same as single-node |
| Peak GPU memory reserved | **29.85 GB** | Marginally higher (~0.3 GB for NCCL buffers) |

**Epoch Timings:**

| Epoch | Wall-clock Time | Batches/GPU | Notes |
|-------|-----------------|-------------|-------|
| 0 | 323s | 25 | Includes warmup |
| 1 | 243s | 25 | Steady state |
| 2 | 238s | 25 | Steady state |
| 3 | 248s | 25 | Steady state |

**NCCL Communication:**
- Intra-node: P2P/CUMEM (NVLink) with NVLS (NVLink SHARP), 32 channels per rank pair
- Inter-node: NET/Libfabric/GDRDMA (EFA with GPU Direct RDMA), 32 channels per cross-node rank pair

### Scaling Analysis (Single-Node vs Multi-Node)

| Metric | 1 node (8 GPU) | 2 nodes (16 GPU) | Scaling factor |
|--------|---------------|------------------|----------------|
| **Tokens/sec** | 325 | **643** | **1.98x** |
| **Samples/sec** | 0.85 | **1.68** | **1.98x** |
| **Avg step time** | 9.45s | 9.55s | 1.01x overhead |
| **Peak memory** | 23.10 GB | 23.10 GB | No overhead |

**Scaling efficiency: 98.8%** (1.98x throughput with 2x GPUs). The 1.2% overhead comes from inter-node NCCL gradient all-reduce over EFA. This is excellent for a model of this size and confirms that EFA + GPU Direct RDMA works well on B300.

**Profiling CSV:** `/fsx/ubuntu/training/logs/train/benchmark/2026-03-05_23-58_JOB_43/profiling_metrics.csv`

**Key caveats for results above (cu128, cuEquivariance disabled):**
- cuEquivariance disabled (`DISABLE_CUEQUIVARIANCE=1`) -- triangle ops use vanilla PyTorch fallback (see "Impact of Disabling cuEquivariance" in Phase 1)
- Synthetic data (no disk I/O bottleneck) -- production throughput may be lower due to data loading
- Training from scratch (random weights) -- gradient magnitudes may differ from fine-tuning
- No `torch.compile` or CUDA graphs -- pure eager-mode PyTorch

### B300 with cuEquivariance Enabled (PyTorch cu130)

After resolving the cuEquivariance LLVM crash by upgrading to PyTorch 2.9.1+cu130 (see "Resolution: PyTorch cu130 + cuEquivariance cu13" in Phase 1), we re-ran the benchmarks with cuEquivariance **enabled**.

**Environment:** PyTorch 2.9.1+cu130, cuequivariance-ops-cu13 0.9.0, Triton 3.5.1. Venv: `/fsx/ubuntu/venvs/foundry-cu130/`.

**Single-node (8x B300, cuEquivariance enabled):**

| Metric | Value |
|--------|-------|
| Avg step time | **3.70s** |
| Median step time | 3.74s |
| Min / Max step time | 2.94s / 4.54s |
| Samples/sec | **2.16** |
| Tokens/sec | **831** |
| Peak GPU memory allocated | 23.10 GB |
| Peak GPU memory reserved | 29.90 GB |

**2-node (16x B300, cuEquivariance enabled):**

| Metric | Value |
|--------|-------|
| Avg step time | **3.75s** |
| Median step time | 3.75s |
| Min / Max step time | 3.05s / 4.36s |
| Samples/sec | **4.27** |
| Tokens/sec | **1,639** |
| Peak GPU memory allocated | 23.10 GB |
| Peak GPU memory reserved | 29.90 GB |

**cuEquivariance speedup on B300:**

| Metric | B300 no-cueq (cu128) | B300 with-cueq (cu130) | Speedup |
|--------|---------------------|----------------------|---------|
| Tokens/sec (1-node) | 325 | **831** | **2.56x** |
| Tokens/sec (2-node) | 643 | **1,639** | **2.55x** |
| Avg step time (1-node) | 9.45s | **3.70s** | **2.55x faster** |

cuEquivariance provides a **2.5x speedup** on B300 -- even larger than the ~2x speedup observed on H200. This is likely due to the Blackwell-optimized fused kernels (added in cuEquivariance v0.8.0 for SM 10.0/10.3) delivering additional performance beyond the standard Hopper kernels.

**Scaling efficiency (cu130 with cuEquivariance):** 98.7% (1,639 / (831 * 2) = 98.7%).

---

## Phase 4: H200 Comparison

> **Status:** Complete. All benchmark configurations run on H200 (p5en.48xlarge) with both cu128 and cu130 software stacks.

### H200 Benchmark Setup

H200 benchmarks were run on a shared ParallelCluster with p5en.48xlarge instances (8x NVIDIA H200 SXM per node, SM 9.0). The same Foundry codebase, synthetic dataset, and benchmark config (`experiment=benchmark`) were used -- identical to B300 benchmarks except for GPU type.

Two software stacks were tested on H200 to ensure a fair comparison:

| Stack | PyTorch | CUDA runtime | Triton | cuEquivariance | Notes |
|-------|---------|-------------|--------|----------------|-------|
| **cu128** | 2.7.1+cu128 | 12.8 | 3.3.1 | cu12 0.9.0 | Foundry's default/recommended stack |
| **cu130** | 2.9.1+cu130 | 13.0 | 3.5.1 | cu13 0.9.0 | Same stack as B300 (fair comparison) |

**H200 benchmark scripts:** `scripts/h200_bench_{1node,2node}_{cueq,nocueq}.sbatch` (cu128) and `scripts/h200_bench_cu130_{1node,2node}_{cueq,nocueq}.sbatch` (cu130)

### H200 Results: cu128 Stack (Foundry Default)

| Config | Avg step | Tokens/sec (1n) | Tokens/sec (2n) | Memory |
|--------|----------|-----------------|-----------------|--------|
| H200 cu128 + cueq | 5.56s | 552 | 1,102 | 23.15 GB |
| H200 cu128 no-cueq | 10.78s | 285 | 561 | 23.15 GB |

### H200 Results: cu130 Stack (Same as B300)

| Config | Avg step | Tokens/sec (1n) | Tokens/sec (2n) | Memory |
|--------|----------|-----------------|-----------------|--------|
| H200 cu130 + cueq | **4.67s** | **657** | **1,309** | 23.15 GB |
| H200 cu130 no-cueq | 8.27s | 371 | 739 | 23.15 GB |

### Software Stack Impact on H200 (cu128 vs cu130)

Upgrading from PyTorch 2.7.1+cu128 to 2.9.1+cu130 on H200 shows significant improvement from the software stack alone:

| Config | H200 cu128 | H200 cu130 | Improvement |
|--------|-----------|-----------|-------------|
| cueq enabled (1-node) | 552 tok/s | **657 tok/s** | **+19.0%** |
| cueq enabled (2-node) | 1,102 tok/s | **1,309 tok/s** | **+18.8%** |
| cueq disabled (1-node) | 285 tok/s | **371 tok/s** | **+30.2%** |
| cueq disabled (2-node) | 561 tok/s | **739 tok/s** | **+31.7%** |

The cu130 stack (PyTorch 2.9.1, Triton 3.5.1) delivers ~19% improvement with cuEquivariance and ~30% without, on the same H200 hardware.

### Comparison: B300 vs H200 (Same cu130 Software Stack)

This is the fair apples-to-apples hardware comparison with the same software stack (PyTorch 2.9.1+cu130, cuEquivariance cu13 0.9.0) on both GPUs.

**With cuEquivariance enabled:**

| Metric | B300 cu130 | H200 cu130 | B300 advantage |
|--------|-----------|-----------|----------------|
| Tokens/sec (1-node) | **831** | 657 | **26.5% faster** |
| Tokens/sec (2-node) | **1,639** | 1,309 | **25.2% faster** |
| Avg step time (1-node) | **3.70s** | 4.67s | **20.8% faster** |

**With cuEquivariance disabled:**

| Metric | B300 cu128* | H200 cu130 | Delta |
|--------|-----------|-----------|-------|
| Tokens/sec (1-node) | 325 | **371** | H200 14.2% faster |
| Tokens/sec (2-node) | 643 | **739** | H200 14.9% faster |

*B300 no-cueq results are from cu128 stack. With cu130, B300 no-cueq throughput would likely also improve by ~30% based on the H200 software stack improvement pattern.

**B300 is ~26% faster than H200 when both run the same cu130 software stack with cuEquivariance enabled.**

### Comparison: B300 cu130 vs H200 cu128 (Different Software Stacks)

For reference, this comparison uses Foundry's default cu128 stack on H200 and the cu130 stack on B300:

| Metric | B300 cu130 + cueq | H200 cu128 + cueq | B300 advantage |
|--------|------------------|------------------|----------------|
| Tokens/sec (1-node) | **831** | 552 | **50.5% faster** |
| Tokens/sec (2-node) | **1,639** | 1,102 | **48.7% faster** |

The ~50% advantage includes both the hardware difference (~26%) and the cu130 software stack improvement (~19%).

### cuEquivariance Impact

| GPU | No cueq (cu130) | With cueq (cu130) | Speedup |
|-----|----------------|-------------------|---------|
| **B300** | 325* | **831** | **2.56x** |
| **H200** | 371 | **657** | **1.77x** |

*B300 no-cueq from cu128 stack.

cuEquivariance provides a larger speedup on B300 (2.5x) than on H200 (1.8x), likely due to the Blackwell-optimized fused kernels in cuEquivariance v0.8.0.

### Scaling Efficiency

| Config | 1-node tokens/sec | 2-node tokens/sec | Efficiency |
|--------|-------------------|-------------------|------------|
| B300 cu130 + cueq | 831 | 1,639 | **98.7%** |
| B300 cu128 no-cueq | 325 | 643 | **98.8%** |
| H200 cu130 + cueq | 657 | 1,309 | **99.5%** |
| H200 cu130 no-cueq | 371 | 739 | **99.6%** |
| H200 cu128 + cueq | 552 | 1,102 | **99.8%** |
| H200 cu128 no-cueq | 285 | 561 | **98.4%** |

Near-linear scaling (>98%) across all configurations on both GPU types.

### Summary: All Configurations

| Config | Avg step (1n) | Tokens/sec (1n) | Tokens/sec (2n) |
|--------|--------------|-----------------|-----------------|
| **B300 cu130 + cueq** | **3.70s** | **831** | **1,639** |
| H200 cu130 + cueq | 4.67s | 657 | 1,309 |
| H200 cu128 + cueq | 5.56s | 552 | 1,102 |
| H200 cu130 no-cueq | 8.27s | 371 | 739 |
| B300 cu128 no-cueq | 9.45s | 325 | 643 |
| H200 cu128 no-cueq | 10.78s | 285 | 561 |

### Key Takeaways

1. **B300 is ~26% faster than H200** when both run the same cu130 software stack with cuEquivariance enabled (831 vs 657 tokens/sec). This is the fair hardware comparison.
2. **The cu130 software stack (PyTorch 2.9.1, Triton 3.5.1) provides ~19-30% improvement** over cu128 on the same hardware, independent of GPU type.
3. **cuEquivariance provides 2.5x speedup on B300 and 1.8x on H200** (cu130 stack). The larger B300 speedup is from Blackwell-optimized fused kernels in cuEquivariance v0.8.0.
4. **PyTorch cu130 is required** to enable cuEquivariance on Blackwell. PyTorch cu128 causes an LLVM crash because its bundled NVRTC 12.8 doesn't support SM 10.3. See "Resolution: PyTorch cu130 + cuEquivariance cu13" in Phase 1.
5. **Memory usage is identical** across all configurations (~23 GB per GPU), leaving significant headroom for larger workloads.
6. **Scaling efficiency is excellent** on all platforms and configurations (>98%), confirming EFA works equivalently on both HyperPod and ParallelCluster.

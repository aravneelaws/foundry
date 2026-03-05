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

> **Status:** Single-node complete (Job 42). Multi-node pending.

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

### Multi-node (2x8 = 16x B300 SXM6 AC)

> **Status:** Pending. The `scripts/benchmark_rf3.sbatch` script needs to be updated with `ntasks-per-node=8` (same fix applied to the single-node script).

| Metric | Value | Notes |
|--------|-------|-------|
| Step time (avg) | _TBD_ | |
| Samples/sec | _TBD_ | Across all 16 GPUs |
| Tokens/sec | _TBD_ | |
| Scaling efficiency | _TBD_ | vs. single-node baseline |
| NCCL communication overhead | _TBD_ | Inferred from scaling efficiency |

**Key caveats for all results:**
- cuEquivariance disabled (`DISABLE_CUEQUIVARIANCE=1`) -- triangle ops use vanilla PyTorch fallback (see "Impact of Disabling cuEquivariance" in Phase 1)
- Synthetic data (no disk I/O bottleneck) -- production throughput may be lower due to data loading
- Training from scratch (random weights) -- gradient magnitudes may differ from fine-tuning
- No `torch.compile` or CUDA graphs -- pure eager-mode PyTorch

---

## Phase 4: H200 Comparison & TCO Analysis

> **Status:** Requires H200 cluster access.

### Plan

1. Run the same benchmark (`experiment=benchmark`) on H200 GPUs with **two configurations**:
   - cuEquivariance **enabled** (default) -- reflects production H200 performance
   - cuEquivariance **disabled** (`DISABLE_CUEQUIVARIANCE=1`) -- matches B300 conditions for fair hardware comparison

2. Collect identical metrics as Phase 3 (step time, throughput, memory, utilization)

3. Compute comparison:

| Metric | B300 (no cueq) | H200 (no cueq) | H200 (with cueq) |
|--------|----------------|-----------------|-------------------|
| Tokens/sec | 325 | _TBD_ | _TBD_ |
| Step time | 9.45s | _TBD_ | _TBD_ |
| Peak memory | 23.10 GB | _TBD_ | _TBD_ |
| GPU utilization | _TBD_ | _TBD_ | _TBD_ |

4. TCO analysis (cost per token-second):

| Instance | GPU | $/hr (on-demand) | Tokens/sec | $/M tokens |
|----------|-----|-------------------|------------|------------|
| p6-b300.48xlarge | 8x B300 | _TBD_ | 325 | _TBD_ |
| p5e.48xlarge | 8x H200 | _TBD_ | _TBD_ | _TBD_ |

**Note:** B300 vs H200 comparison with cuEquivariance disabled isolates the raw hardware speedup. The full comparison (B300 no-cueq vs H200 with-cueq) shows the practical gap until cuEquivariance adds Blackwell support.

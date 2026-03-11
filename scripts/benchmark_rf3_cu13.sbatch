#!/bin/bash
#SBATCH --job-name=rf3-bench-cu13
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%j-bench-cu13.out
#SBATCH --error=slurm_logs/%j-bench-cu13.err

# RF3 Training Benchmark: cu13 cuEquivariance experiment
# Same config as benchmark_rf3_single.sbatch but with cuEquivariance ENABLED.
# Uses cu13 builds of cuequivariance-ops which have Blackwell-optimized
# triangle_attention kernels (SM 10.0/10.3).
#
# Usage:
#   mkdir -p slurm_logs
#   sbatch scripts/benchmark_rf3_cu13.sbatch

set -euo pipefail

# ---- Environment ----
source /fsx/ubuntu/venvs/foundry/bin/activate
cd /fsx/ubuntu/projects/foundry

# cuEquivariance ENABLED (no DISABLE_CUEQUIVARIANCE)
# The cu13 builds provide Blackwell-optimized triangle_attention kernels.
# triangle_multiplicative_update may crash on Blackwell (known issue since v0.6.0)
# but the code has graceful fallback to vanilla PyTorch per-op.

# Disable AOT tuning (use default kernel configs for consistent benchmarking)
export CUEQ_DISABLE_AOT_TUNING=1
export CUEQ_DEFAULT_CONFIG=1

# ---- NCCL configuration ----
export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:${LD_LIBRARY_PATH:-}

# ---- Distributed training env vars ----
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# ---- Print environment ----
echo "===== cu13 cuEquivariance Benchmark ====="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: 8"
echo "cuEquivariance: ENABLED (cu13)"
echo "=========================================="

# ---- Launch training ----
srun python models/rf3/src/rf3/train.py \
    experiment=benchmark \
    trainer.devices_per_node=8 \
    trainer.num_nodes=1

echo ""
echo "===== Benchmark Complete ====="
echo "Check profiling_metrics.csv in the Hydra output directory for detailed metrics."

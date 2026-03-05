#!/bin/bash
#SBATCH --job-name=rf3-bench-1node
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%j-bench-1node.out
#SBATCH --error=slurm_logs/%j-bench-1node.err

# RF3 Training Benchmark: 1 node x 8 B300 GPUs
# Uses synthetic data for throughput measurement.
# Run this first as a smoke test before the 2-node benchmark.
#
# Usage:
#   mkdir -p slurm_logs
#   sbatch scripts/benchmark_rf3_single.sbatch

set -euo pipefail

# ---- Environment ----
source /fsx/ubuntu/venvs/foundry/bin/activate
cd /fsx/ubuntu/projects/foundry

# Disable cuequivariance (LLVM backend does not support Blackwell SM 10.3 yet)
export DISABLE_CUEQUIVARIANCE=1

# ---- NCCL configuration ----
# Even for single-node, set these for consistency
export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:${LD_LIBRARY_PATH:-}

# ---- Distributed training env vars ----
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Override SLURM task count so Lightning Fabric's SLURMEnvironment validation
# passes. We use ntasks-per-node=1 at the SLURM level but need Fabric to see
# 8 tasks (one per GPU) for DDP. Fabric's SubprocessScriptLauncher then spawns
# the additional worker processes.
export SLURM_NTASKS=8
export SLURM_NTASKS_PER_NODE=8

# ---- Print environment ----
echo "===== Benchmark Configuration ====="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: 8"
echo "===================================="

# ---- Launch training ----
srun python models/rf3/src/rf3/train.py \
    experiment=benchmark \
    trainer.devices_per_node=8 \
    trainer.num_nodes=1

echo ""
echo "===== Benchmark Complete ====="
echo "Check profiling_metrics.csv in the Hydra output directory for detailed metrics."

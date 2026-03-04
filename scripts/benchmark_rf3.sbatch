#!/bin/bash
#SBATCH --job-name=rf3-bench-2node
#SBATCH --partition=dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j-bench-2node.out
#SBATCH --error=slurm_logs/%j-bench-2node.err

# RF3 Training Benchmark: 2 nodes x 8 B300 GPUs (16 GPUs total)
# Uses synthetic data for throughput measurement.
#
# Usage:
#   mkdir -p slurm_logs
#   sbatch scripts/benchmark_rf3.sbatch

set -euo pipefail

# ---- Environment ----
source /fsx/ubuntu/venvs/foundry/bin/activate
cd /fsx/ubuntu/projects/foundry

# ---- NCCL / EFA configuration ----
export NCCL_DEBUG=INFO
export NCCL_PROTO=Simple
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:${LD_LIBRARY_PATH:-}
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-tuner-ofi.so

# ---- Distributed training env vars ----
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

# ---- Print environment ----
echo "===== Benchmark Configuration ====="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "GPUs per node: 8"
echo "Total GPUs: $((SLURM_NNODES * 8))"
echo "===================================="

# ---- Launch training ----
srun python models/rf3/src/rf3/train.py \
    experiment=benchmark \
    trainer.devices_per_node=8 \
    trainer.num_nodes=2

echo ""
echo "===== Benchmark Complete ====="
echo "Check profiling_metrics.csv in the Hydra output directory for detailed metrics."

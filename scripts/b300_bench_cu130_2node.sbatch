#!/bin/bash
#SBATCH --job-name=b300-cu130-2n
#SBATCH --partition=dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j-b300-cu130-2n.out
#SBATCH --error=slurm_logs/%j-b300-cu130-2n.err

# B300 2-node with cu130 PyTorch + cuEquivariance ENABLED
set -euo pipefail
source /fsx/ubuntu/venvs/foundry-cu130/bin/activate
cd /fsx/ubuntu/projects/foundry

export CUEQ_DISABLE_AOT_TUNING=1
export CUEQ_DEFAULT_CONFIG=1
export NCCL_DEBUG=INFO
export NCCL_PROTO=Simple
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:${LD_LIBRARY_PATH:-}
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-tuner-ofi.so
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

echo "===== B300 cu130 Benchmark: 2-node, cuEquivariance ENABLED ====="
echo "Date: $(date) | Job: $SLURM_JOB_ID | Nodes: $SLURM_NODELIST"
echo "PyTorch: cu130 | cuEquivariance: ENABLED"

srun python models/rf3/src/rf3/train.py experiment=benchmark trainer.devices_per_node=8 trainer.num_nodes=2

echo "===== Benchmark Complete ====="

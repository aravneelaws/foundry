#!/bin/bash
#SBATCH --job-name=h200-1n-nocueq
#SBATCH --partition=p5en
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%j-h200-1n-nocueq.out
#SBATCH --error=slurm_logs/%j-h200-1n-nocueq.err

# Config A: H200 1-node, cuEquivariance DISABLED
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
source /fsx/aravneel/venvs/foundry/bin/activate
cd /fsx/aravneel/projects/foundry

export DISABLE_CUEQUIVARIANCE=1
export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:${LD_LIBRARY_PATH:-}
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

echo "===== H200 Benchmark: 1-node, cuEquivariance DISABLED ====="
echo "Date: $(date) | Job: $SLURM_JOB_ID | Node: $(hostname)"

srun python models/rf3/src/rf3/train.py experiment=benchmark paths.log_dir=/fsx/aravneel/training/logs trainer.devices_per_node=8 trainer.num_nodes=1

echo "===== Benchmark Complete ====="

#!/bin/bash -l
#SBATCH --job-name=finetune_phi_new
#SBATCH --output=res.txt
#SBATCH --error=error.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:8 -C a100_80
#SBATCH --partition=a100
#SBATCH --export=NONE              # do not export environment from submitting shell


# debug info
hostname
nvidia-smi

source ~/.bashrc
module load python/3.9-anaconda
module load cuda/12.6
conda activate /home/atuin/b211dd/b211dd20/software/private/conda/envs/compression

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export WANDB_API_KEY=7f44c9bc6dbfe5a583a45074c3fef843a7d6c215
export WANDB_MODE="offline"


sh /home/hpc/b211dd/b211dd20/compression/tinyllava_framework/scripts/train/train_phi_new.sh
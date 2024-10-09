#!/bin/bash -l
#SBATCH --job-name=bpc-phi-finetune
#SBATCH --output=res_bpc_finetune_math_all.txt
#SBATCH --error=error_bpc_finetune_math_all.txt
#SBATCH --ntasks=1    
#SBATCH --gres=gpu:a100:2
#SBATCH --time=6:00:00
#SBATCH --partition=a100
#SBATCH --export=NONE

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


python /home/hpc/b211dd/b211dd20/compression/tinyllava_framework/tinyllava/eval/bpc4tinyllava.py \
    --task_name arxiv_math \
    --block_size 1024 \
    --stride 512 \
    --batch_size 8 \
    --cache_dir /home/atuin/b211dd/b211dd20/bpc
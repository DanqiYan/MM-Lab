#!/bin/bash

#SBATCH --job-name=pretrain_0630_finetune_lora         # 作业名称
#SBATCH --output=output_%j.txt          # 标准输出和错误日志文件名 (%j 将被作业ID替换)

#SBATCH --gres=gpu:4                   # 需要的GPU数量
#SBATCH --time=24:00:00                 # 运行时间上限 (hh:mm:ss)
#SBATCH --partition=accelerated-h100               # 作业提交的分区名称
#SBATCH --mail-type=FAIL,END  
#SBATCH --mail-user=yujun_wang_cn@hotmail.com

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tlf1


sh scripts/exp/exp_0630/finetune_lora_llava-dataset.sh
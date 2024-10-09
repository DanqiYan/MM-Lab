#!/bin/bash -l

#SBATCH --job-name=textvqa-eval
#SBATCH --output=res_textvqa.txt
#SBATCH --error=error_textvqa.txt
#SBATCH --ntasks=1    
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --partition=a40
#SBATCH --export=NONE

# debug info
hostname
nvidia-smi

source ~/.bashrc
module load python/3.9-anaconda
module load cuda/12.6
conda activate /home/atuin/b211dd/b211dd20/software/private/conda/envs/compression


MODEL_PATH="/home/atuin/b211dd/b211dd20/tinyllava/checkpoints/llava_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune_full"
MODEL_NAME="tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune"
EVAL_DIR="/home/atuin/b211dd/b211dd20/dataset"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $EVAL_DIR/textvqa/train_images \
    --answers-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python -m tinyllava.eval.eval_textvqa \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val.json \
    --result-file $EVAL_DIR/textvqa/answers/$MODEL_NAME.jsonl


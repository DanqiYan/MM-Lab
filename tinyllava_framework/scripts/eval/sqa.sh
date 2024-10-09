#!/bin/bash -l

#SBATCH --job-name=scienceqa-eval
#SBATCH --output=res_sqa.txt
#SBATCH --error=error_sqa.txt
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
python -m tinyllava.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/scienceqa/llava_test_CQM-A.json \
    --image-folder $EVAL_DIR/scienceqa/images/test \
    --answers-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi

python tinyllava/eval/eval_science_qa.py \
    --base-dir $EVAL_DIR/scienceqa \
    --result-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --output-file $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_output.jsonl \
    --output-result $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_result.json


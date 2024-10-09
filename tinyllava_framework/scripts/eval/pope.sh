#!/bin/bash

MODEL_PATH= "/home/atuin/b211dd/b211dd20/tinyllava/checkpoints/llava_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune_full"
MODEL_NAME= "tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune_full"
EVAL_DIR="/home/ai/data/llava/dataset/eval"

python -m tinyllava.eval.model_vqa_pope \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl

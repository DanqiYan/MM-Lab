#!/bin/bash
# if [ $# -ne 10 ] || [ $# -ne 10 ]; then
#     echo "Usage: $0 <DATA_PATH> <IMAGE_PATH> <TINYLLAVA_VERSION> <PRETRAINED_MODEL_PATH> <VERSION> <TRAIN_RECIPE> <LOREFT_CONFIG_PATH> <INTERVENTION_POSITIONS> <REFT_SHARE_WEIGHTS> <INTERVENE_INCLUDE_IMG_EMBED> <CONV_VERSION> <LLM_VERSION> <VT_VERSION> <VT_VERSION2> <CN_VERSION> <MODEL_MAX_LENGTH>"
#     exit 1
# fi

# Assign the arguments to variables
DATA_PATH="$1"
IMAGE_PATH="$2"
VERSION="$3"
TRAIN_RECIPE="$4"
LOREFT_CONFIG_PATH="$5"
INTERVENTION_POSITIONS="$6"
REFT_SHARE_WEIGHTS="$7"
INTERVENE_INCLUDE_IMG_EMBED="$8"

TINYLLAVA_VERSION="${9:-None}"
PRETRAINED_MODEL_PATH="${10:-None}"

CONV_VERSION="${11:-None}"
LLM_VERSION="${12:-None}"
VT_VERSION="${13:-None}"
VT_VERSION2="${14:-None}"
CN_VERSION="${15:-None}"
MODEL_MAX_LENGTH="${16:-None}"

TUNE_TYPE_LLM="${17:-None}"

TINYLLAVA_VERSION_NAME=$(echo $TINYLLAVA_VERSION | cut -d'/' -f2)

cmd="deepspeed --include localhost:0,1,2,3 --master_port 29503 tinyllava/train/train_for_finetune.py \
    --deepspeed ./scripts/zero2.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --fp16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --loreft_config_path "$LOREFT_CONFIG_PATH" \
    --intervention_positions "$INTERVENTION_POSITIONS" \
    --reft_share_weights "$REFT_SHARE_WEIGHTS" \
    --intervene_include_img_embed "$INTERVENE_INCLUDE_IMG_EMBED" \
    --lora_r 128 \
    --lora_alpha 256 \
    --group_by_modality_length False \
    --output_dir /home/atuin/b211dd/b211dd19/data/checkpoints/llava_factory/three_stage_finetuning/${VERSION} \
    --num_train_epochs 6 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --tokenizer_use_fast False \
    --run_name tiny-llava-3rd_stage_finetuning-${VERSION}"


if [ "$TINYLLAVA_VERSION" != "None" ]; then
    cmd="$cmd --tinyllava_version \"$TINYLLAVA_VERSION\""
fi

if [ "$PRETRAINED_MODEL_PATH" != "None" ]; then
    cmd="$cmd --pretrained_model_path \"$PRETRAINED_MODEL_PATH\""
fi

if [ "$CONV_VERSION" != "None" ]; then
    cmd="$cmd --conv_version \"$CONV_VERSION\""
fi
if [ "$LLM_VERSION" != "None" ]; then
    cmd="$cmd --model_name_or_path \"$LLM_VERSION\""
fi
if [ "$VT_VERSION" != "None" ]; then
    cmd="$cmd --vision_tower \"$VT_VERSION\""
fi
if [ "$VT_VERSION2" != "None" ]; then
    cmd="$cmd --vision_tower2 \"$VT_VERSION2\""
fi
if [ "$CN_VERSION" != "None" ]; then
    cmd="$cmd --connector_type \"$CN_VERSION\""
fi
if [ "$MODEL_MAX_LENGTH" != "None" ]; then
    cmd="$cmd --model_max_length \"$MODEL_MAX_LENGTH\""
fi

if [ "$TUNE_TYPE_LLM" != "None" ]; then
    cmd="$cmd --tune_type_llm \"$TUNE_TYPE_LLM\""
fi


eval $cmd
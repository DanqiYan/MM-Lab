# DATA_PATH=/home/ai/data/llava/dataset/text_files/blip_laion_cc_sbu_558k.json
FINETUNE_DATA_PATH=/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/train/text_files/llava_v1_5_mix665k.json
# IMAGE_PATH=/home/ai/data/llava/dataset/llava/llava_pretrain/images
FINETUNE_IMAGE_PATH=/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/train

LLM_VERSION=microsoft/phi-2
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
CONV_VERSION=phi
VERSION=base-llava_dataset-bs32
PRETRAIN_TRAIN_RECIPE=common
FINETUNE_TRAIN_RECIPE=lora
MODEL_MAX_LENGTH=3072


# bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$PRETRAIN_TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/lora/finetune_lora.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$FINETUNE_TRAIN_RECIPE" "$MODEL_MAX_LENGTH"

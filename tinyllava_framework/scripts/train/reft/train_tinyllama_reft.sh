DATA_PATH=/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/train/text_files/blip_laion_cc_sbu_558k.json
# FINETUNE_DATA_PATH=/home/hk-project-starter-p0022188/tum_piz8108/code/TinyLLaVA_Factory/data_few_samples.json
IMAGE_PATH=/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/train/llava/llava_pretrain/images
# FINETUNE_IMAGE_PATH=/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/train/llava/llava_pretrain/images

# FINETUNE_DATA_PATH=/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/train/text_files/llava_v1_5_mix665k.json
# FINETUNE_IMAGE_PATH=/home/ai/data/llava/dataset

LOREFT_CONFIG_PATH=scripts/train/reft/loreft_config.json
# microsoft/phi-2 TinyLlama/TinyLlama-1.1B-Chat-v1.0
LLM_VERSION=microsoft/phi-2
# google/siglip-so400m-patch14-384
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
# phi llama
CONV_VERSION=phi
VERSION=base-llava_dataset-bs32
PRETRAIN_TRAIN_RECIPE=common
FINETUNE_TRAIN_RECIPE=reft
# 3072 2048
MODEL_MAX_LENGTH=3072



bash scripts/train/pretrain_reft.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$PRETRAIN_TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
# bash scripts/train/reft/finetune_reft.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$FINETUNE_TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$LOREFT_CONFIG_PATH"

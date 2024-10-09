DATA_PATH=/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/train/text_files/blip_laion_cc_sbu_558k.json
IMAGE_PATH=/home/hk-project-starter-p0022188/tum_piz8108/data/tinyllava/train/llava/llava_pretrain/images


LLM_VERSION=microsoft/phi-2
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
CONV_VERSION=phi
VERSION=base-llava_dataset-bs32
PRETRAIN_TRAIN_RECIPE=common
FINETUNE_TRAIN_RECIPE=reft
MODEL_MAX_LENGTH=3072



bash scripts/train/pretrain_reft.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$PRETRAIN_TRAIN_RECIPE" "$MODEL_MAX_LENGTH"

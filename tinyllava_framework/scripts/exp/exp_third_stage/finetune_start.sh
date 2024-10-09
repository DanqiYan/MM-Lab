# FINETUNE_DATA_PATH=/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/text_files/llava_v1_5_mix665k.json
# FINETUNE_DATA_PATH=/home/hpc/b211dd/b211dd19/code/reft_vlm/data_few_samples.json
# FINETUNE_DATA_PATH=/home/hpc/b211dd/b211dd19/code/reft_vlm/data_few_samples_text_caps.json
FINETUNE_DATA_PATH=/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train/text_files_small_dataset/text_caps.json
FINETUNE_IMAGE_PATH=/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/train

# reft config
LOREFT_CONFIG_PATH=scripts/exp/exp_third_stage/loreft_config1.json
INTERVENTION_POSITIONS=f9+l9
REFT_SHARE_WEIGHTS=True
INTERVENE_INCLUDE_IMG_EMBED=True

TINYLLAVA_VERSION=tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B
# PRETRAINED_MODEL_PATH=/home/atuin/b211dd/b211dd19/data/checkpoints/llava_factory/three_stage_finetuning/third_stage-text_caps-lora
# PRETRAINED_MODEL_PATH=/home/atuin/b211dd/b211dd19/data/checkpoints/llava_factory/three_stage_finetuning/text_caps-loreft-pre-test1
# PRETRAINED_MODEL_PATH=/home/atuin/b211dd/b211dd19/data/checkpoints/llava_factory/three_stage_finetuning/text_caps-loreft-pre
CONV_VERSION=phi

# LLM_VERSION=microsoft/phi-2
# VT_VERSION=google/siglip-so400m-patch14-384
# VT_VERSION2=""
# CN_VERSION=mlp2x_gelu
# MODEL_MAX_LENGTH=3072

VERSION=text_caps-phi-lora-pre-small
FINETUNE_TRAIN_RECIPE=lora

TUNE_TYPE_LLM=lora


bash scripts/exp/exp_third_stage/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$VERSION" "$FINETUNE_TRAIN_RECIPE" "$LOREFT_CONFIG_PATH" "$INTERVENTION_POSITIONS" "$REFT_SHARE_WEIGHTS" "$INTERVENE_INCLUDE_IMG_EMBED" "$TINYLLAVA_VERSION" "$PRETRAINED_MODEL_PATH" "$CONV_VERSION" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$MODEL_MAX_LENGTH" "$TUNE_TYPE_LLM"

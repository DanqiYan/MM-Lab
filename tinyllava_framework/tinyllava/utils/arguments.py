from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, TYPE_CHECKING
import transformers


if TYPE_CHECKING:
    import transformers


def init_reft_pos_configs(
    intervention_positions="f1+l1",
    reft_share_weights=True,
    intervened_prompt_part="first_round",
    intervene_include_img_embed=False,
    num_interventions=None,
    img_embed_token_len=None
):
    return {
        "intervention_positions": intervention_positions,
        "reft_share_weights": reft_share_weights,
        "intervened_prompt_part": intervened_prompt_part,
        "intervene_include_img_embed": intervene_include_img_embed,
        "num_interventions": num_interventions,
        "img_embed_token_len": img_embed_token_len
    }


@dataclass
class ModelArguments:
    cache_dir: Optional[str] = field(default=None)
    
    tinyllava_version: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer_name_or_path: Optional[str] = field(default=None)
    attn_implementation: Optional[str] = field(default=None)
    vision_tower: Optional[str] = field(default='')
    vision_tower2: Optional[str] = field(default='')
    connector_type: str = field(default='linear')
    
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    resampler_hidden_size: Optional[int] = field(default=768)
    num_queries: Optional[int] = field(default=128)
    num_resampler_layers: Optional[int] = field(default=3)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tokenizer_use_fast: bool = field(default=False)
    tokenizer_padding_side: str = field(default='right')


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    conv_version: str = 'pretrain'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_recipe: str = field(default='common')
    tune_type_llm: str = field(default="frozen") # support only: frozen, full, lora, qlora_int4, qlora_int8, loreft
    tune_type_vision_tower: str = field(default="frozen") # support only: frozen, full, partially-tune, loreft
    tune_vision_tower_from_layer: Optional[int] = field(default=10)
    tune_type_connector: str = field(default="full") # support only: frozen, full
    tune_embed_tokens: Optional[int] = field(default=False)
    
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    vision_tower_lr: Optional[float] = None
    pretrained_model_path: Optional[str] = None
    
    # reft configs
    loreft_config_path: str = field(default="none")
    intervention_positions: Optional[str] = None
    reft_share_weights: Optional[bool] = None
    intervened_prompt_part: Optional[str] = None
    reft_pos_configs: Optional[Dict] = None
    intervene_include_img_embed: Optional[bool] = None
   
    # extra added configs
    epoch_to_save: int = 1
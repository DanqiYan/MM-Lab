from packaging import version
import pathlib

import tokenizers
import transformers
import debugpy

# import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.train import *
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module
from tinyllava.utils import log

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments) 
    return model_args

def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args

def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args

def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args

def reft_state(training_arguments):
    train_with_reft = hasattr(training_arguments, 'training_recipe') and training_arguments.training_recipe == "reft"
    pretrained_reft = hasattr(training_arguments, 'pretrained_model_path') and training_arguments.pretrained_model_path and "reft" in os.listdir(training_arguments.pretrained_model_path)
    # reft_pos_configs_exist = train_with_reft and hasattr(model, 'reft_pos_configs') and model.reft_pos_configs
    return train_with_reft, pretrained_reft

def generate_model_args(config_in_model, **kwargs):
    # TODO: do not consider model_name_or_path2 (for vision_tower 2)
    pretrained_model_path = kwargs['pretrained_model_path'] if 'pretrained_model_path' in kwargs else None

    return {
        "llm": {
            "model_name_or_path": config_in_model.llm_model_name_or_path,
            "cache_dir": config_in_model.cache_dir,
            "attn_implementation": getattr(kwargs, "attn_implementation", "flash_attention_2"),
            "torch_dtype": getattr(kwargs, "torch_dtype", torch.float16),
            "pretrained_llm_path": os.path.join(pretrained_model_path, "language_model")
        },
        "vision_tower": {
            "model_name_or_path": config_in_model.vision_model_name_or_path,
            "pretrained_vision_tower_path": os.path.join(pretrained_model_path, "vision_tower")
        },
        "connector": {
            "connector_type": config_in_model.connector_type,
            "pretrained_connector_path": os.path.join(pretrained_model_path, "connector")
        }
    }

def determine_train_mode(model_arguments, training_arguments):
    def print_important_message(message):
        """
        print with red and bold font
        """
        # red = '\033[91m'
        green = '\033[92m'
        bold = '\033[1m'
        end_format = '\033[0m'

        formatted_message = f"{bold}{green}{message}{end_format}"
        return formatted_message
    
    print(print_important_message('============================================='))
    if training_arguments.pretrained_model_path:
        log(print_important_message("Model will be loaded from tinyllava-ckpt."))
        train_mode = 'tl-ckpt'
    elif model_arguments.tinyllava_version: 
        log(print_important_message("Model will be loaded from hugging-face repo of tinnyllava-model."))
        train_mode = 'hf-tl-ckpt'
    else:
        log(print_important_message("Model will be initialized by loading llm and vision_tower seperately."))
        train_mode = 'hf-llm_vt'
    print(print_important_message('============================================='))
    return train_mode


def train():
    """
    The train has 3 modes (ranked by priority):
    (1) load model and weights from tinyllava-ckpt (if pretrained_model_path exists)
    (2) load model and weights from hf-tinyllava-ckpt (if tinyllava_version exists)
    (3) load model and weights seperately from llm and vt (otherwise)
    """
    # debugpy.listen(("0.0.0.0", 5677))
    # print("waitng for debugger attach ...")
    # debugpy.wait_for_client()
    # debugpy.breakpoint()
    # print("debugger is attached!")
    
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    
    logger_setting(getattr(training_arguments, 'output_dir', None))
    train_mode = determine_train_mode(model_arguments, training_arguments)

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 

    if train_mode == "tl-ckpt":
        model_config = TinyLlavaConfig.from_pretrained(training_arguments.pretrained_model_path)
        model = TinyLlavaForConditionalGeneration(model_config)
        model_args = generate_model_args(config_in_model=model_config,
                                        pretrained_model_path=training_arguments.pretrained_model_path)
        
        model = training_recipe.load(model, model_args)
    elif train_mode == "hf-tl-ckpt":
        model = TinyLlavaForConditionalGeneration.from_pretrained(model_arguments.tinyllava_version, trust_remote_code=True)
    elif train_mode == "hf-llm_vt":
        model_args = load_settings(model_arguments, data_arguments, training_arguments)
        model_args = training_recipe.add_args(model_args)
        model_config = TinyLlavaConfig()
        model_config.load_from_config(model_arguments)
        model = TinyLlavaForConditionalGeneration(model_config)
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])
    else:
        raise ValueError("NOT supported training mode!")

    # reft
    train_with_reft, pretrained_reft = reft_state(training_arguments)
    if train_with_reft and not pretrained_reft:
        model.add_reft_adaptor_from_training_arguments(training_arguments)
    elif train_with_reft and pretrained_reft:
        model.load_reft_adaptor_from_ckpt(training_arguments.pretrained_model_path)

    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer

    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    if train_with_reft and hasattr(model, 'reft_pos_configs') and model.reft_pos_configs:
        data_arguments.reft_pos_configs = model.reft_pos_configs
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)
    log_trainable_params(model)  # not work well with zero3

    trainer = LLaVATrainer(model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)
    trainer.add_callback(SaveCallback(trainer, training_recipe))
    trainer.add_callback(WandbLogCallback(model))
    
    trainer.train()

if __name__ == "__main__":
    train()

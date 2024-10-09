import os, json, copy, re

import torch
import pyvene

from .base import BaseTrainingRecipe
from . import register_training_recipe
from ..utils.train_utils import *
from ..utils import log, init_reft_pos_configs
from ..model.llm import LLM_FACTORY, LLMFactory, full_name_to_LLM_register_name
from ..model.vision_tower import VISION_TOWER_FACTORY, VisionTowerFactory, full_name_to_VT_register_name
from ..model.reft_mappings import *
from ..model.reft_model_shell import *


@register_training_recipe('reft')
class ReFTTrainingRecipe(BaseTrainingRecipe):
    def __init__(self, training_arguments):
        super().__init__(training_arguments)
        self.training_arguments = training_arguments
        self.loreft_skip_module = ['connector', 'vision_tower', 'language_model']

    def training_model_converse(self, model):
        if self.training_arguments.bits == 16:
            if self.training_arguments.bf16:
                model.to(torch.bfloat16)
            if self.training_arguments.fp16:
                model.to(torch.float16)

        return model
    
        
    def save(self, model, trainer):
        model.config.use_cache = True
        #save tokenizer       
        model.tokenizer.save_pretrained(self.training_arguments.output_dir)
        #save entire model config
        model.config.save_pretrained(self.training_arguments.output_dir, from_pt=True)
        #save trainer
        trainer.save_state() 

        #save language model base params
        if isinstance(model.language_model, ReftModelShell):
            lm = model.language_model.model
            reft_lm_output_dir = os.path.join(os.path.join(self.training_arguments.output_dir, 'reft'), 'language_model')
            model.language_model.save(reft_lm_output_dir)   # save reft params
            with open(os.path.join(reft_lm_output_dir, 'intervention_config.json'), 'w') as json_file:
                json.dump(model.reft_config['llm'], json_file, indent=4)
            if model.reft_pos_configs:
                with open(os.path.join(reft_lm_output_dir, 'reft_pos_configs.json'), 'w') as json_file:
                    json.dump(model.reft_pos_configs, json_file, indent=4)
            print(f"The reft parameters of language model has been successfully saved in {reft_lm_output_dir}.")
        else:
            lm = model.language_model
        language_model_state_dict = get_peft_state_non_lora_maybe_zero_3(lm.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            language_model_output_dir = os.path.join(self.training_arguments.output_dir, 'language_model')
            os.makedirs(language_model_output_dir, exist_ok=True)
            language_model_output_path = os.path.join(self.training_arguments.output_dir, 'language_model/pytorch_model.bin')
            torch.save(language_model_state_dict, language_model_output_path)
            model.config.text_config.save_pretrained(language_model_output_dir, from_pt=True)
            print(f"The language model has been successfully saved in {language_model_output_dir}.")

        #save vision tower base params
        if isinstance(model.vision_tower._vision_tower, ReftModelShell):
            vt = model.vision_tower._vision_tower.model
            reft_vt_output_dir = os.path.join(os.path.join(self.training_arguments.output_dir, 'reft'), 'vision_tower')
            model.vision_tower._vision_tower.save(reft_vt_output_dir)   # save reft params
            with open(os.path.join(reft_vt_output_dir, 'intervention_config.json'), 'w') as json_file:
                json.dump(self.reft_config['vision_tower'], json_file, indent=4)
            print(f"The reft parameters of vision tower has been successfully saved in {reft_lm_output_dir}.")
        else:
            vt = model.vision_tower._vision_tower
        vision_tower_state_dict = get_peft_state_non_lora_maybe_zero_3(vt.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision_tower_output_dir = os.path.join(self.training_arguments.output_dir, 'vision_tower')
            os.makedirs(vision_tower_output_dir, exist_ok=True)
            vision_tower_output_path = os.path.join(self.training_arguments.output_dir, 'vision_tower/pytorch_model.bin')
            torch.save(vision_tower_state_dict, vision_tower_output_path)
            model.config.vision_config.save_pretrained(vision_tower_output_dir, from_pt=True)
            print(f"The vision tower has been successfully saved in {vision_tower_output_dir}.")

        #save connector base params
        connector_state_dict = get_peft_state_non_lora_maybe_zero_3(model.connector.named_parameters(),  False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            connector_output_dir = os.path.join(self.training_arguments.output_dir, 'connector')
            os.makedirs(connector_output_dir, exist_ok=True)
            connector_output_path = os.path.join(self.training_arguments.output_dir, 'connector/pytorch_model.bin')
            torch.save(connector_state_dict, connector_output_path)
            print(f"The connector has been successfully saved in {connector_output_dir}.")

        # save lora params
        lora_state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), self.training_arguments.lora_bias
        )
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            model.save_pretrained(self.training_arguments.output_dir, state_dict=lora_state_dict)
        
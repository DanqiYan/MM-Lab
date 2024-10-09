from dataclasses import dataclass
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
import json, re, copy

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from pyreft import ReftModel

from . import LLMFactory, ConnectorFactory, VisionTowerFactory
from .reft_model_shell import *
from .llm import full_name_to_LLM_register_name
from .vision_tower import VisionTowerFactory, full_name_to_VT_register_name
from .configuration_tinyllava import TinyLlavaConfig
from ..utils.constants import *
from ..utils import log, init_reft_pos_configs

# from tinyllava.utils.data_utils import get_value_from_kwargs

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None
    


class TinyLlavaPreTrainedModel(PreTrainedModel):
    config_class = TinyLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa


class TinyLlavaForConditionalGeneration(TinyLlavaPreTrainedModel):
    def __init__(self, config: TinyLlavaConfig):
        
        super().__init__(config)

        self.language_model = LLMFactory(config.llm_model_name_or_path)[0](config.text_config)
        self.vision_tower = VisionTowerFactory(config.vision_model_name_or_path)(config.vision_config)
        self.connector = ConnectorFactory(config.connector_type)(config)

        (Tokenizer, post_load) = LLMFactory(config.llm_model_name_or_path)[1]
        self.tokenizer = post_load(Tokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir = config.cache_dir,
            model_max_length = config.tokenizer_model_max_length,
            padding_side = config.tokenizer_padding_side,
            use_fast = config.tokenizer_use_fast,
        ))
        self.post_init()
        # set device for interventions before forward()
        self.already_set_device_for_interventions = False
    
    def _set_device_for_interventions(self):
        if isinstance(self.language_model, ReftModel):
            self.language_model.set_device(self.device)
        if isinstance(self.vision_tower.vision_tower, ReftModel):
            self.vision_tower.vision_tower.set_device(self.device)
        self.already_set_device_for_interventions = True
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    
    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     images: Optional[torch.FloatTensor] = None,
    #     image_sizes: Optional[List[List[int]]] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
    #     if not self.already_set_device_for_interventions:
    #         self._set_device_for_interventions()
    #     use_cache = use_cache if use_cache is not None else self.config.use_cache
    #     if inputs_embeds is None:
    #         (
    #             input_ids,
    #             position_ids,
    #             attention_mask,
    #             past_key_values,
    #             inputs_embeds,
    #             labels
    #         ) = self.prepare_inputs_labels_for_multimodal(
    #             input_ids,
    #             position_ids,
    #             attention_mask,
    #             past_key_values,
    #             labels,
    #             images,
    #             image_sizes
    #         )
        # kwargs = {}
        # kwargs['inputs_ids'] = input_ids
        # kwargs['attention_mask'] = attention_mask
        # kwargs['position_ids'] = position_ids
        # kwargs['past_key_values'] = past_key_values
        # kwargs['inputs_embeds'] = inputs_embeds
        # kwargs['labels'] = labels
        # kwargs['use_cache'] = use_cache
        # kwargs['output_attentions'] = output_attentions
        # kwargs['output_hidden_states'] = output_hidden_states
        # kwargs['return_dict'] = return_dict

        # return self.language_model.forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )

        # for reft method, this is the entry to ReftModelShell
        # return self.language_model.forward(input_ids, **kwargs)

    def forward(
            self,
            **kwargs
        ) -> Union[Tuple, CausalLMOutputWithPast]:

        input_ids = kwargs.get('input_ids', None)
        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)
        past_key_values = kwargs.get('past_key_values', None)
        inputs_embeds = kwargs.get('inputs_embeds', None)
        labels = kwargs.get('labels', None)
        use_cache = kwargs.get('use_cache', None)
        output_attentions = kwargs.get('output_attentions', None)
        output_hidden_states = kwargs.get('output_hidden_states', None)
        images = kwargs.get('images', None)
        image_sizes = kwargs.get('image_sizes', None)
        return_dict = kwargs.get('return_dict', None)

        if not self.already_set_device_for_interventions:
            self._set_device_for_interventions()
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        if isinstance(self.language_model, ReftModelShell):
            kwargs['input_ids'] = input_ids
            kwargs['position_ids'] = position_ids
            kwargs['attention_mask'] = attention_mask
            kwargs['past_key_values'] = past_key_values
            kwargs['inputs_embeds'] = inputs_embeds
            kwargs['labels'] = labels
            return self.language_model.forward(**kwargs)
        else:
            return self.language_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(inputs)


        if isinstance(self.language_model, ReftModelShell):
            kwargs['position_ids'] = position_ids
            kwargs['attention_mask'] = attention_mask
            kwargs['inputs_embeds'] = inputs_embeds
            return self.language_model.generate(**kwargs)[1]
        else:
            kwargs.pop("intervention_locations") if "intervention_locations" in kwargs else None
            return self.language_model.generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )


        
    def encode_images(self, images):
        kwargs = {}
        kwargs['vision_feature_layer'] = self.config.vision_feature_layer
        kwargs['vision_feature_select_strategy'] = self.config.vision_feature_select_strategy
        images = images.to(device=self.device, dtype=self.dtype)
        image_features = self.vision_tower(images, **kwargs)
        image_features = self.connector(image_features)
        return image_features
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
        
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        
        image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    

    
    
    def load_llm(self, **kwargs):
        language_model_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        pretrained_llm_path = get_value_from_kwargs(kwargs, 'pretrained_llm_path')
        if pretrained_llm_path is not None:
            language_model_name = pretrained_llm_path
        if language_model_name is not None:
            self.language_model = self.language_model.from_pretrained(
                language_model_name, **kwargs
            )
        print('loading language model from ', language_model_name)
        self.language_model.requires_grad_(False)
        
        self.config.text_config.torch_dtype = kwargs.get('torch_dtype', None)
        self.config.pad_token = getattr(self.tokenizer, 'pad_token', None)
        self.config.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        #self.config.tokenizer_padding_side = getattr(self.tokenizer, 'padding_side', None)
        #self.config.tokenizer_model_max_length =  getattr(self.tokenizer, 'model_max_length', None)
        
        
    def load_vision_tower(self, **kwargs):
        vision_tower_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        self.vision_tower.load_model(vision_tower_name, **kwargs)

        
    def load_connector(self, **kwargs):
        self.connector.load_model(**kwargs)

    def parse_reft_config(self, reft_config, model):
        for model_type, config_ in reft_config.items():
            for k, v in config_.items():
                if k == "representations":
                    extend_repres = []
                    for i, repre in enumerate(v):
                        if repre['layer'] == "all":
                            if model_type == "llm":
                                num_blocks = len(get_model_attribute(
                                    model.language_model,
                                    model_name_to_module_path_mapping[model.config.llm_model_name_or_path]['layers']
                                    ))
                            elif model_type == "vision_tower":
                                num_blocks = len(get_model_attribute(
                                    model.vision_tower.vision_tower,
                                    model_name_to_module_path_mapping[model.config.vision_model_name_or_path]['layers']
                                    ))
                            parsed_nums = [i for i in range(num_blocks)]
                        elif re.match(r'^\d+$', repre['layer']):
                            parsed_nums = [int(repre['layer'])]
                        elif re.match(r'^(\d+;)*\d+;?$', repre['layer']):
                            parsed_nums = [int(x) for x in repre['layer'].rstrip(';').split(';')]
                        elif re.match(r'^(\d+,)*\d+,?$', repre['layer']):
                            parsed_nums = [int(x) for x in repre['layer'].rstrip(',').split(',')]
                        else:
                            raise ValueError("argument 'layer' must be 'all' or a single number or in the format '2;10;18;26' or '2,10,18,26'!")
                        v.pop(i)
                        extend_repres.extend([
                            copy.deepcopy({**repre, 'layer': i}) for i in parsed_nums
                        ])
                    v.extend(extend_repres)
        return reft_config

    def add_reft_adaptor_from_training_arguments(self, training_arguments):
        if not training_arguments.loreft_config_path and not hasattr(self, 'reft_config'):
            log("No reft_config is provided, no reft adaptors will be added!")
        # load reft_config and add reft modules to model
        if training_arguments.loreft_config_path and not hasattr(self, 'reft_config'):
            self.generate_reft_config(training_arguments.loreft_config_path)

        if not hasattr(self, 'reft_pos_configs'):
            self.generate_reft_pos_configs(self.reft_config, self, training_arguments) if training_arguments.reft_pos_configs == None else None
        
        self.add_reft_adaptor()

    def load_reft_adaptor_from_ckpt(self, pretrained_model_path: str):
        self._extend_type_to_module_mapping()
        reft_save_path = os.path.join(pretrained_model_path, 'reft')
        if "language_model" in os.listdir(reft_save_path):
            lm_subpath = os.path.join(reft_save_path, 'language_model')
            self.language_model = ReftModelShell.load(
                lm_subpath,
                self.language_model,
                model_register_name=full_name_to_LLM_register_name(self.config.llm_model_name_or_path),
                model_task_type='llm'
            )
            if "reft_pos_configs.json" in os.listdir(lm_subpath):
                with open(os.path.join(lm_subpath, "reft_pos_configs.json"), 'r') as file:
                    data = json.load(file)
                    self.reft_pos_configs = data
            # for possibly saving reft_config again later
            if "intervention_config.json" in os.listdir(lm_subpath):
                with open(os.path.join(lm_subpath, "intervention_config.json"), 'r') as file:
                    data = json.load(file)
                    self.reft_config = {'llm': data}


    def add_reft_adaptor(self):
        self._extend_type_to_module_mapping()
        reft_config = copy.deepcopy(self.reft_config)
        # llm reft adaptor
        if 'llm' in reft_config.keys():
            log("Adding LoReFT adapters to llm...")
            loreft_config_llm = ReftConfigGenerator(**reft_config['llm'])
            # model.language_model = pyreft.get_reft_model(model.language_model, loreft_config_llm)
            self.language_model = get_reft_shell(self.language_model,
                                                loreft_config_llm,
                                                full_name_to_LLM_register_name(self.config.llm_model_name_or_path),
                                                'llm')
            print("trainable params for language model:")
            self.language_model.print_trainable_parameters()
        # vision tower reft adaptor
        # if 'vision_tower' in reft_config.keys():
        #     log("Adding LoReFT adapters to vision tower...")
        #     loreft_config_vt = ReftConfigGenerator(**reft_config['vision_tower'])
        #     # TODO: vision_tower_2 is not considered.
        #     self.vision_tower._vision_tower = get_reft_shell(self.vision_tower._vision_tower,
        #                                                     loreft_config_vt,
        #                                                     full_name_to_VT_register_name(self.config.vision_model_name_or_path),
        #                                                     'vision_tower')
        #     print("trainable params for vision tower:")
        #     self.vision_tower._vision_tower.print_trainable_parameters()


    def generate_reft_pos_configs(self, reft_config, model, training_arguments):
        if model.config.vision_feature_select_strategy == 'patch':
            img_embed_token_len = (model.vision_tower.config.image_size // model.vision_tower.config.patch_size) ** 2 - 1
        elif model.config.vision_feature_select_strategy == 'cls_patch':
            img_embed_token_len = (model.vision_tower.config.image_size // model.vision_tower.config.patch_size) ** 2
        else:
            raise ValueError(f"Unexpected select feature: {model.config.vision_feature_select_strategy}")
        configs2upate = {
            'num_interventions': {model_type: len(config_['representations']) for model_type, config_ in reft_config.items() if 'representations' in config_},
            'img_embed_token_len': img_embed_token_len
            }
        if training_arguments.intervention_positions is not None:
            configs2upate['intervention_positions'] = training_arguments.intervention_positions
        if training_arguments.reft_share_weights is not None:
            configs2upate['reft_share_weights'] = training_arguments.reft_share_weights
        if training_arguments.intervened_prompt_part is not None:
            configs2upate['intervened_prompt_part'] = training_arguments.intervened_prompt_part
        if training_arguments.intervene_include_img_embed is not None:
            configs2upate['intervene_include_img_embed'] = training_arguments.intervene_include_img_embed

        reft_pos_configs = init_reft_pos_configs()
        reft_pos_configs.update(configs2upate)
        training_arguments.reft_pos_configs = reft_pos_configs
        self.reft_pos_configs = reft_pos_configs

    def generate_reft_config(self, loreft_config_path):
        with open(loreft_config_path, 'r') as file:
            self.reft_config = json.load(file)
        self.parse_reft_config(self.reft_config, self)

    def _extend_type_to_module_mapping(self):
        if self.config.llm_model_name_or_path in EXTENSION_type_to_module_mapping.keys():
            pyvene.models.intervenable_modelcard.type_to_module_mapping.update({
                self.language_model.__class__: EXTENSION_type_to_module_mapping[self.config.llm_model_name_or_path]
            })

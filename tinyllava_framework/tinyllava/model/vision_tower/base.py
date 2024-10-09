import os
from typing import Tuple

import torch
import torch.nn as nn

from transformers import PreTrainedModel
from pyvene.models.intervenable_base import IntervenableModelOutput
# from tinyllava.utils.data_utils import get_value_from_kwargs

def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None

class VisionTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._vision_tower = None
        self._image_processor = None
        self.config = cfg
    

    def load_model(self, vision_tower_name, **kwargs):
        self._load_model(vision_tower_name, **kwargs)
        self._vision_tower.requires_grad_(False)



        
    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = get_value_from_kwargs(kwargs, 'pretrained_vision_tower_path')
        if isinstance(self._vision_tower, PreTrainedModel): # hf model
            if pretrained_vision_tower_path is not None:
                vision_tower_name = pretrained_vision_tower_path
            self._vision_tower = self._vision_tower.from_pretrained(vision_tower_name, **kwargs)      
        else: # nn.Module
            if pretrained_vision_tower_path is not None:
                vision_tower_weights = torch.load(os.path.join(pretrained_vision_tower_path, 'pytorch_model.bin'), map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self._vision_tower.load_state_dict(vision_tower_weights)

        print("Loading vision tower from ", vision_tower_name)
        


    def forward(self, x, **kwargs):
        # for reft method, this is the entry to ReftModelShell.forward()
        image_features = self._vision_tower(x, output_hidden_states=True)
        # TODO: is this the best way? 
        # current solution: fit on the imple of ReftModelShell.forward()
        if isinstance(image_features, IntervenableModelOutput):
            image_features = image_features.intervened_outputs
        elif isinstance(image_features, Tuple):
            if image_features[1] is None:   # no counterfactual output, just return base output
                if isinstance(image_features[0], Tuple):
                    image_features = image_features[0][0]
                else:
                    image_features = image_features[0]
            else:
                image_features = image_features[1]
        
        # original output without ReFT
        image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]

        if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            image_features = image_features[:, 1:]   # 第一个特征（通常是 cls token)
        elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")

        return image_features
        

    
    @property
    def vision_tower(self):
        return self._vision_tower
        
    @vision_tower.setter
    def vision_tower(self, vision_tower):
        self._vision_tower = vision_tower
        
    

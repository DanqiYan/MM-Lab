from .intervention_models import (
    LoreftIntervention_,
    NoreftIntervention_
)


INTERVENTION_TYPE_MAPPING = {
    # 'loreft': pyreft.LoreftIntervention
    'loreft': LoreftIntervention_,
    'noreft': NoreftIntervention_
}


EXTENSION_type_to_module_mapping = {
    'microsoft/phi-2': {
        'block_input': ('model.layers[%s]', 'register_forward_pre_hook'),
        'block_output': ('model.layers[%s]', 'register_forward_hook'),
        'mlp_activation': ('model.layers[%s].mlp.activation_fn', 'register_forward_hook'),
        'mlp_output': ('model.layers[%s].mlp', 'register_forward_hook'),
        'mlp_input': ('model.layers[%s].mlp', 'register_forward_pre_hook'),
        'attention_output': ('model.layers[%s].self_attn', 'register_forward_hook'),
        'attention_input': ('model.layers[%s].self_attn', 'register_forward_pre_hook'),
        'query_output': ('model.layers[%s].self_attn.q_proj', 'register_forward_hook'),
        'key_output': ('model.layers[%s].self_attn.k_proj', 'register_forward_hook'),
        'value_output': ('model.layers[%s].self_attn.v_proj', 'register_forward_hook'),
        'head_query_output': ('model.layers[%s].self_attn.q_proj', 'register_forward_hook'),
        'head_key_output': ('model.layers[%s].self_attn.k_proj', 'register_forward_hook'),
        'head_value_output': ('model.layers[%s].self_attn.v_proj', 'register_forward_hook')
    }
}

model_name_to_module_path_mapping = {
    'microsoft/phi-2': {
        "embed_tokens": "model.embed_tokens",
        "layers": "model.layers",
        "lm_head": "lm_head"
    }
}
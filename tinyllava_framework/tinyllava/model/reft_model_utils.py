
def adapt_language_model_args(_, **kwargs):
    input_ids = kwargs.get('input_ids', None)
    attention_mask = kwargs.get('attention_mask', None)
    position_ids = kwargs.get('position_ids', None)
    past_key_values = kwargs.get('past_key_values', None)
    inputs_embeds = kwargs.get('inputs_embeds', None)
    labels = kwargs.get('labels', None)
    use_cache = kwargs.get('use_cache', None)
    output_attentions = kwargs.get('output_attentions', None)
    output_hidden_states = kwargs.get('output_hidden_states', None)
    return_dict = kwargs.get('return_dict', None)
    cache_position = kwargs.get('cache_position', None)

    return _, {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'past_key_values': past_key_values,
        'inputs_embeds': inputs_embeds,
        'labels': labels,
        'use_cache': use_cache,
        'output_attentions': output_attentions,
        'output_hidden_states': output_hidden_states,
        'return_dict': return_dict,
        'cache_position': cache_position
    }


def adapt_language_model_args_without_cache_pos(_, **kwargs):
    input_ids = kwargs.get('input_ids', None)
    attention_mask = kwargs.get('attention_mask', None)
    position_ids = kwargs.get('position_ids', None)
    past_key_values = kwargs.get('past_key_values', None)
    inputs_embeds = kwargs.get('inputs_embeds', None)
    labels = kwargs.get('labels', None)
    use_cache = kwargs.get('use_cache', True)
    output_attentions = kwargs.get('output_attentions', False)
    output_hidden_states = kwargs.get('output_hidden_states', False)
    return_dict = kwargs.get('return_dict', True)

    return _, {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'past_key_values': past_key_values,
        'inputs_embeds': inputs_embeds,
        'labels': labels,
        'use_cache': use_cache,
        'output_attentions': output_attentions,
        'output_hidden_states': output_hidden_states,
        'return_dict': return_dict
    }



def adapt_vt_normal_args(images, **kwargs):
    # adapt args for the vision_models without positional arguments
    return images, kwargs


def adapt_vt_siglip_args(images, **kwargs):
    # adapt args for siglip, clip
    output_attentions = kwargs.get('output_attentions', None)
    output_hidden_states = kwargs.get('output_hidden_states', None)
    return_dict = kwargs.get('return_dict', None)
    return images, {
        'output_attentions': output_attentions,
        'output_hidden_states': output_hidden_states,
        'return_dict': return_dict,
    }


def adapt_vt_dinov2_args(images, **kwargs):
    # adapt args for dinov2
    bool_masked_pos = kwargs.get('bool_masked_pos', None)
    head_mask = kwargs.get('head_mask', None)
    output_attentions = kwargs.get('output_attentions', None)
    output_hidden_states = kwargs.get('output_hidden_states', None)
    return_dict = kwargs.get('return_dict', None)
    return images, {
        'bool_masked_pos': bool_masked_pos,
        'head_mask': head_mask,
        'output_attentions': output_attentions,
        'output_hidden_states': output_hidden_states,
        'return_dict': return_dict,
    }

MODEL_TYPE_TO_ARGS_ADAPT_MAPPING = {
    'gemma': adapt_language_model_args_without_cache_pos,
    'openelm': adapt_language_model_args_without_cache_pos,
    'phi': adapt_language_model_args_without_cache_pos,
    'qwen': adapt_language_model_args,
    'stablelm': adapt_language_model_args,
    'tinyllama': adapt_language_model_args_without_cache_pos,
    'clip': adapt_vt_siglip_args,
    'dinov2': adapt_vt_dinov2_args,
    'mof': adapt_vt_normal_args,
    'siglip': adapt_vt_siglip_args,
}


def adapt_llm_generate_args(_, **kwargs):
    attention_mask = kwargs.get('attention_mask', None)
    position_ids = kwargs.get('position_ids', None)
    inputs_embeds = kwargs.get('inputs_embeds', None)

    filtered_args = {
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'inputs_embeds': inputs_embeds
    }
    
    generate_rel_args = [
        'pad_token_id',
        'do_sample',
        'temperature',
        'top_p',
        'num_beams',
        'max_new_tokens',
        # maybe not necessary
        'image_size',
        'use_cache'
    ]
    
    for k, v in kwargs.items():
        if k in generate_rel_args:
            filtered_args[k] = v

    return _, filtered_args

MODEL_TYPE_TO_ARGS_ADAPT_MAPPING_FOR_GENERATE = {
    'phi': adapt_llm_generate_args,
}

class ReftModelProxy:
    # This is used in ReftModel to filter args the model needs
    def __init__(self, model_register_name):
        # self.model = model
        self.model_register_name = model_register_name

    def get_adapted_args(self, *args, **kwargs):
        adapted_args = MODEL_TYPE_TO_ARGS_ADAPT_MAPPING[self.model_register_name](*args, **kwargs)
        return adapted_args
    
class ReftModelProxy4Generate:
    # This is used in ReftModel to filter args the model needs when calling ReftModelShell.generate()
    def __init__(self, model_register_name):
        # self.model = model
        self.model_register_name = model_register_name

    def get_adapted_args(self, *args, **kwargs):
        adapted_args = MODEL_TYPE_TO_ARGS_ADAPT_MAPPING_FOR_GENERATE[self.model_register_name](*args, **kwargs)
        return adapted_args
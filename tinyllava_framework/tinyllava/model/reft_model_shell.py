import os, json, copy

from collections import OrderedDict
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

import torch
import pyreft
import pyvene

from pyreft.reft_model import ReftModel
from pyvene.models.interventions import (
    TrainableIntervention,
    CollectIntervention,
    ZeroIntervention,
    SourcelessIntervention
)
from pyvene.models.basic_utils import get_batch_size, get_type_from_string, create_directory
from pyvene.models.intervenable_base import IntervenableModelOutput, IntervenableModel
from pyvene.models.configuration_intervenable_model import (
    IntervenableConfig,
    RepresentationConfig,
)
from pyvene.models.interventions import (
    TrainableIntervention,
    CollectIntervention
)

from ..utils.train_utils import *
from .reft_model_utils import ReftModelProxy, ReftModelProxy4Generate
from .reft_mappings import *



def get_model_attribute(model, path):
    attributes = path.split('.')
    for attr in attributes:
        model = getattr(model, attr)
    return model


def get_reft_shell(model, reft_config, model_register_name, model_task_type, set_device=True, disable_model_grads=True):
    """
    Create an instance of ReFT model.
    """
    assert not isinstance(model, ReftModelShell), \
        "The model has been already integrated with ReFT method. The language_model/vision_tower can only be added with ReFT once!"
    # reft_model = ReftModelShell(reft_config, model, ReftModelProxy(model_register_name), model_task_type)
    reft_model = ReftModelShell(reft_config, model, model_register_name, model_task_type)
    if set_device:
        reft_model.set_device(model.device)
    if disable_model_grads:
        reft_model.disable_model_gradients()    
    return reft_model

class ReftModelShell(ReftModel):
    """
    Shell for ReftModel, mainly forward() modified
    """
    def __init__(self, config, model, model_register_name, model_task_type, **kwargs):
        self.reftModelProxy = ReftModelProxy(model_register_name)
        self.reftModelProxy4Generate = ReftModelProxy4Generate(model_register_name)
        self.model_task_type = model_task_type
        super().__init__(config, model, **kwargs)

    @staticmethod
    def load_intervenable_model(load_directory, model, local_directory=None, from_huggingface_hub=False):
        """
        modified from load() of IntervenableModel
        """
        if not os.path.exists(load_directory) or from_huggingface_hub:
            from_huggingface_hub = True
            
            from huggingface_hub import snapshot_download
            load_directory = snapshot_download(
                repo_id=load_directory,
                local_dir=local_directory,
            )

        # load config
        saving_config = IntervenableConfig.from_pretrained(load_directory)
        casted_intervention_types = []

        for type_str in saving_config.intervention_types:
            casted_intervention_types += [get_type_from_string(type_str)]
        saving_config.intervention_types = (
            casted_intervention_types
        )

        with open(os.path.join(load_directory, 'intervention_config.json'), 'r') as f:
            intervention_config = json.load(f)
        new_repres = []
        if "representations" in intervention_config:
            for reft_args in intervention_config['representations']:
                if "intervention" in reft_args:
                    inter_type = reft_args["intervention"]["type"]
                    del reft_args["intervention"]["type"]
                    assert inter_type in INTERVENTION_TYPE_MAPPING, f'The intervention type doesn\'t exist, please use the following intervention types: [{", ".join(INTERVENTION_TYPE_MAPPING.keys())}]'
                    reft_args["intervention"] = INTERVENTION_TYPE_MAPPING[inter_type](**reft_args["intervention"])
                new_repres.append(reft_args)
            # for i, inter in enumerate(new_repres):
            #     saving_config.representations[i].intervention = inter['intervention']
        casted_representations = []
        for (
            i, representation_opts
        ) in enumerate(saving_config.representations):
            l = list(representation_opts)
            l[6] = new_repres[i]['intervention']
            casted_representations += [
                RepresentationConfig(*l)
            ]
        saving_config.representations = casted_representations
        intervenable = IntervenableModel(saving_config, model)

        # load binary files
        for i, (k, v) in enumerate(intervenable.interventions.items()):
            intervention = v[0]
            binary_filename = f"intkey_{k}.bin"
            intervention.is_source_constant = \
                saving_config.intervention_constant_sources[i]
            intervention.set_interchange_dim(saving_config.intervention_dimensions[i])
            if saving_config.intervention_constant_sources[i] and \
                not isinstance(intervention, ZeroIntervention) and \
                not isinstance(intervention, SourcelessIntervention):
                # logging.warn(f"Loading trainable intervention from {binary_filename}.")
                saved_state_dict = torch.load(os.path.join(load_directory, binary_filename))
                try:
                    intervention.register_buffer(
                        'source_representation', saved_state_dict['source_representation']
                    )
                except:
                    intervention.source_representation = saved_state_dict['source_representation']
            elif isinstance(intervention, TrainableIntervention):
                saved_state_dict = torch.load(os.path.join(load_directory, binary_filename))
                intervention.load_state_dict(saved_state_dict)

        return intervenable
    
    @staticmethod
    def load_reftmodel(*args, **kwargs):
        model = ReftModelShell.load_intervenable_model(*args, **kwargs)
        return ReftModel._convert_to_reft_model(model)

    @staticmethod
    def load(*args, **kwargs):
        model_register_name = kwargs.pop('model_register_name')
        model_task_type = kwargs.pop('model_task_type')
        model = ReftModelShell.load_reftmodel(*args, **kwargs)
        return ReftModelShell._convert_to_reft_model_shell(model,
                                                           model_register_name,
                                                           model_task_type)

    @staticmethod
    def _convert_to_reft_model_shell(reft_model,
                                     model_register_name,
                                     model_task_type):
        reft_model_shell = ReftModelShell(reft_model.config, reft_model.model, model_register_name, model_task_type)
        # Copy any other necessary attributes
        for attr in vars(reft_model):
            setattr(reft_model, attr, getattr(reft_model, attr))
        return reft_model_shell

    def forward(
        self,
        *args,
        **kwargs
    ):
        """
        For detailed documentation, please check the forward() of pv.IntervenableModel
        """
        assert len(args) in [0, 1], "To much positional arguments for ReftModelShell.forward()!"
        if len(args) == 1:   # for vision_tower
            base_ = args[0]
        elif len(args) == 0:
            base_ = kwargs['inputs_embeds']
        sources_ = kwargs.get('sources_', None)
        unit_locations_ = kwargs.get('unit_locations_', None)
        source_representations_ = kwargs.get('source_representations_', None)
        subspaces_ = kwargs.get('subspaces_', None)
        labels_ = kwargs.get('labels_', None)
        output_original_output_ = kwargs.get('output_original_output_', False)
        return_dict_ = kwargs.get('return_dict_', None)
        use_cache_ = kwargs.get('use_cache_', True)
        # kwargs['input_ids'] = base   # for following language model input

        activations_sources = source_representations_
        if sources_ is not None and not isinstance(sources_, list):
            sources_ = [sources_]
        
        self._cleanup_states()

        # if no source input or intervention, we return base
        if sources_ is None and activations_sources is None \
            and unit_locations_ is None and len(self.interventions) == 0:
            if self.model_task_type == 'vision_tower':
                return self.model(self.reftModelProxy.get_adapted_args(base_, kwargs)[0],
                              **self.reftModelProxy.get_adapted_args(base_, kwargs)[1]), None
            elif self.model_task_type == 'llm':
                return self.model(**self.reftModelProxy.get_adapted_args(base_, kwargs)[1]), None
            else:
                raise ValueError('ReftModel only applies to llm or vision_tower')
            
            
        # broadcast
        unit_locations = self._broadcast_unit_locations(get_batch_size(base_), unit_locations_)
        sources = [None]*len(self._intervention_group) if sources_ is None else sources_
        sources = self._broadcast_sources(sources)
        activations_sources = self._broadcast_source_representations(activations_sources)
        subspaces = self._broadcast_subspaces(get_batch_size(base_), subspaces_)
        
        self._input_validation(
            base_,
            sources,
            unit_locations,
            activations_sources,
            subspaces,
        )
        
        base_outputs = None
        if output_original_output_:
            # returning un-intervened output with gradients
            if self.model_task_type == 'vision_tower':
                base_outputs = self.model(self.reftModelProxy.get_adapted_args(base_, kwargs)[0],
                                        **self.reftModelProxy.get_adapted_args(base_, kwargs)[1])
            elif self.model_task_type == 'llm':
                base_outputs = self.model(**self.reftModelProxy.get_adapted_args(base_, kwargs)[1])
            else:
                raise ValueError('ReftModel only applies to llm or vision_tower')
        try:
            # intervene
            if self.mode == "parallel":   # 每个干预都是独立的 可以在同一批次中不同的样本上同时应用
                set_handlers_to_remove = (
                    self._wait_for_forward_with_parallel_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )
            elif self.mode == "serial":
                set_handlers_to_remove = (
                    self._wait_for_forward_with_serial_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )

            # run intervened forward
            model_kwargs = {}
            if labels_ is not None: # for training
                model_kwargs["labels_"] = labels_
            if 'use_cache_' in self.model.config.to_dict(): # for transformer models
                model_kwargs["use_cache_"] = use_cache_

            # counterfactual_outputs = self.model(**base_, **model_kwargs)
            if self.model_task_type == 'vision_tower':
                counterfactual_outputs = self.model(self.reftModelProxy.get_adapted_args(base_, **kwargs)[0],
                                                    **self.reftModelProxy.get_adapted_args(base_, **kwargs)[1],
                                                    **model_kwargs)
            elif self.model_task_type == 'llm':
                counterfactual_outputs = self.model(**self.reftModelProxy.get_adapted_args(base_, **kwargs)[1],
                                                    **model_kwargs)
            else:
                raise ValueError('ReftModel only applies to llm or vision_tower')

            set_handlers_to_remove.remove()

            self._output_validation()
            
            collected_activations = []
            if self.return_collect_activations:
                for key in self.sorted_keys:
                    if isinstance(
                        self.interventions[key][0],
                        CollectIntervention
                    ):
                        collected_activations += self.activations[key]

        except Exception as e:
            raise e
        finally:
            self._cleanup_states(
                skip_activation_gc = \
                    (sources is None and activations_sources is not None) or \
                    self.return_collect_activations
            )
        
        return counterfactual_outputs
    
    def generate(
        self,
        *args,
        **kwargs
    ):
        """
        override the function of IntervenableModel.generate()
        """
        assert len(args) in [0, 1], "To much positional arguments for ReftModelShell.forward()!"
        if len(args) == 1:   # for vision_tower
            base_ = args[0]
        elif len(args) == 0:
            base_ = kwargs['inputs_embeds']
        kwargs = self._prepare_inputs(kwargs)
        sources_ = kwargs.get('sources_', None)
        unit_locations_ = kwargs.get('unit_locations_', None)
        source_representations_ = kwargs.get('source_representations_', None)
        subspaces_ = kwargs.get('subspaces_', None)
        intervene_on_prompt_ = kwargs.get('intervene_on_prompt_', False)
        output_original_output_ = kwargs.get('output_original_output_', False)

        # TODO: forgive me now, i will change this later.
        activations_sources = source_representations_
        if sources_ is not None and not isinstance(sources_, list):
            sources_ = [sources_]
            
        self._cleanup_states()

        self._intervene_on_prompt = intervene_on_prompt_
        self._is_generation = True
        
        if not intervene_on_prompt_ and unit_locations_ is None:
            # that means, we intervene on every generated tokens!
            unit_locations_ = {"base": 0}

        # broadcast
        unit_locations = self._broadcast_unit_locations(get_batch_size(base_), unit_locations_)
        sources = [None]*len(self._intervention_group) if sources_ is None else sources_
        sources = self._broadcast_sources(sources)
        activations_sources = self._broadcast_source_representations(activations_sources)
        subspaces = self._broadcast_subspaces(get_batch_size(base_), subspaces_)

        self._input_validation(
            base_,
            sources,
            unit_locations,
            activations_sources,
            subspaces,
        )
        
        base_outputs = None
        if output_original_output_:
            if self.model_task_type == 'llm':
                base_outputs = self.model.generate(**self.reftModelProxy4Generate.get_adapted_args(base_, kwargs)[1])
            else:
                raise ValueError('ReftModel only applies to llm or vision_tower')

        set_handlers_to_remove = None
        try:
            # intervene
            if self.mode == "parallel":
                set_handlers_to_remove = (
                    self._wait_for_forward_with_parallel_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )
            elif self.mode == "serial":
                set_handlers_to_remove = (
                    self._wait_for_forward_with_serial_intervention(
                        sources,
                        unit_locations,
                        activations_sources,
                        subspaces,
                    )
                )

            if self.model_task_type == 'llm':
                counterfactual_outputs = self.model.generate(**self.reftModelProxy4Generate.get_adapted_args(base_, **kwargs)[1])
            else:
                raise ValueError('ReftModel only applies to llm')

            # run intervened generate
            # counterfactual_outputs = self.model.generate(
            #     **base, **kwargs
            # )

            collected_activations = []
            if self.return_collect_activations:
                for key in self.sorted_keys:
                    if isinstance(
                        self.interventions[key][0],
                        CollectIntervention
                    ):
                        collected_activations += self.activations[key]
        except Exception as e:
            raise e
        finally:
            if set_handlers_to_remove is not None:
                set_handlers_to_remove.remove()
            self._is_generation = False
            self._cleanup_states(
                skip_activation_gc = \
                    (sources is None and activations_sources is not None) or \
                    self.return_collect_activations
            )
        
        if self.return_collect_activations:
            return (base_outputs, collected_activations), counterfactual_outputs
        
        return base_outputs, counterfactual_outputs

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        During training, there's _prepare_inputs in the trainer where one can make adjustment for reft model, while during inference, there's no such api.
        Thus this function is added here for the function of generate().
        """
        if "intervention_locations" in inputs:
            inputs["unit_locations_"] = {"sources->base": (
                None,
                inputs["intervention_locations"].permute(1, 0, 2).tolist()
            )} 
        inputs["subspaces_"] = inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        inputs['intervene_on_prompt_'] = True

        return inputs

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        r"""rewrite method from nn.Module. Add also the intervention modules
        from ReftModel.

        Yields:
            (str, Module): Tuple containing a name and child module
        """
        memo = set()
        for name, module in self._modules.items():
            # if name == "parametrizations":   # avoid adding weigths of parametrization in LowRankRotateLayer
            #     continue
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module
        interventions = getattr(self, "interventions", None)
        if interventions is not None:
            for name, inter in interventions.items():
                # if name == "parametrizations":   # avoid adding weigths of parametrization in LowRankRotateLayer
                #     continue
                module = inter[0]
                if module is not None and module not in memo:
                    memo.add(module)
                    yield name, module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (str, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        """

        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                # if module is None or name == "parametrizations":
                #     continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m
            interventions = getattr(self, "interventions", None)
            if interventions is not None:
                for name, inter in interventions.items():
                    module = inter[0]
                    # if module is None or name == "parametrizations":
                    #     continue
                    submodule_prefix = prefix + ('.' if prefix else '') + name
                    for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                        yield m

    def parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            yield param
        for name, param in self.model.named_parameters(recurse=recurse):
            yield param
                
    def save_intervention_state_dict(self, intervention, save_path):
        state_dict = intervention.state_dict()
        for key, tensor in state_dict.items():
            if tensor.is_cuda:
                state_dict[key] = tensor.cpu()
        torch.save(state_dict, save_path)
    
    def save(
        self, save_directory, save_to_hf_hub=False, hf_repo_name="my-awesome-model"
    ):
        """
        Save interventions to disk or hub
        """
        # import torch.distributed as dist
        # def is_main_process():
        #     return not dist.is_initialized() or dist.get_rank() == 0
        # if not is_main_process():
        #     return
        if save_to_hf_hub:
            from huggingface_hub import HfApi

            api = HfApi()

        create_directory(save_directory)

        saving_config = copy.deepcopy(self.config)
        saving_config.sorted_keys = self.sorted_keys
        saving_config.model_type = str(
            saving_config.model_type
        )
        saving_config.intervention_types = []
        saving_config.intervention_dimensions = []
        saving_config.intervention_constant_sources = []
        
        # handle constant source reprs if passed in.
        serialized_representations = []
        for reprs in saving_config.representations:
            serialized_reprs = {}
            for k, v in reprs._asdict().items():
                if k == "hidden_source_representation":
                    continue
                if k == "source_representation":
                    # hidden flag only set here
                    if v is not None:
                        serialized_reprs["hidden_source_representation"] = True
                    serialized_reprs[k] = None
                elif k == "intervention_type":
                    serialized_reprs[k] = None
                elif k == "intervention":
                    serialized_reprs[k] = None
                else:
                    serialized_reprs[k] = v
            serialized_representations += [
                RepresentationConfig(**serialized_reprs)
            ]
        saving_config.representations = \
            serialized_representations
        
        for k, v in self.interventions.items():
            intervention = v[0]
            saving_config.intervention_types += [str(type(intervention))]
            binary_filename = f"intkey_{k}.bin"
            # save intervention binary file
            if isinstance(intervention, TrainableIntervention) or \
                intervention.source_representation is not None:
                # logging.info(f"Saving trainable intervention to {binary_filename}.")
                self.save_intervention_state_dict(intervention, os.path.join(save_directory, binary_filename))
                if save_to_hf_hub:
                    # push to huggingface hub
                    try:
                        api.create_repo(hf_repo_name)
                    except:
                        logging.info(
                            f"Uploading: {binary_filename}, but skipping creating the repo since "
                            f"either {hf_repo_name} exists or having authentication error."
                        )
                    api.upload_file(
                        path_or_fileobj=os.path.join(save_directory, binary_filename),
                        path_in_repo=binary_filename,
                        repo_id=hf_repo_name,
                        repo_type="model",
                    )
            if intervention.interchange_dim is None:
                saving_config.intervention_dimensions += [None]
            else:
                saving_config.intervention_dimensions += [intervention.interchange_dim.tolist()]
            saving_config.intervention_constant_sources += [intervention.is_source_constant]
            
        # save metadata config
        saving_config.save_pretrained(save_directory)
        if save_to_hf_hub:
            # push to huggingface hub
            try:
                api.create_repo(hf_repo_name)
            except:
                logging.info(
                    f"Uploading the config, Skipping creating the repo since "
                    f"either {hf_repo_name} exists or having authentication error."
                )
            api.upload_file(
                path_or_fileobj=os.path.join(save_directory, "config.json"),
                path_in_repo="config.json",
                repo_id=hf_repo_name,
                repo_type="model",
            )



class ReftConfigGenerator(pyreft.ReftConfig):
    """
    generate Reft config for Reft methods.
    """
    def __init__(
        self, **kwargs,
    ):
        if "representations" in kwargs:
            new_repres = []
            for reft_args in kwargs['representations']:
                if "intervention" in reft_args:
                    inter_type = reft_args["intervention"]["type"]
                    del reft_args["intervention"]["type"]
                    assert inter_type in INTERVENTION_TYPE_MAPPING, f'The intervention type doesn\'t exist, please use the following intervention types: [{", ".join(INTERVENTION_TYPE_MAPPING.keys())}]'
                    reft_args["intervention"] = INTERVENTION_TYPE_MAPPING[inter_type](**reft_args["intervention"])
                new_repres.append(reft_args)
            kwargs['representations'] = new_repres
        super().__init__(**kwargs)



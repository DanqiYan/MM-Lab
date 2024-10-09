import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import debugpy
from tqdm import tqdm
import shortuuid
from dataclasses import dataclass, field
from typing import Optional

from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data import *

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
from datasets import load_dataset



# prepare dataset for bpc calculation
class EvalDataset(Dataset):
    def __init__(self, args, task_name, block_size, stride, tokenizer, file_num=-1, dtype="auto", vocab_size=None):
        self.args = args
        self.task_name = task_name
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.file_num = file_num
        self.data = None
        self.stride = stride
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype

        self._prepare()
        self.prev_end_loc = 0
        self.seq_len = len(self.data)
        self.begin_loc = 0

    def _prepare(self):
        self._curr_idx = 0
        self._arr = []


        "从huggingface上下载数据"
        # split_str = 'test[:10]' if self.file_num == -1 else f"test[:{self.file_num // 10}]"
        # split_str = f"test[:{self.file_num}]"

        # self._raw_dataset = load_dataset(
        #     "hkust-nlp/llm-compression",
        #     self.task_name,
        #     split=split_str,
        #     cache_dir=self.args.cache_dir,
        # )

        "使用缓存的数据"
        local_file_path = "/home/atuin/b211dd/b211dd20/bpc/hkust-nlp___llm-compression/arxiv_math/arxiv_math.jsonl"  # Replace with your local path
        self._raw_dataset = load_dataset(
            'json',
            data_files=local_file_path,
            # split='test[:]' if self.file_num == -1 else f"test[:{self.file_num}]",
            split="train"
        )


        self.raw_dataset = self._raw_dataset.filter(lambda example: len(example['content']) > 0)
        self.character_num = 0
        for i in range(len(self.raw_dataset)):
            self.character_num += len(self.raw_dataset[i]['content'])

        self.data = self.raw_dataset.map(
            lambda example: {"encoding": np.array(self.tokenizer.encode(example['content']), dtype=self._dtype)}, num_proc=8)

        self.data = np.concatenate([a['encoding'] for a in self.data], axis=0)

    def __len__(self):
        return math.floor((len(self.data)-self.block_size)/self.stride+1)

    def __getitem__(self,item):
        end_loc = min(self.begin_loc+self.block_size, self.seq_len)
        trg_len = end_loc - self.prev_end_loc
        input_ids = self.data[self.begin_loc:end_loc]
        attention_mask = np.ones((len(input_ids),), dtype=bool)
        attention_mask[:-trg_len] = False
        self.prev_end_loc = end_loc
        self.begin_loc = self.begin_loc + self.stride
        return torch.tensor(input_ids), torch.tensor(attention_mask, dtype=bool)


class Evaluator:
    def __init__(self, eval_args):
        self.eval_args = eval_args

    def load(self, model, model_args={}):
        if not ('lora' in self.eval_args.pretrained_model_path and os.path.exists(os.path.join(self.eval_args.pretrained_model_path, 'adapter_config.json'))): # loading model for non-lora/non-qlora pretraining
            model.load_llm(**model_args['llm'])
            model.load_vision_tower(**model_args['vision_tower'])
            model.load_connector(**model_args['connector'])
        else:
            model.language_model = model.language_model.from_pretrained(model_args['llm']['model_name_or_path'],attn_implementation='flash_attention_2',torch_dtype=model_args['llm']['torch_dtype'])
            model.load_vision_tower(**model_args['vision_tower'])
            model.load_connector(**model_args['connector'])
            model.to(model_args['llm']['torch_dtype'])
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, self.eval_args.pretrained_model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')

        return model
    
    @staticmethod
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

    def load_pretrained_model(self, load_type='hf', load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", **kwargs):
        model_name_or_path = self.eval_args.pretrained_model_path
        kwargs = {"device_map": device_map, **kwargs}
        if device != "cuda":
            kwargs['device_map'] = {"": device}

        if load_8bit:
            kwargs['load_in_8bit'] = True
        elif load_4bit:
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        else:
            kwargs['torch_dtype'] = torch.float16

        model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
        model = TinyLlavaForConditionalGeneration(model_config)

        model_args = self.generate_model_args(config_in_model=model_config,
                                              pretrained_model_path=args.pretrained_model_path)
        model = self.load(model, model_args)
        #########################
        # hf_path = 'tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B'
        # model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)

        # config = model.config
        # tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)
        #########################
        image_processor = model.vision_tower._image_processor
        context_len = getattr(model.config, 'max_sequence_length', 2048)

        tokenizer = model.tokenizer

        return model, tokenizer, image_processor, context_len

    
    def load_finetune_model(self, load_type='hf', load_8bit=False, load_4bit=False, device_map="auto",
                            device="cuda", **kwargs):
        model_name_or_path = self.eval_args.finetune_model_path
        kwargs = {"device_map": device_map, **kwargs}
        if device != "cuda":
            kwargs['device_map'] = {"": device}

        if load_8bit:
            kwargs['load_in_8bit'] = True
        elif load_4bit:
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        else:
            kwargs['torch_dtype'] = torch.float16
        if model_name_or_path is not None and 'lora' not in model_name_or_path:
            model = TinyLlavaForConditionalGeneration.from_pretrained(model_name_or_path,low_cpu_mem_usage=True,torch_dtype=torch.float16)
            
        """从load_model.py来的，还没添加缺失的函数"""
        # elif model_name_or_path is not None and 'lora' in model_name_or_path:
        #     if os.path.exists(os.path.join(model_name_or_path, 'adapter_config.json')):
        #         model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
        #         model = TinyLlavaForConditionalGeneration(model_config)
        #         language_model_ckp_path = os.path.join(model_name_or_path, 'language_model/pytorch_model.bin')
        #         language_model_ckp = load_base_ckp_for_lora(language_model_ckp_path)
        #         model.language_model.load_state_dict(language_model_ckp)
        #         vision_tower_ckp_path = os.path.join(model_name_or_path, 'vision_tower/pytorch_model.bin')
        #         vision_tower_ckp = load_base_ckp_for_lora(vision_tower_ckp_path)
        #         model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)
        #         connector_ckp_path = os.path.join(model_name_or_path, 'connector/pytorch_model.bin')
        #         connector_ckp = load_base_ckp_for_lora(connector_ckp_path)
        #         model.connector.load_state_dict(connector_ckp)
        #         model.to(torch.float16)
        #         from peft import PeftModel
        #         print('Loading LoRA weights...')
        #         model = PeftModel.from_pretrained(model, model_name_or_path)
        #         print('Merging LoRA weights...')
        #         model = model.merge_and_unload()
        #         print('Model is loaded...')
            
        image_processor = model.vision_tower._image_processor
        context_len = getattr(model.config, 'max_sequence_length', 2048)
        # tokenizer = AutoTokenizer.from_pretrained(model.config.llm_model_name_or_path, use_fast=False, padding_side="right")
        tokenizer = model.tokenizer
        #tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer, image_processor, context_len


def cross_entropy(
    logits, targets, attention_mask: torch.Tensor = None
):

    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if attention_mask is not None:
        attention_mask = attention_mask.reshape(-1)
        targets = targets.masked_fill(~attention_mask, -1)

    return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1, reduction='sum')


@torch.no_grad()
def eval_model(args):

    # Model
    disable_torch_init()
    evaluator = Evaluator(args)
    
    if evaluator.eval_args.pretrained_model_path is not None:
        model, tokenizer, image_processor, context_len = evaluator.load_pretrained_model()
    if evaluator.eval_args.finetune_model_path is not None:
        model, tokenizer, image_processor, context_len = evaluator.load_finetune_model()

    model.eval()
    
    eval_dataset = EvalDataset(
        args=args,
        task_name=args.task_name,
        block_size=args.block_size + 1,
        tokenizer=tokenizer,
        stride=args.stride,
        vocab_size=tokenizer.vocab_size,
        file_num=args.file_num
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    model.to(device='cuda')
    model = model.to(torch.bfloat16)
    losses = []
    for k, (eval_data, attention_mask) in enumerate(tqdm(eval_dataloader)):
        input_ids = eval_data[:, 0: args.block_size].contiguous().to(device='cuda')
        targets = eval_data[:, 1: args.block_size + 1].contiguous().long().to(device='cuda')
        attention_mask = attention_mask[:, 1: args.block_size + 1].contiguous().to(device='cuda')
        logits = model(input_ids=input_ids).logits
        loss = cross_entropy(logits, targets, attention_mask=attention_mask)
        loss = loss.cpu().item()
        losses.append(loss)
        print("%.8f" % loss)
    
    total_loss = np.array(losses).sum()

    print("-"*10, "Result", "-"*10)
    print("Total loss:", total_loss)
    print("Character num:", eval_dataset.character_num)
    print("BPC:", total_loss / (eval_dataset.character_num * np.log(2)) )

    print("finished")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """Test pretrain"""
    # parser.add_argument("--pretrained_model_path", type=str, default="/home/atuin/b211dd/b211dd20/tinyllava/checkpoints/llava_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-pretrain")
    # parser.add_argument("--finetune_model_path", type=str, default=None)
 
    """Test finetune"""
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--finetune_model_path", type=str, default="/home/atuin/b211dd/b211dd20/tinyllava/checkpoints/llava_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune_full")
    
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="phi")
    parser.add_argument("--temperature", type=float, default=0.2)

    # BPC
    parser.add_argument(
        "--task_name",
        type=str,
    )
    parser.add_argument(
        '--block_size',
        type=int,
        default=1900,
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=512,
    )
    parser.add_argument(
        '--batch_size',
        type=int
    )
    parser.add_argument(
        '--file_num',
        default=-1,
        type=int
    )
    parser.add_argument(
        '--flash',
        action="store_true",
        help="set this if you want to use flash attention",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)

    args = parser.parse_args()
    print(args)

    eval_model(args)

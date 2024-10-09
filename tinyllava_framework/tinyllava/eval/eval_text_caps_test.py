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

from torch.utils.data import Dataset, DataLoader



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def collate_fn(batch):
    input_ids = batch[0]['input_ids'].unsqueeze(0)
    image_tensors = batch[0]['image'].unsqueeze(0)
    image_name = batch[0]['image_name']

    if 'intervention_locations' in batch[0] and all(ins.get("intervention_locations") is not None for ins in batch):
        intervention_locations = [instance['intervention_locations'] for instance in batch]
        if all(isinstance(x[0], list) and isinstance(x, list) and len(intervention_locations[0]) == len(x) and len(intervention_locations[0][0]) == len(x[0]) for x in intervention_locations):
            intervention_locations = torch.tensor(intervention_locations)

    data_dict = {}
    data_dict['input_ids'] = input_ids
    data_dict['image'] = image_tensors
    if 'intervention_locations' in batch[0] and batch[0]['intervention_locations'] != None:
        data_dict['intervention_locations'] = intervention_locations
    data_dict['image_name'] = image_name

    return data_dict


class Converter2LlamaFormat:
    PROMPT_TEXT_CAPS = "<image>\nProvide a one-sentence caption for the provided image.\nReference OCR token: "

    def __init__(self, question_file):
        self.question_file = question_file
        with open(self.question_file, 'r') as file:
            data = json.load(file)
        print("start converting data to llama format ...")
        self.data_in_llama_format = []
        self._convert_data(data)
        print("succeed in converting data to llama format!")

    def _fit_in_template(self, img_id: str):
        return {
            "id": img_id,
            "image": f"test_images/{img_id}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": self.PROMPT_TEXT_CAPS
                }
            ]
        }

    def _convert_data(self, data):
        for sample in tqdm(data['data']):
            self.data_in_llama_format.append(self._fit_in_template(sample['image_name']))


class LazyEvalDataset(Dataset):
    """Dataset for evaluation."""

    def __init__(self, list_data_dict: List[Dict],
                 tokenizer: transformers.PreTrainedTokenizer,
                 image_processor: any,
                 data_args: any,
                 eval_args: any,
                 **kwargs):
        super(LazyEvalDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.eval_args = eval_args
        self.image_preprocess = ImagePreprocess(image_processor, data_args)
        self.text_preprocess = TextPreprocessReft(tokenizer, eval_args.conv_mode, eval_args.reft_pos_configs) \
            if hasattr(eval_args, 'include_reft') else TextPreprocess(tokenizer, eval_args.conv_mode)


class TextCapsEvalDataset(LazyEvalDataset):
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]), mode='eval')
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            data_dict['image_name'] = image_file
            image_folder = self.eval_args.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = self.image_preprocess(image)
            data_dict['image'] = image
        return data_dict


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


def eval_model(args):
    debugpy.listen(("0.0.0.0", 5677))
    print("waitng for debugger attach ...")
    debugpy.wait_for_client()
    debugpy.breakpoint()
    print("debugger is attached!")

    # Model
    disable_torch_init()

    evaluator = Evaluator(args)
    model, tokenizer, image_processor, context_len = evaluator.load_pretrained_model()
    
    model.load_reft_adaptor_from_ckpt(args.pretrained_model_path) if args.include_reft else None

    converter = Converter2LlamaFormat(args.question_file)
    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    args.reft_pos_configs = model.reft_pos_configs if args.include_reft == True else None
    eval_dataset = TextCapsEvalDataset(list_data_dict=converter.data_in_llama_format,
                                    tokenizer=tokenizer,
                                    data_args=model.config,
                                    eval_args=args,
                                    image_processor=image_processor)
    data_loader = DataLoader(eval_dataset, batch_size=1, num_workers=8, shuffle=False, collate_fn=collate_fn)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True) if os.path.dirname(answers_file) else None
    # ans_file = open(answers_file, "w")
    res = []

    # print("Tokenizer's eos token: ", tokenizer.eos_token)
    model.to(device='cuda')
    model = model.to(torch.bfloat16)
    for data_dict in tqdm(data_loader):
        # text preprocessing
        input_ids = data_dict['input_ids']
        intervention_locations = data_dict['intervention_locations'] if hasattr(args, 'include_reft') and args.include_reft == True else None
        # image preprocessing
        image_tensor = data_dict['image']
        image_name = data_dict['image_name']
        match = re.search(r'test_images/(.*)\.jpg', image_name)
        image_id = match.group(1)
        image_sizes = image_tensor.size

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        data_dict = dict(images=image_tensor.to(dtype=torch.float32, device='cuda', non_blocking=True),
                         pad_token_id=tokenizer.pad_token_id,
                         do_sample=True if args.temperature > 0 else False,
                         temperature=args.temperature,
                         top_p=args.top_p,
                         num_beams=args.num_beams,
                         max_new_tokens=args.max_new_tokens,
                         image_sizes=image_sizes,
                         use_cache=True,
                         intervention_locations=intervention_locations)
        with torch.inference_mode():
            output_ids = model.generate(input_ids, **data_dict)

        outputs = tokenizer.batch_decode(output_ids, skip_special_token=False)[0].strip()
        print('------------------------------')
        print(outputs)
        res.append({"image_id": image_id, "caption": outputs})
        # ans_file.write(json.dumps({"image_id": str(input_ids), "caption": outputs}))
    
    with open(answers_file, "w") as ans_file:
        json.dump(res, ans_file, indent=4)

    #     ans_id = shortuuid.uuid()
    #     ans_file.write(json.dumps({"question_id": idx,
    #                             "prompt": cur_prompt,
    #                             "text": outputs,
    #                             "answer_id": ans_id,
    #                             "model_id": args.model_base,
    #                             "metadata": {}}) + "\n")
    #     # ans_file.flush()
    # ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="/home/atuin/b211dd/b211dd19/data/checkpoints/llava_factory/three_stage_finetuning/text_caps-phi-full-pre")
    # /home/hpc/b211dd/b211dd19/code/text_caps-reft-pre-save_again2
    # text_caps-reft-pre-save_again2 ~/code
    # "/home/atuin/b211dd/b211dd19/data/checkpoints/llava_factory/three_stage_finetuning/text_caps-reft-pre-save_again2"
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval/text_caps")
    parser.add_argument("--question-file", type=str, default="/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval/text_file_small_dataset/TextCaps_0.1_test.json")
    parser.add_argument("--answers-file", type=str, default="answer_full_new.jsonl")
    parser.add_argument("--conv-mode", type=str, default="phi")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--image_aspect_ratio", type=str, default="square")

    parser.add_argument("--include_reft", type=bool, default=False)
    args = parser.parse_args()

    eval_model(args)
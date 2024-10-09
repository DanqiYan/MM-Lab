from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import copy

from .formatter import EmptyFormatter, StringFormatter
from .formatter import Formatter
from ...utils.constants import *
from .utils_template import parse_positions, get_intervention_locations, replace_pad_in_2d_list_intervention_locations

from transformers import PreTrainedTokenizer
import torch
    

def add_integer_to_2d_list(lst, integer):
    return [[(element + integer) if element != -1 else -1 for element in sublist] for sublist in lst]

def add_integer_to_list(lst, integer):
    return [(element + integer) if element != -1 else -1 for element in lst]


@dataclass
class Template:
    format_image_token: "Formatter"
    format_user: "Formatter"
    format_assistant: "Formatter"
    system: "Formatter"
    separator: "Formatter"
    reft_pos_configs: Dict = None

    def get_intervention_locarions_for_multimodal_prompt(self, question_prompt, tokenizer, question_content):
        # intervention locations
        question_prompt_ids = self.tokenizer_image_token(question_prompt, tokenizer, return_tensors='pt')

        img_token_position = torch.where(question_prompt_ids == IMAGE_TOKEN_INDEX)[0]
        num_img_token = len(img_token_position)

        question_prompt_length = len(question_prompt_ids)
        
        first_n, last_n = parse_positions(self.reft_pos_configs['intervention_positions'])
        img_embed_token_len = self.reft_pos_configs['img_embed_token_len']
        question_prompt_embed_length = question_prompt_length + (img_embed_token_len-1)*num_img_token
        if num_img_token == 0:
            question_content_ids = self.tokenizer_image_token(question_content, tokenizer, return_tensors='pt')
            question_content_length = len(question_content_ids)
            embed_bias = question_prompt_length - question_content_length
        else:
            if not self.reft_pos_configs['intervene_include_img_embed']:
                embed_bias = img_token_position[-1].item() + 1 + num_img_token * (img_embed_token_len-1)
            else:
                embed_bias = img_token_position[0].item()
        intervention_part_len = question_prompt_embed_length - embed_bias
        intervention_locations = get_intervention_locations(
            last_position=intervention_part_len, 
            first_n=first_n,
            last_n=last_n,
            pad_mode="first",
            share_weights=self.reft_pos_configs['reft_share_weights'],
            num_interventions=self.reft_pos_configs['num_interventions']['llm']
        )

        intervention_locations = add_integer_to_2d_list(intervention_locations, embed_bias)
        replace_pad_in_2d_list_intervention_locations(intervention_locations)
        return intervention_locations
                    

    def encode(self, messages, tokenizer, mode='train'):
        """
        1. get list form messages(conversations:[{from:human, value:message}, {from:gpt, value:message}])
            ===>  human_list, value_list
        2. prompt two list
        3. tokenize prompt
        4. make target
        """
        question_list, answer_list = self.get_list_from_message(messages, mode)
        prompt, question_prompt, question_content = self.prompt(question_list, answer_list, return_prompt=True)
        input_ids = self.tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
        intervention_locations = self.get_intervention_locarions_for_multimodal_prompt(question_prompt, tokenizer, question_content) \
            if self.reft_pos_configs != None else None
        if mode == 'train':
            labels = self.make_labels(input_ids, prompt, tokenizer)
            return dict(
                input_ids=input_ids,
                labels=labels,
                intervention_locations=intervention_locations,
            )
        else:
            return dict(
                input_ids=input_ids,
                prompt=prompt,
                intervention_locations=intervention_locations,
            )
        
    
    def get_list_from_message(self, messages, mode='train'):
        return self._get_list_from_message(messages, mode)
    
    def _get_list_from_message(self, messages, mode='train'):
        """
        messages  ====>  [{from:human, value:message}, {from:gpt, value:message}]
        """
        question_list = []
        answer_list = []
        first_is_not_question = 0
        for i, message in enumerate(messages):
            if i == 0 and message['from'] != 'human':
                first_is_not_question = 1
                continue
            if i % 2 == first_is_not_question:
                question_list.append(message['value'])
            else:
                answer_list.append(message['value'])
        if mode=='train':
            assert len(question_list) == len(answer_list) , \
                f"qa is not match : length_q:{len(question_list)} vs length_a:{len(answer_list)}"
        return question_list, answer_list
    

    def prompt(
        self,
        question_list, answer_list,
        return_prompt: bool = False
    ):
        if type(question_list) is str:
            question_list = [question_list]
        if type(answer_list) is str:
            answer_list = [answer_list]    
        msg, question_prompt, question_content = self._prompt(question_list, answer_list) if len(answer_list) == len(question_list) else self._prompt_only_q(question_list)
        if not return_prompt:
            return msg
        else:
            return msg, question_prompt, question_content

    def _prompt(
        self,
        question_list, answer_list,
    ):
        msg = ""
        question_prompt = ""
        question_content = None
        for i, (question, answer) in enumerate(zip(question_list, answer_list)):
            if i == 0:
                msg += self.system.apply()
            if DEFAULT_IMAGE_TOKEN in question:
                question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                question = self.format_image_token.apply(content=question).strip()
            msg += self.format_user.apply(content=question)
            if self.reft_pos_configs != None and self.reft_pos_configs['intervened_prompt_part'] == "first_round" and i == 0:
                question_prompt = msg
                if DEFAULT_IMAGE_TOKEN not in question:
                    question_content = question
            msg += self.format_assistant.apply(content=answer)
        return msg, question_prompt, question_content
    
    def _prompt_only_q(
        self,
        question_list
    ):
        msg = ""
        question_prompt = ""
        question_content = None
        for i, question in enumerate(question_list):
            if i == 0:
                msg += self.system.apply()
            if DEFAULT_IMAGE_TOKEN in question:
                question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                question = self.format_image_token.apply(content=question).strip()
            msg += self.format_user.apply(content=question)
            if self.reft_pos_configs != None and self.reft_pos_configs['intervened_prompt_part'] == "first_round" and i == 0:
                question_prompt = msg
                if DEFAULT_IMAGE_TOKEN not in question:
                    question_content = question
        return msg, question_prompt, question_content
    
    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        sep, eos_token = self.separator.apply()
        total_len = int(labels.ne(tokenizer.pad_token_id).sum())
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            total_len += prompt.count(eos_token)
        rounds = prompt.split(eos_token)
        eos_token_length = len(tokenizer.encode(eos_token))
        labels, cur_len = self._make_masks(labels, tokenizer, sep, eos_token_length, rounds)
        if cur_len < tokenizer.model_max_length:
            import time
            if cur_len != total_len:
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print("number of rounds: ", len(rounds) - 1)
                print("rounds: ", rounds[:-1])
                print("prompt: ", prompt)
                print(labels)
                print(input_ids)
                time.sleep(5)
                labels[:] = IGNORE_INDEX
        return labels
        
        
        
    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 0
        for rou in rounds:
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) + eos_token_length
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len
        
    @classmethod    
    def tokenizer_image_token(cls, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        def _insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in _insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids






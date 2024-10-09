from typing import Any

from .template import TemplateFactory


class TextPreprocess:
    def __init__(self, tokenizer, version):
        self.tokenizer = tokenizer
        self.template = TemplateFactory(version)()
    
    def __call__(self, messages, mode='train'):
        return self.template.encode(messages, self.tokenizer, mode)
    
class TextPreprocessReft(TextPreprocess):
    def __init__(self, tokenizer, version, reft_pos_configs):
        super().__init__(tokenizer, version)
        self.template = TemplateFactory(version)(reft_pos_configs=reft_pos_configs)
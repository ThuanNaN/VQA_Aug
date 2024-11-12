from transformers import AutoTokenizer, DeiTImageProcessor
from .utils import dict_map
from .configuration import VQAConfig


class TextProcessorWrapper:
    def __init__(self, config: VQAConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.lang_model_name)
        self.dict_map = dict_map

    def __call__(self, text):
        return self.text_processor(text)

    def text_processor(self, text_input):
        text_input = self.text_tone_normalize(text_input)
        input_ids = self.tokenizer(text_input,
                                   max_length=self.config.max_length,
                                   padding=self.config.padding,
                                   truncation=self.config.truncation,
                                   return_token_type_ids=self.config.return_token_type_ids,
                                   return_tensors='pt')
        return input_ids

    def text_tone_normalize(self, text):
        for i, j in self.dict_map.items():
            text = text.replace(i, j)
        return text

class VQAProcessor:
    def __init__(self, config: VQAConfig = VQAConfig()):
        self.config = config

    def get_img_processor(self):
        return DeiTImageProcessor.from_pretrained(self.config.vis_model_name)

    def get_text_processor(self):
        return TextProcessorWrapper(self.config)
    
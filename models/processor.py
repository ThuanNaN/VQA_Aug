from transformers import AutoTokenizer
from .utils import dict_map
from .configuration import VQAConfig
import timm

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
        model_name = self.config.vis_model_name
        if model_name ==  "beitv2_base_patch16_224.in1k_ft_in22k":
            data_config = timm.data.resolve_model_data_config(
                {
                    'input_size': (3, 224, 224),
                    'interpolation': 'bicubic',
                    'mean': (0.485, 0.456, 0.406),
                    'std': (0.229, 0.224, 0.225),
                    'crop_pct': 0.9,
                    'crop_mode': 'center'
                }
            )
            transforms = timm.data.create_transform(**data_config, is_training=False)
            return transforms
        elif model_name == "timm/resnet18.a1_in1k":
            data_config = timm.data.resolve_model_data_config(
                {
                    'input_size': (3, 224, 224),
                    'interpolation': 'bicubic',
                    'mean': (0.485, 0.456, 0.406),
                    'std': (0.229, 0.224, 0.225),
                    'crop_pct': 0.95,
                    'crop_mode': 'center'
                }
            )
            transforms = timm.data.create_transform(**data_config, is_training=False)
            return transforms
        else:
            raise ValueError(f"Invalid model name {model_name}") 

    def get_text_processor(self):
        return TextProcessorWrapper(self.config)
    
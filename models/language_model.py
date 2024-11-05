import torch
from torch import nn, Tensor
from transformers import AutoModel, AutoTokenizer
from .utils import dict_map
from .configuration import LanguageConfig

class TextProcessorWrapper:
    def __init__(self, config: LanguageConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
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


class LanguageModel(nn.Module):
    config: LanguageConfig = LanguageConfig()
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.config = LanguageModel.config
        self.model = AutoModel.from_pretrained(self.config.model_name)
        self.projection = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.projection_dim),
            nn.ReLU()
        )

    def forward(self, inputs: list, augment_thresh: float) -> Tensor:
        r_thresh = torch.rand(1)
        if self.training and self.config.is_augment and r_thresh < augment_thresh:
            embed_lst = [self.model(**input_dict)['last_hidden_state'][:, 0, :] 
                         for input_dict in inputs]
            para_features_t = torch.stack(embed_lst, dim=1)
            x = torch.sum(para_features_t, dim=1)

        else:
            x = self.model(**inputs[0])
            x = x['last_hidden_state'][:, 0, :] # (batch, hidden_size)

        return self.projection(x)

    @staticmethod
    def get_text_processor():
        return TextProcessorWrapper(LanguageModel.config)

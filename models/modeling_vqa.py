from typing import List, Dict
import torch
from torch import nn, Tensor
from .language_model import LanguageModel
from .vision_model import VisionModel
from .configuration import VQAConfig


class MLP(nn.Module):
    def __init__(self, config: VQAConfig):
        super(MLP, self).__init__()
        self.config = config
        self.act_fn = config.act_fn
        self.fc1 = nn.Linear(config.hidden_size * 2, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.num_classes)

    def forward(self, 
                language_output: Tensor, 
                vision_output: Tensor
                ) -> Tensor:
        x = torch.cat((language_output, vision_output), 1)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x

class VQAModel(nn.Module):
    config: VQAConfig = VQAConfig()
    def __init__(self):
        super(VQAModel, self).__init__()
        self.config = VQAModel.config
        self.vision_model = VisionModel()
        self.language_model = LanguageModel()
        self.mlp = MLP(VQAModel.config)

    def forward(self,
                text_inputs_lst: List[Dict[str, Tensor]],  
                img_inputs_lst: Tensor
                )-> Tensor:
        language_thresh, vision_thresh = self.get_threshold()
        language_output = self.language_model(text_inputs_lst, language_thresh) # (batch, 512)
        vision_output = self.vision_model(img_inputs_lst, vision_thresh) # (batch, 512)
        logits = self.mlp(language_output, vision_output)
        return logits

    def get_threshold(self):
        if not self.config.use_dynamic_thresh:
            return self.config.language_augment_thresh, self.config.vision_augment_thresh
        
        return self.config.language_augment_thresh, self.config.vision_augment_thresh

        # decay = (self.start_threshold - self.min_threshold) * (self.current_epoch / self.total_epochs)
        # updated_thresh = max(self.start_threshold - decay, self.min_threshold)

        # return updated_thresh, updated_thresh


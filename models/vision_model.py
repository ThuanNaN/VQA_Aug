import torch
from torch import nn, Tensor
from transformers import DeiTModel, DeiTImageProcessor
from .configuration import VisionConfig


class VisionModel(nn.Module):
    config: VisionConfig = VisionConfig()
    def __init__(self):
        super(VisionModel, self).__init__()
        self.config = VisionModel.config
        self.model = DeiTModel.from_pretrained(self.config.model_name,
                                               attn_implementation="sdpa", 
                                               torch_dtype=self.config.torch_dtype)
        self.projection = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.projection_dim),
            nn.ReLU()
        )
    def forward(self, 
                inputs: Tensor,
                augment_thresh: float,
                )-> Tensor:
        r_thresh = torch.rand(1)
        if self.training and self.config.is_augment and r_thresh < augment_thresh:
            img_features_t = [self.model(inputs[:, idx])["pooler_output"]
                              for idx in range(inputs.size(1))]
            img_features_t = torch.stack(img_features_t, dim=1)
            x = torch.sum(img_features_t, dim=1)
        else:
            x = self.model(inputs[:, 0])["pooler_output"] # (batch, hidden_size)

        return self.projection(x)

    @staticmethod
    def get_img_processor():
        return DeiTImageProcessor.from_pretrained(VisionModel.config.model_name)

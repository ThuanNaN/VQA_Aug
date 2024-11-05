from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class LanguageConfig:
    model_name: str = "vinai/bartpho-word"
    max_length: int = 50
    padding: str = 'max_length'
    truncation: bool = True
    return_token_type_ids: bool = False
    hidden_size: int = 1024
    projection_dim = 512
    is_augment: bool = True


@dataclass
class VisionConfig:
    model_name = "facebook/deit-base-distilled-patch16-224"
    torch_dtype: torch.dtype = torch.float16
    hidden_size: int = 768
    projection_dim = 512
    is_augment: bool = True



@dataclass
class VQAConfig:
    act_fn: nn.Module = nn.GELU()
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_classes: int = 353

    use_dynamic_thresh: bool = False
    language_augment_thresh: float = 0.5
    vision_augment_thresh: float = 0.5


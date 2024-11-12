from dataclasses import dataclass, field
import torch
from torch import nn
from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

@dataclass
class VQAConfig:
    lang_model_name: str = "vinai/bartpho-word"
    max_length: int = 32
    padding: str = 'max_length'
    truncation: bool = True
    return_token_type_ids: bool = False
    lang_hidden_size: int = 1024

    vis_model_name = "facebook/deit-base-distilled-patch16-224"
    vis_hidden_size: int = 768

    projection_scale: int = 4
    attn_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    activation_fn: nn.Module = nn.GELU()
    hidden_size: int = 1024
    intermediate_size: int = 2048
    cross_augment_heads: int = 8
    cross_augment_layer: int = 1
    query_attn_heads: int = 8
    query_layer: int = 1
    num_classes: int = 353

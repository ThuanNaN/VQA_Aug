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
class LanguageConfig:
    model_name: str = "vinai/bartpho-word"
    max_length: int = 32
    padding: str = 'max_length'
    truncation: bool = True
    return_token_type_ids: bool = False
    hidden_size: int = 1024
    self_attn_heads: int = 8
    self_attn_head_dim: int = 128
    cross_attn_heads: int = 8
    attn_dropout: float = 0.1
    mlp_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1

@dataclass
class VisionConfig:
    model_name = "facebook/deit-base-distilled-patch16-224"
    hidden_size: int = 768
    self_attn_heads: int = 8
    self_attn_head_dim: int = 96
    cross_attn_heads: int = 8
    attn_dropout: float = 0.1
    mlp_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1

@dataclass
class VQAConfig:
    activation_fn: nn.Module = nn.GELU()
    hidden_size: int = 1024
    intermediate_size: int = 2048
    query_attn_heads: int = 8
    layer_norm_eps: float = 1e-6
    attn_dropout: float = 0.1
    query_layer: int = 4
    num_classes: int = 353
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    language_config: LanguageConfig = field(default_factory=LanguageConfig)

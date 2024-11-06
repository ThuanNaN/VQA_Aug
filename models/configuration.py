from dataclasses import dataclass, field
from torch import nn

@dataclass
class LanguageConfig:
    model_name: str = "vinai/bartpho-word"
    max_length: int = 32
    padding: str = 'max_length'
    truncation: bool = True
    return_token_type_ids: bool = False
    frozen_backbone: bool = True
    hidden_size: int = 1024
    self_attn_heads: int = 8
    self_attn_head_dim: int = 128
    cross_attn_heads: int = 8
    attn_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    is_augment: bool = True

@dataclass
class VisionConfig:
    model_name = "facebook/deit-base-distilled-patch16-224"
    frozen_backbone: bool = True
    hidden_size: int = 768
    self_attn_heads: int = 8
    self_attn_head_dim: int = 96
    cross_attn_heads: int = 8
    attn_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    is_augment: bool = True

@dataclass
class VQAConfig:
    activation_fn: nn.Module = nn.GELU()
    hidden_size: int = 1024
    intermediate_size: int = 2048
    query_attn_heads: int = 8
    layer_norm_eps: float = 1e-6
    attn_dropout: float = 0.1
    num_classes: int = 353
    use_dynamic_thresh: bool = False
    language_augment_thresh: float = 0.5
    vision_augment_thresh: float = 0.5
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    language_config: LanguageConfig = field(default_factory=LanguageConfig)


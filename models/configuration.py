from dataclasses import dataclass


@dataclass
class VQAConfig:
    lang_model_name: str = "vinai/bartpho-word"
    max_length: int = 32
    padding: str = 'max_length'
    truncation: bool = True
    return_token_type_ids: bool = False

    vis_model_name: str = "beitv2_base_patch16_224.in1k_ft_in22k"
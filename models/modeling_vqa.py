from typing import List, Dict
import math
import torch
from torch import nn, Tensor
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .configuration import VQAConfig, double_quant_config


class CrossAttention(nn.Module):
    def __init__(self, encoder_hidden_size, num_heads, hidden_size, dropout) -> None:
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(encoder_hidden_size, self.all_head_size)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states=None,
                encoder_hidden_states=None,
                ):
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs_dropped = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)

        output = context_layer.view(*new_context_layer_shape)
        return self.resid_dropout(self.projection(output))


class MLP(nn.Module):
    def __init__(self, input_dim, projection_scale, activation_fn) -> None:
        super().__init__()
        intermediate_dim = int(input_dim * projection_scale)
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, input_dim)
        self.activation_fn = activation_fn

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class LanguageModel(nn.Module):
    def __init__(self, config: VQAConfig):
        super().__init__()
        base_model = AutoModel.from_pretrained(config.lang_model_name,
                                               quantization_config=double_quant_config,
                                               low_cpu_mem_usage=True)
        base_model.config.use_cache = False

        quantized_model = prepare_model_for_kbit_training(base_model, double_quant_config)

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["k_proj", "v_proj", "q_proj", "out_proj"],
            lora_dropout=config.lora_dropout,
            bias="none")
        self.model = get_peft_model(quantized_model, lora_config)
        self.model.print_trainable_parameters()


    def forward(self, inputs: list) -> List[Tensor]:
        outputs = [self.model(**input_dict)['last_hidden_state'] for input_dict in inputs]
        return outputs


class VisionModel(nn.Module):
    def __init__(self, config: VQAConfig):
        super().__init__()
        self.config = config
        base_model = AutoModel.from_pretrained(config.vis_model_name,
                                               attn_implementation="sdpa",
                                               quantization_config=double_quant_config,
                                               low_cpu_mem_usage=True)
        base_model.config.use_cache = False

        quantized_model = prepare_model_for_kbit_training(base_model, double_quant_config)

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["attention.query", "attention.key", "attention.value"],
            lora_dropout=config.lora_dropout,
            bias="none"
        )
        self.model  = get_peft_model(quantized_model, lora_config)
        self.model.print_trainable_parameters()


    def forward(self, inputs: Tensor ) -> Tensor:
        outputs = [self.model(inputs[:, idx])['last_hidden_state'] for idx in range(inputs.size(1))]
        return outputs


class CrossAugmentation(nn.Module):
    def __init__(self, 
                 encoder_hidden_size, 
                 num_heads, 
                 hidden_size, 
                 projection_scale, 
                 activation_fn, 
                 dropout, 
                 layer_norm_eps
                 ) -> None:
        super().__init__()
        self.attn = CrossAttention(
            encoder_hidden_size=encoder_hidden_size,
            num_heads=num_heads,
            hidden_size=hidden_size,
            dropout=dropout
        )
        self.mlp = MLP(
            input_dim=hidden_size,
            projection_scale=projection_scale,
            activation_fn=activation_fn
        )
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, query: Tensor, encode_hidden_states: List[Tensor]) -> Tensor:
        for encode_hidden_state in encode_hidden_states:
            query = self.norm(query + self.attn(query, encode_hidden_state))
            query = self.norm(query + self.mlp(query))
        return query


class Classifier(nn.Module):
    def __init__(self, input_dim, intermediate_dim, activation_fn):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, input_dim)
        self.activation_fn = activation_fn

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x

class VQAModel(nn.Module):
    def __init__(self, config: VQAConfig = VQAConfig()):
        super().__init__()
        self.config = config
        self.vision_model = VisionModel(config)
        self.language_model = LanguageModel(config)

        self.lang_cross_augment = nn.ModuleList(
            CrossAugmentation(
                encoder_hidden_size=config.vis_hidden_size, 
                num_heads=config.cross_augment_heads, 
                hidden_size=config.lang_hidden_size, 
                projection_scale=config.projection_scale, 
                activation_fn=config.activation_fn, 
                dropout=config.attn_dropout, 
                layer_norm_eps=config.layer_norm_eps
            ) for _ in range(config.cross_augment_layer)
        )

        self.vision_cross_augment = nn.ModuleList(
            CrossAugmentation(
                encoder_hidden_size=config.lang_hidden_size,
                num_heads=config.cross_augment_heads,
                hidden_size=config.vis_hidden_size,
                projection_scale=config.projection_scale,
                activation_fn=config.activation_fn,
                dropout=config.attn_dropout,
                layer_norm_eps=config.layer_norm_eps
            ) for _ in range(config.cross_augment_layer)
        )

        self.query_attn = nn.ModuleList(
            CrossAttention(
                encoder_hidden_size=config.vis_hidden_size, 
                num_heads=config.query_attn_heads, 
                hidden_size=config.lang_hidden_size, 
                dropout=config.attn_dropout
            )
            for _ in range(config.query_layer)
        )

        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = Classifier(config.hidden_size, config.intermediate_size, config.activation_fn)

    def forward(self,
                text_inputs_lst: List[Dict[str, Tensor]],
                img_inputs_lst: Tensor
                ) -> Tensor:
        language_hidden_states = self.language_model(text_inputs_lst)
        vision_hidden_states = self.vision_model(img_inputs_lst)

        ori_lang, aug_lang = language_hidden_states[0], language_hidden_states
        ori_vision, aug_vision = vision_hidden_states[0], vision_hidden_states

        for layer in range(self.config.cross_augment_layer):
            ori_lang = self.lang_cross_augment[layer](ori_lang, aug_vision) 
            ori_vision = self.vision_cross_augment[layer](ori_vision, aug_lang)

        for query_attn_layer in self.query_attn:
            ori_lang = query_attn_layer(ori_lang, ori_vision)  

        pooled_output = ori_lang[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        logits = self.classifier(pooled_output)
        return logits

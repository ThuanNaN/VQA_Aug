from typing import List, Dict
import math
import torch
from torch import nn, Tensor
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .configuration import LanguageConfig, VisionConfig, VQAConfig, double_quant_config


# class SelfAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, head_dim, dropout) -> None:
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = head_dim
#         if self.head_dim * self.num_heads != self.embed_dim:
#             raise ValueError(
#                 f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
#                 f" {self.num_heads})."
#             )
#         self.scale = self.head_dim**-0.5
#         self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
#         self.projection = nn.Linear(self.embed_dim, self.embed_dim)
#         self.attn_dropout = nn.Dropout(dropout)
#         self.resid_dropout = nn.Dropout(dropout)

#     def forward(self, inputs: Tensor) -> Tensor:
#         bsz, num_feat, embed_dim = inputs.size()
#         mixed_qkv = self.qkv(inputs)
#         mixed_qkv = mixed_qkv.reshape(bsz, num_feat, 3, self.num_heads, embed_dim // self.num_heads
#                                       ).permute(2, 0, 3, 1, 4)
#         query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

#         attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
#         attention_scores = attention_scores * self.scale

#         attention_probs = nn.functional.softmax(attention_scores, dim=-1)
#         attention_probs = self.attn_dropout(attention_probs)

#         context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)
#         new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
#         context_layer = context_layer.reshape(new_context_layer_shape)

#         return self.resid_dropout(self.projection(context_layer))


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
    def __init__(self, config: LanguageConfig | VisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size*4)
        self.fc2 = nn.Linear(config.hidden_size*4, config.hidden_size)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(config.mlp_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class LanguageModel(nn.Module):
    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.config = config
        base_model = AutoModel.from_pretrained(config.model_name,
                                               quantization_config=double_quant_config,
                                               low_cpu_mem_usage=True)
        base_model.config.use_cache = False
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["k_proj", "v_proj", "q_proj", "out_proj"],
            lora_dropout=config.lora_dropout,
            bias="none")
        peft_model = get_peft_model(base_model, lora_config)
        self.model = prepare_model_for_kbit_training(peft_model, double_quant_config)


    def forward(self, inputs: list) -> List[Tensor]:
        outputs = [self.model(**input_dict)['last_hidden_state'] for input_dict in inputs]
        return outputs


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        base_model = AutoModel.from_pretrained(config.model_name,
                                               attn_implementation="sdpa",
                                               quantization_config=double_quant_config,
                                               low_cpu_mem_usage=True)
        base_model.config.use_cache = False
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["attention.query", "attention.key", "attention.value"],
            lora_dropout=config.lora_dropout,
            bias="none"
        )
        peft_model = get_peft_model(base_model, lora_config)
        self.model = prepare_model_for_kbit_training(peft_model, double_quant_config)


    def forward(self, inputs: Tensor ) -> Tensor:
        outputs = [self.model(inputs[:, idx])['last_hidden_state'] for idx in range(inputs.size(1))]
        return outputs


class CrossAugmentation(nn.Module):
    def __init__(self, attn_layer, mlp_layer, norm_layer):
        super().__init__()
        self.attn = attn_layer
        self.mlp = mlp_layer
        self.norm = norm_layer

    def forward(self, query: Tensor, encode_hidden_states: List[Tensor]) -> Tensor:
        for encode_hidden_state in encode_hidden_states:
            query = self.norm(query + self.attn(query, encode_hidden_state))
            query = self.norm(query + self.mlp(query))
        return query


class Classifier(nn.Module):
    def __init__(self, config: VQAConfig):
        super().__init__()
        self.config = config
        self.activation_fn = config.activation_fn
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x

class VQAModel(nn.Module):
    def __init__(self, config: VQAConfig = VQAConfig()):
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config
        self.language_config = config.language_config
        self.vision_model = VisionModel(self.vision_config)
        self.language_model = LanguageModel(self.language_config)

        self.language_query_attn = CrossAttention(encoder_hidden_size=self.vision_config.hidden_size,
                                                  num_heads=self.config.query_attn_heads,
                                                  hidden_size=self.language_config.hidden_size,
                                                  dropout=self.config.attn_dropout)

        self.vision_query_attn = CrossAttention(encoder_hidden_size=self.language_config.hidden_size,
                                                num_heads=self.config.query_attn_heads,
                                                hidden_size=self.vision_config.hidden_size,
                                                dropout=self.config.attn_dropout)

        self.lang_cross_augment = CrossAugmentation(
            attn_layer=self.language_query_attn,
            mlp_layer=MLP(self.language_config),
            norm_layer=nn.LayerNorm(self.language_config.hidden_size, eps=config.layer_norm_eps)
        )

        self.vision_cross_augment = CrossAugmentation(
            attn_layer=self.vision_query_attn,
            mlp_layer=MLP(self.vision_config),
            norm_layer=nn.LayerNorm(self.vision_config.hidden_size, eps=config.layer_norm_eps)
        )

        self.post_layernorm = nn.LayerNorm(self.language_config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = Classifier(self.config)

    def forward(self,
                text_inputs_lst: List[Dict[str, Tensor]],
                img_inputs_lst: Tensor
                ) -> Tensor:
        language_output = self.language_model(text_inputs_lst)
        vision_output = self.vision_model(img_inputs_lst)

        query_hidden_state = self.lang_cross_augment(language_output[0], vision_output)  
        encode_hidden_state = self.vision_cross_augment(vision_output[0], language_output)  

        for _ in range(self.config.query_layer):
            query_hidden_state = self.language_query_attn(query_hidden_state, encode_hidden_state)  

        pooled_output = query_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        logits = self.classifier(pooled_output)
        return logits

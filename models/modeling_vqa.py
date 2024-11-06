from typing import List, Dict
import math
import torch
from torch import nn, Tensor
from transformers import AutoModel
from .configuration import LanguageConfig, VisionConfig, VQAConfig

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, dropout) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, feat_inputs: Tensor) -> Tensor:
        bsz, num_feat, embed_dim = feat_inputs.size() 
        mixed_qkv = self.qkv(feat_inputs)
        mixed_qkv = mixed_qkv.reshape(bsz, num_feat, 3, self.num_heads, embed_dim // self.num_heads
                                      ).permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = attention_scores * self.scale

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.projection(context_layer)
        return output


class CrossAttention(nn.Module):
    def __init__(self, encoder_hidden_size, num_heads, hidden_size, dropout) -> None:
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(encoder_hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout)

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
        attention_probs_dropped = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs_dropped, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        output = context_layer.view(*new_context_layer_shape)
        return output


class LanguageModel(nn.Module):
    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(self.config.model_name)
        if self.config.frozen_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        if self.config.is_augment:
            self.self_attention = SelfAttention(self.config.hidden_size, 
                                                self.config.self_attn_heads, 
                                                self.config.self_attn_head_dim, 
                                                self.config.attn_dropout)
            self.cross_attention = CrossAttention(self.config.hidden_size, 
                                                  self.config.cross_attn_heads, 
                                                  self.config.hidden_size, 
                                                  self.config.attn_dropout)
            self.norm = nn.LayerNorm(self.config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, inputs: list, augment_thresh: float) -> Tensor:
        hidden_state = self.model(**inputs[0])['last_hidden_state'] 

        r_thresh = torch.rand(1)
        if self.training and self.config.is_augment and r_thresh < augment_thresh:
            for input_dict in inputs[1:]:
                encoder_hidden_state = self.model(**input_dict)['last_hidden_state']
                hidden_state = self.norm(hidden_state) + self.cross_attention(hidden_state, 
                                                                              encoder_hidden_state)
                hidden_state = self.norm(hidden_state) + self.self_attention(hidden_state)

        return hidden_state


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(self.config.model_name,
                                               attn_implementation="sdpa")
        if self.config.frozen_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        if self.config.is_augment:
            self.self_attention = SelfAttention(self.config.hidden_size, 
                                                self.config.self_attn_heads, 
                                                self.config.self_attn_head_dim, 
                                                self.config.attn_dropout)
            self.cross_attention = CrossAttention(self.config.hidden_size, 
                                                  self.config.cross_attn_heads, 
                                                  self.config.hidden_size, 
                                                  self.config.attn_dropout)
            self.norm = nn.LayerNorm(self.config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, 
                inputs: Tensor,
                augment_thresh: float,
                )-> Tensor:
        hidden_state = self.model(inputs[:, 0])["last_hidden_state"]
        
        r_thresh = torch.rand(1)
        if self.training and self.config.is_augment and r_thresh < augment_thresh:
            num_sample = inputs.size(1)
            for idx in range(num_sample):
                encoder_hidden_state = self.model(inputs[:, idx])['last_hidden_state']
                hidden_state = self.norm(hidden_state) + self.cross_attention(hidden_state,
                                                                              encoder_hidden_state)
                hidden_state = self.norm(hidden_state) + self.self_attention(hidden_state)

        return hidden_state
    

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
        self.query_attention = CrossAttention(encoder_hidden_size=self.vision_config.hidden_size, 
                                              num_heads=self.config.query_attn_heads, 
                                              hidden_size=self.language_config.hidden_size,
                                              dropout = self.config.attn_dropout)
        self.post_layernorm = nn.LayerNorm(self.language_config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = Classifier(self.config)

    def forward(self,
                text_inputs_lst: List[Dict[str, Tensor]],  
                img_inputs_lst: Tensor
                )-> Tensor:
        thresh = 1.
        language_thresh, vision_thresh = thresh, thresh
        language_output = self.language_model(text_inputs_lst, language_thresh) 
        vision_output = self.vision_model(img_inputs_lst, vision_thresh) 

        query_hidden_state = self.query_attention(language_output, vision_output)
        
        pooled_output = query_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        
        logits = self.classifier(pooled_output)
        return logits

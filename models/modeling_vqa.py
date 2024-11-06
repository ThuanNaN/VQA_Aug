from typing import List, Dict
import torch
from torch import nn, Tensor
from transformers import AutoModel
from .configuration import LanguageConfig, VisionConfig, VQAConfig


class BottleneckBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), 
            nn.ReLU(), 
            nn.Linear(input_dim // 2, input_dim), 
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.proj(x) + x
        x = self.norm(x)
        return x 

class SelfAttentionBlock(nn.Module):
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

    def forward(self, feat_inputs: List[Tensor]) -> Tensor:
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

        output = self.projection(context_layer)[:, 0, :] # get the first sample
        return self.norm(output)


class LanguageModel(nn.Module):
    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(self.config.model_name)
        # disable grad for model
        for param in self.model.parameters():
            param.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.projection_dim),
            nn.ReLU()
        )
        if self.config.is_augment:
            self.bottleneck = BottleneckBlock(self.config.hidden_size)
            self.self_attention = SelfAttentionBlock(self.config.hidden_size, 16, 64, 0.1)

    def forward(self, inputs: list, augment_thresh: float) -> Tensor:
        r_thresh = torch.rand(1)
        if self.training and self.config.is_augment and r_thresh < augment_thresh:
            embed_lst = [self.model(**input_dict)['last_hidden_state'][:, 0, :] 
                         for input_dict in inputs]
            para_features_t = torch.stack(embed_lst, dim=1)

            # x = torch.sum(para_features_t, dim=1)
            # x = self.bottleneck(x)
            x = self.self_attention(para_features_t)
        else:
            x = self.model(**inputs[0])
            x = x['last_hidden_state'][:, 0, :] # (batch, hidden_size)

        return self.projection(x)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(self.config.model_name,
                                               attn_implementation="sdpa")
        # disable grad for model except for deit.pooler.dense
        for name, param in self.model.named_parameters():
            param.requires_grad = 'pooler.dense' in name
            
        self.projection = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.projection_dim),
            nn.ReLU()
        )
        if self.config.is_augment:
            self.bottleneck = BottleneckBlock(self.config.hidden_size)
            self.self_attention = SelfAttentionBlock(self.config.hidden_size, 8, 96, 0.1)

    def forward(self, 
                inputs: Tensor,
                augment_thresh: float,
                )-> Tensor:
        r_thresh = torch.rand(1)
        if self.training and self.config.is_augment and r_thresh < augment_thresh:
            img_features_t = [self.model(inputs[:, idx])["pooler_output"]
                              for idx in range(inputs.size(1))]
            img_features_t = torch.stack(img_features_t, dim=1)

            # x = torch.sum(img_features_t, dim=1)
            # x = self.bottleneck(x)
            x = self.self_attention(img_features_t)
        else:
            x = self.model(inputs[:, 0])["pooler_output"] # (batch, hidden_size)

        return self.projection(x)
    

class MLP(nn.Module):
    def __init__(self, config: VQAConfig):
        super().__init__()
        self.config = config
        self.activation_fn = config.activation_fn
        self.fc1 = nn.Linear(config.hidden_size * 2, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.num_classes)

    def forward(self, 
                language_output: Tensor, 
                vision_output: Tensor
                ) -> Tensor:
        x = torch.cat((language_output, vision_output), 1)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x

class VQAModel(nn.Module):
    def __init__(self, config: VQAConfig = VQAConfig()):
        super().__init__()
        self.config = config
        self.vision_model = VisionModel(self.config.vision_config)
        self.language_model = LanguageModel(self.config.language_config)
        self.mlp = MLP(self.config)

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


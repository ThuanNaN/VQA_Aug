import torch
from typing import List
from torch import nn, Tensor
from transformers import AutoModel
import timm


class BottleneckBlock(nn.Module):
    def __init__(self, projection_dim, intermediate_dim):
        super().__init__()
        self.proj_in_ori = nn.Linear(projection_dim, intermediate_dim)
        self.proj_in_para = nn.Linear(projection_dim, intermediate_dim)
        self.proj_out = nn.Linear(intermediate_dim, projection_dim)
        self.relu = nn.ReLU()

    def forward(self, x_ori: Tensor, x_paras: List[Tensor]) -> Tensor:
        x = self.proj_in_ori(x_ori)
        for x_para in x_paras:
            x = x + self.proj_in_para(x_para)
        x = self.proj_out(x)
        x = self.relu(x) + x_ori
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_name, projection_dim, is_text_augment):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

        self.is_text_augment = is_text_augment
        self.hidden_size = self.model.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, projection_dim),
            nn.ReLU()
        )

        if self.is_text_augment:
            intermediate_dim = int(projection_dim*2)
            self.augment_linear = BottleneckBlock(projection_dim, intermediate_dim)

        self.norm = nn.LayerNorm(projection_dim)


    def forward(self, text_inputs_lst, augment_thresh):
        origin_text_inputs = text_inputs_lst[0]
        x_origin = self.model(**origin_text_inputs)
        x_origin = x_origin['last_hidden_state'][:, 0, :]
        x_origin = self.proj(x_origin)

        r = torch.rand(1)
        if self.training and self.is_text_augment and r < augment_thresh:
            x_para = []
            for text_inputs in text_inputs_lst[1:]:
                x = self.model(**text_inputs)
                x = x['last_hidden_state'][:, 0, :]
                x = self.proj(x)
                x_para.append(x)
            
            x_origin = self.augment_linear(x_origin, x_para)

        return self.norm(x_origin)


class ImageEncoder(nn.Module):
    def __init__(self, model_name, projection_dim):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0, 
        )

        hidden_size = self.model.num_features
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
        )

    def forward(self, img_inputs_lst):
        x = self.model(img_inputs_lst[0])
        x = self.proj(x)
        return x


class Classifier(nn.Module):
    def __init__(self, projection_dim, hidden_dim, answer_space):
        super().__init__()
        self.fc = nn.Linear(projection_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, answer_space)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, text_f, img_f):
        x = torch.cat((img_f, text_f), 1)
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class Light_ViVQAModel(nn.Module):
    def __init__(self, 
                 lang_model_name,
                 vis_model_name,
                 projection_dim, 
                 hidden_dim, 
                 answer_space_len,
                 is_text_augment=True,
                 use_dynamic_thresh=True,
                 text_para_thresh=0.6):
        super().__init__()
        self.text_encoder = TextEncoder(
            model_name=lang_model_name,
                    projection_dim=projection_dim,
                                        is_text_augment=is_text_augment)

        self.img_encoder = ImageEncoder(
            model_name=vis_model_name,
            projection_dim=projection_dim)

        self.classifier = Classifier(projection_dim=projection_dim,
                                     hidden_dim=hidden_dim,
                                     answer_space=answer_space_len)

        self.use_dynamic_thresh = use_dynamic_thresh
        self.text_para_thresh = text_para_thresh

    
    def forward(self, text_inputs: list, img_inputs: list):

        text_thresh = self.text_para_thresh

        text_f = self.text_encoder(text_inputs, text_thresh)
        img_f = self.img_encoder(img_inputs)

        logits = self.classifier(text_f, img_f)
        return logits

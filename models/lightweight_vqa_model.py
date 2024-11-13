import torch
from typing import List
from torch import nn, Tensor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from transformers import AutoModel
import timm


double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

class BottleneckBlock(nn.Module):
    def __init__(self, input_dim, intermediate_dim):
        super().__init__()
        self.proj_in_ori = nn.Linear(input_dim, intermediate_dim)
        self.proj_in_para = nn.Linear(input_dim, intermediate_dim)
        self.proj_out = nn.Linear(intermediate_dim, input_dim)
        self.norm = nn.LayerNorm(intermediate_dim)

    def forward(self, x_ori: Tensor, X_para: List[Tensor]) -> Tensor:
        x = self.proj_in_ori(x_ori)
        for x_para in X_para:
            x = self.norm(x + self.proj_in_para(x_para))
        return self.proj_out(x)


class TextEncoder(nn.Module):
    def __init__(self, projection_dim, is_text_augment):
        super().__init__()
        base_model = AutoModel.from_pretrained("vinai/bartpho-word",
                                               quantization_config=double_quant_config,
                                               low_cpu_mem_usage=True)
        base_model.config.use_cache = False
        quantized_model = prepare_model_for_kbit_training(base_model, double_quant_config)
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["k_proj", "v_proj", "q_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none")
        self.model  = get_peft_model(quantized_model, lora_config)
        self.model.print_trainable_parameters()

        self.is_text_augment = is_text_augment
        self.hidden_size = self.model.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, projection_dim),
            nn.ReLU()
        )

        if self.is_text_augment:
            input_dim = self.hidden_size
            intermediate_dim = self.hidden_size // 2
            self.augment_linear = BottleneckBlock(input_dim, intermediate_dim)


    def forward(self, text_inputs_lst, augment_thresh):
        r = torch.rand(1)
        if self.training and self.is_text_augment and r < augment_thresh:

            embed_lst = []
            for text_inputs in text_inputs_lst:
                x = self.model(**text_inputs)
                x = x['last_hidden_state'][:, 0, :]
                embed_lst.append(x)
            x = self.augment_linear(embed_lst[0], embed_lst[1:])
        else:
            text_inputs = text_inputs_lst[0]
            x = self.model(**text_inputs)
            x = x['last_hidden_state'][:, 0, :]

        return self.proj(x)


class ImageEncoder(nn.Module):
    def __init__(self, projection_dim):
        super().__init__()
        base_model = timm.create_model(
            'beitv2_base_patch16_224.in1k_ft_in22k',
            pretrained=True,
            num_classes=0, 
        )
        quantized_model = prepare_model_for_kbit_training(base_model, double_quant_config)
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["qkv", "proj"],
            lora_dropout=0.1,
            bias="none")
        self.model  = get_peft_model(quantized_model, lora_config)
        self.model.print_trainable_parameters()

        hidden_size = self.model.num_features
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU()
        )

    def forward(self, img_inputs_lst):
        x = self.model(img_inputs_lst[:, 0])
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
                 projection_dim, 
                 hidden_dim, 
                 answer_space_len,
                 is_text_augment=True,
                 use_dynamic_thresh=True,
                 text_para_thresh=0.6):
        super().__init__()

        self.text_encoder = TextEncoder(projection_dim=projection_dim,
                                        is_text_augment=is_text_augment)

        self.img_encoder = ImageEncoder(projection_dim=projection_dim)

        self.classifier = Classifier(projection_dim=projection_dim,
                                     hidden_dim=hidden_dim,
                                     answer_space=answer_space_len)

        self.use_dynamic_thresh = use_dynamic_thresh
        self.text_para_thresh = text_para_thresh

    
    def forward(self, text_inputs, img_inputs):
        text_thresh = self.text_para_thresh

        text_f = self.text_encoder(text_inputs, text_thresh)
        img_f = self.img_encoder(img_inputs)

        logits = self.classifier(text_f, img_f)
        return logits

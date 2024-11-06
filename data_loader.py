import ast
import random
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from utils import DataPath
from torchvision import transforms


class ViVQADataset(Dataset):
    def __init__(self,
                    data_path,
                    data_mode, 
                    text_processor, 
                    image_processor, 
                    label_encoder=None, 
                    is_text_augment=False, 
                    n_text_paras=1, 
                    is_img_augment=False,
                    n_img_augments=1, 
                    )-> None:
        self.vivqa_dir = DataPath.ViVQA_PATH
        self.data_path = data_path
        self.data_mode = data_mode
        self.is_text_augment = is_text_augment
        self.is_img_augment = is_img_augment
        self.label_encoder = label_encoder
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.n_text_paras = n_text_paras
        self.n_img_augments = n_img_augments

        # Random augmentations for images
        self.aug_transforms = transforms.AutoAugment(
            policy=transforms.AutoAugmentPolicy.IMAGENET,
            interpolation=transforms.InterpolationMode.BICUBIC
        )

        self.questions, self.para_questions, self.img_paths, self.answers = self.get_data()


    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        img_path = self.img_paths[idx]

        img_pil = Image.open(img_path).convert('RGB')
        label = self.label_encoder[answer]
        
        img_pil_lst = [img_pil]
        if self.data_mode == 'train' and self.is_img_augment:
            augmented_imges = self.augment_images(img_pil, self.n_img_augments) 
            img_pil_lst.extend(augmented_imges)
        img_inputs_lst = self.image_processor(img_pil_lst, return_tensors='pt').pixel_values

        text_inputs_lst = [self.text_processor(question)]
        
        if self.data_mode == 'train' and self.is_text_augment:
            para_questions = ast.literal_eval(self.para_questions[idx])
            selected_para_questions = random.sample(para_questions, self.n_text_paras)
            paraphrase_inputs_lst = [self.text_processor(text) for text in selected_para_questions]
            text_inputs_lst.extend(paraphrase_inputs_lst)
        
        return {
            'text_inputs': text_inputs_lst, # List[Dict]
            'img_inputs': img_inputs_lst, # List[Tensor]
            'labels': torch.tensor(label, dtype=torch.long) # Tensor
        }

    def __len__(self):
        return len(self.questions)
        
    def get_data(self):
        df = pd.read_csv(self.data_path, index_col=0)
        questions, para_questions, answers, img_paths = [], [], [], []
        if self.data_mode == 'train':
            para_questions = df["question_paraphrase"].values.tolist()

        for _, row in df.iterrows():
            img_path = self.vivqa_dir / 'images' / f'{row["img_id"]:012}.jpg'
            img_paths.append(img_path)
            questions.append(row['question'])
            answers.append(row['answer'])

        return questions, para_questions, img_paths, answers 
    

    def augment_images(self, origin_image, n_img_augments):
        """
        Augment images using torchvision transforms.

        Args:
            origin_image (PIL.Image): Original image
            n_img_augments (int): Number of augmented images

        Returns:
            list: List of augmented images
        """
        return [self.aug_transforms(origin_image) for _ in range(n_img_augments)]

import argparse
from dataclasses import dataclass
from utils import seed_everything, get_label_encoder
import torch
from torch import nn
from data_loader import ViVQADataset
from torch.utils.data import DataLoader
from models import VisionModel, LanguageModel, VQAModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class BaseTrainingConfig:
    seed: int = 59
    train_data_path: str = "./datasets/vivqa/20_filtered_question_paraphrases.csv"
    val_data_path: str = "./datasets/vivqa/test.csv"
    is_text_augment: bool = True
    n_text_paras: int = 1
    is_img_augment: bool = True
    n_img_augments: int = 1
    train_batch_size: int = 4
    val_batch_size: int = 8
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-5


def train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        epochs,
        device,
        use_amp=True,
        load_best_model=False,
):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        for batch in train_loader:
            # (n_aug, batch, length)
            text_inputs_lst = [{k: v.squeeze().to(device, non_blocking=True) for k, v in input_ids.items()} 
                               for input_ids in batch['text_inputs']]
            # (batch, n_aug, C, H, W)
            img_inputs_lst = batch['img_inputs'].to(device, non_blocking=True)
            # (batch)
            labels = batch['labels'].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits = model(text_inputs_lst, img_inputs_lst)
                loss = criterion(logits, labels)
                print(loss.item())

            break

        break   



def main():
    base_config = BaseTrainingConfig()
    parser = argparse.ArgumentParser(description='ViVQA Training Script')
    parser.add_argument('--seed',
                        type=int,
                        default=base_config.seed,
                        help='Random seed')
    parser.add_argument('--train_data_path',
                        type=str,
                        default=base_config.train_data_path,
                        help='Path to training data')
    parser.add_argument('--val_data_path',
                        type=str,
                        default=base_config.val_data_path,
                        help='Path to validation data')
    parser.add_argument('--is_text_augment',
                        type=bool,
                        default=base_config.is_text_augment,
                        help='Text augmentation')
    parser.add_argument('--n_text_paras',
                        type=int,
                        default=base_config.n_text_paras,
                        help='Number of text paraphrases')
    parser.add_argument('--is_img_augment',
                        type=bool,
                        default=base_config.is_img_augment,
                        help='Image augmentation')
    parser.add_argument('--n_img_augments',
                        type=int,
                        default=base_config.n_img_augments,
                        help='Number of image augmentations')
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=base_config.train_batch_size,
                        help='Training batch size')
    parser.add_argument('--val_batch_size',
                        type=int,
                        default=base_config.val_batch_size,
                        help='Validation batch size')
    parser.add_argument('--epochs',
                        type=int,
                        default=base_config.epochs,
                        help='Number of epochs')
    parser.add_argument('--lr',
                        type=float,
                        default=base_config.lr,
                        help='Learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=base_config.weight_decay,
                        help='Weight decay')

    args = parser.parse_args()

    seed_everything(args.seed)

    label2idx, idx2label, answer_space_len = get_label_encoder()
    text_processor = LanguageModel.get_text_processor()
    image_processor = VisionModel.get_img_processor()

    train_dataset = ViVQADataset(data_path=args.train_data_path,
                                 data_mode="train",
                                 text_processor=text_processor,
                                 image_processor=image_processor,
                                 label_encoder=label2idx,
                                 is_text_augment=args.is_text_augment,
                                 n_text_paras=args.n_text_paras,
                                 is_img_augment=args.is_img_augment,
                                 n_img_augments=args.n_img_augments,
                                 )

    test_dataset = ViVQADataset(data_path=args.val_data_path,
                                data_mode="val",
                                text_processor=text_processor,
                                image_processor=image_processor,
                                label_encoder=label2idx,
                                )

    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              pin_memory=True,
                              shuffle=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.val_batch_size,
                             pin_memory=True,
                             shuffle=False)
    
    model = VQAModel()
    print("Total parameters: ", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.lr, 
                                 weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_model(model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=None,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                device=device,
                load_best_model=True
                )


if __name__ == '__main__':
    main()

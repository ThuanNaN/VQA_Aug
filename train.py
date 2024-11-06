import argparse
from dataclasses import dataclass
from utils import (
    seed_everything, 
    get_label_encoder, 
    colorstr, 
    save_model_ckpt
)
from tqdm import tqdm
import time
import os
import logging
import copy
import wandb
import torch
from torch import nn
from data_loader import ViVQADataset
from torch.utils.data import DataLoader
from models import VQAModel, VQAProcessor

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("PyTorch-CLS")

@dataclass
class BaseTrainingConfig:
    seed: int = 59
    train_data_path: str = "./datasets/vivqa/20_filtered_question_paraphrases.csv"
    val_data_path: str = "./datasets/vivqa/test.csv"
    is_text_augment: bool = True
    n_text_paras: int = 2
    is_img_augment: bool = True
    n_img_augments: int = 2
    train_batch_size: int = 32
    val_batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = True
    wandb_log: bool = False
    wandb_name: str = "ViVQA_Aug"
    run_name: str = "exp"
    save_ckpt: bool = True
    device_ids: int = 0


def train_model(
        model,
        criterion,
        optimizer,
        scaler,
        lr_scheduler,
        dataloaders,
        epochs,
        device,
        use_amp=False,
        args=None,
):
    since = time.perf_counter()
    LOGGER.info(f"\n{colorstr('Device:')} {device}")
    LOGGER.info(f"\n{colorstr('Optimizer:')} {optimizer}")

    if lr_scheduler:
        LOGGER.info(
            f"\n{colorstr('LR Scheduler:')} {type(lr_scheduler).__name__}")
    
    LOGGER.info(f"\n{colorstr('Loss:')} {type(criterion).__name__}")

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_optim = copy.deepcopy(optimizer.state_dict())
    best_val_acc = 0.0

    for epoch in range(epochs):
        LOGGER.info(colorstr(f'\nEpoch {epoch}/{epochs-1}:'))
        for phase in ["train", "val"]:
            if phase == "train":
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) %
                            ('Training:', 'gpu_mem', 'loss', 'acc'))
                model.train()
            else:
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) %
                            ('Validation:', 'gpu_mem', 'loss', 'acc'))
                model.eval()
            running_items = 0
            running_loss = 0.0
            running_corrects = 0
            _phase = tqdm(dataloaders[phase],
                          total=len(dataloaders[phase]),
                          bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                          unit='batch')

            for batch in _phase:
                text_inputs_lst = [
                    {
                        k: v.squeeze().to(device, non_blocking=True)
                        for k, v in input_ids.items()
                    }
                    for input_ids in batch['text_inputs']
                ]  # (n_aug, batch, length)
                img_inputs_lst = batch['img_inputs'].to(device, non_blocking=True)  # (batch, n_aug, C, H, W)
                labels = batch['labels'].to(device, non_blocking=True)  # (batch)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        logits = model(text_inputs_lst, img_inputs_lst)
                        loss = criterion(logits, labels)

                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            history['lr'].append(lr_scheduler.optimizer.param_groups[0]
                                ["lr"]) if lr_scheduler else history['lr'].append(args.lr)
                            if lr_scheduler is not None:
                                lr_scheduler.step()

                _, preds = torch.max(logits, 1)
                running_items += labels.size(0)
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / running_items
                epoch_acc = running_corrects / running_items
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                desc = ('%35s' + '%15.6g' * 2) % (mem, running_loss /
                                                  running_items, running_corrects / running_items)
                _phase.set_description_str(desc)

            if phase == 'train':
                if args.wandb_log:
                    wandb.log({"train_acc": epoch_acc, "train_loss": epoch_loss}, step = epoch)
                    if lr_scheduler:
                        wandb.log(
                            {"lr": lr_scheduler.optimizer.param_groups[0]["lr"]}, step = epoch)
                    else:
                        wandb.log({"lr": args.lr})
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                if args.wandb_log:
                    wandb.log({"val_acc": epoch_acc, "val_loss": epoch_loss}, step = epoch)
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # best_model_optim = copy.deepcopy(optimizer.state_dict())
                    if args.save_ckpt:
                        # save_model_ckpt(model, DIR_SAVE, "best.pt")
                        pass

    time_elapsed = time.perf_counter() - since
    LOGGER.info(f"Training complete in {time_elapsed // 3600}h {time_elapsed % 3600 // 60}m { time_elapsed % 60}s with {epochs} epochs")
    LOGGER.info(f"Best val Acc: {round(best_val_acc.item(), 6)}")

    model.load_state_dict(best_model_wts)
    return model, best_val_acc.item()


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
    parser.add_argument('--use_amp',
                        type=bool,
                        default=base_config.use_amp,
                        help='Automatic Mixed Precision (AMP)')
    parser.add_argument('--wandb_log',
                        type=bool,
                        default=base_config.wandb_log,
                        help='Log to Weights and Biases')
    parser.add_argument('--save_ckpt',
                        type=bool,
                        default=base_config.save_ckpt,
                        help='Save best checkpoint')
    parser.add_argument('--wandb_name',
                        type=str,
                        default=base_config.wandb_name,
                        help='Weights and Biases project name')
    parser.add_argument('--run_name',   
                        type=str,
                        default=base_config.run_name,
                        help='Weights and Biases run name')
    parser.add_argument('--device_ids',
                        type=int,
                        default=base_config.device_ids,
                        help='Number of device ids')
    args = parser.parse_args()

    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_ids)

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    try:
        device_name = os.getlogin()
    except:
        device_name = "Colab/Cloud"
    if args.wandb_log:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project=args.wandb_name,
            name=args.run_name,
            tags=[device_name],
            config=vars(args))
    else:
        wandb = None

    label2idx, idx2label, answer_space_len = get_label_encoder()
    text_processor = VQAProcessor().get_text_processor()
    image_processor = VQAProcessor().get_img_processor()

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
    dataloaders = {
        "train": train_loader,
        "val": test_loader
    }

    model = VQAModel().to(device)
    model.config.num_classes = answer_space_len

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"Total Parameters: {total_params // 1e6}M")
    LOGGER.info(f"Trainable Parameters: {trainable_params // 1e6}M")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=args.use_amp)
    best_model, best_val_acc = train_model(model=model,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        scaler=scaler,
                                        lr_scheduler=None,
                                        dataloaders=dataloaders,
                                        epochs=args.epochs,
                                        device=device,
                                        args=args)

    if args.wandb_log:
        wandb.finish()

if __name__ == '__main__':
    main()

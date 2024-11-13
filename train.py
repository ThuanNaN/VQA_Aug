import argparse
import yaml
from dataclasses import dataclass
from utils import (
    seed_everything, 
    get_label_encoder, 
    colorstr, 
    save_model_ckpt,
    plot_and_log_result,
    set_threads,
    DataPath
)
from tqdm import tqdm
import time
import os
import logging
import copy
import wandb
import torch
from torch import nn, optim
import pytorch_warmup as warmup
from data_loader import ViVQADataset
from torch.utils.data import DataLoader
from models import VQAModel, VQAProcessor, Light_ViVQAModel

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("VQA Training")

@dataclass
class BaseTrainingConfig:
    seed: int = 59
    train_data_path: str = "./datasets/vivqa/20_filtered_question_paraphrases.csv"
    val_data_path: str = "./datasets/vivqa/test.csv"
    is_text_augment: bool = True
    n_text_paras: int = 2
    is_img_augment: bool = True
    n_img_augments: int = 2
    train_batch_size: int = 64
    val_batch_size: int = 64
    epochs: int = 50
    patience: int = 5
    lr: float = 1e-4
    use_scheduler: bool = True
    warmup_steps: int = 500
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    use_amp: bool = False
    wandb_log: bool = False
    wandb_name: str = "ViVQA_Aug"
    log_result: bool = True
    run_name: str = "exp"
    save_ckpt: bool = True
    num_workers: int = 4


def train_model(
        model,
        scaler,
        criterion,
        optimizer,
        lr_scheduler,
        warmup_scheduler,
        dataloaders,
        epochs,
        device,
        use_amp=True,
        args=None,
):
    since = time.perf_counter()

    if args.log_result:
        DIR_SAVE = DataPath.RUN_TRAIN_DIR/f"{args.run_name}/run_seed_{args.seed}"
        if not DIR_SAVE.exists():
            DIR_SAVE.mkdir(parents=True, exist_ok=True)
        save_opt = os.path.join(DIR_SAVE, "opt.yaml")
        with open(save_opt, 'w') as f:
            yaml.dump(args.__dict__, f, sort_keys=False)

    LOGGER.info(f"\n{colorstr('Device:')} {device}")
    LOGGER.info(f"\n{colorstr('Optimizer:')} {optimizer}")
    if lr_scheduler:
        LOGGER.info(
            f"\n{colorstr('LR Scheduler:')} {type(lr_scheduler).__name__}")
    if warmup_scheduler:
        LOGGER.info(
            f"\n{colorstr('Warmup Scheduler:')} {type(warmup_scheduler).__name__}")
    
    LOGGER.info(f"\n{colorstr('Loss:')} {type(criterion).__name__}")

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"\n{colorstr('Total Parameters:')} {total_params / 1e6}M")
    LOGGER.info(f"\n{colorstr('Trainable Parameters:')} {trainable_params / 1e6}M")

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
            running_patience = 0
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
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        logits = model(text_inputs_lst, img_inputs_lst)
                        loss = criterion(logits, labels)
                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            
                            batch_lr = optimizer.param_groups[0]["lr"]
                            history['lr'].append(batch_lr)
                            
                            if lr_scheduler is not None:
                                with warmup_scheduler.dampening():
                                    if warmup_scheduler.last_step + 1 >= args.warmup_steps:
                                        lr_scheduler.step()

                _, preds = torch.max(logits, 1)
                running_items += labels.size(0)
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / running_items
                epoch_acc = running_corrects / running_items

                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                desc = ('%35s' + '%15.6g' * 2) % (mem, running_loss/running_items, running_corrects/running_items)
                _phase.set_description_str(desc)
                _phase.set_postfix_str(f"step: {warmup_scheduler.last_step}, lr: {batch_lr}")

            if phase == 'train':
                if args.wandb_log:
                    wandb.log({"train_acc": epoch_acc, "train_loss": epoch_loss}, step = epoch)
                    if lr_scheduler:
                        wandb.log(
                            {"lr": batch_lr}, step = epoch)
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
                    if args.save_ckpt:
                        save_model_ckpt(model, DIR_SAVE, "best.pt")
                else:
                    running_patience += 1
                    if running_patience > args.patience:
                        LOGGER.info(f"Early stopping at epoch {epoch}")
                        LOGGER.info(f"Best val Acc: {round(best_val_acc.item(), 6)}")

    if args.save_ckpt:
        save_model_ckpt(model, DIR_SAVE, "last.pt")
    if args.save_ckpt:
        LOGGER.info(f"Best model weight saved at {DIR_SAVE}/weights/best.pt")
        LOGGER.info(f"Last model weight saved at {DIR_SAVE}/weights/last.pt")

    if args.log_result:
        plot_and_log_result(DIR_SAVE, history)
    
    time_elapsed = time.perf_counter() - since
    LOGGER.info(f"Training complete in {time_elapsed // 3600}h {time_elapsed % 3600 // 60}m {time_elapsed % 60//1}s with {epochs} epochs")
    LOGGER.info(f"Best val Acc: {round(best_val_acc.item(), 6)}")

    model.load_state_dict(best_model_wts)
    return model, best_val_acc.item(), history


def main():
    base_config = BaseTrainingConfig()
    parser = argparse.ArgumentParser(description='ViVQA Training Script')
    parser.add_argument('--seed', type=int, default=base_config.seed, help='Random seed')
    parser.add_argument('--train_data_path', type=str, default=base_config.train_data_path, help='Path to training data')
    parser.add_argument('--val_data_path', type=str, default=base_config.val_data_path, help='Path to validation data')
    parser.add_argument('--is_text_augment', type=bool, default=base_config.is_text_augment, help='Text augmentation')
    parser.add_argument('--n_text_paras', type=int, default=base_config.n_text_paras, help='Number of text paraphrases')
    parser.add_argument('--is_img_augment', type=bool, default=base_config.is_img_augment, help='Image augmentation')
    parser.add_argument('--n_img_augments', type=int, default=base_config.n_img_augments, help='Number of image augmentations')
    parser.add_argument('--train_batch_size', type=int, default=base_config.train_batch_size, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=base_config.val_batch_size, help='Validation batch size')
    parser.add_argument('--epochs', type=int, default=base_config.epochs, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=base_config.patience, help='Patience for early stopping')
    parser.add_argument('--lr', type=float, default=base_config.lr, help='Learning rate')
    parser.add_argument('--use_scheduler', type=bool, default=base_config.use_scheduler, help='Use learning rate scheduler')
    parser.add_argument('--lr_min', type=float, default=base_config.lr_min, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=base_config.weight_decay, help='Weight decay')
    parser.add_argument('--use_amp', type=bool, default=base_config.use_amp, help='Automatic Mixed Precision (AMP)')
    parser.add_argument('--wandb_log', type=bool, default=base_config.wandb_log, help='Log to Weights and Biases')
    parser.add_argument('--save_ckpt', type=bool, default=base_config.save_ckpt, help='Save best checkpoint')
    parser.add_argument('--wandb_name', type=str, default=base_config.wandb_name, help='Weights and Biases project name')
    parser.add_argument('--log_result', type=bool, default=base_config.log_result, help='Log training results')
    parser.add_argument('--run_name', type=str, default=base_config.run_name, help='Weights and Biases run name')
    parser.add_argument('--warmup_steps', type=int, default=base_config.warmup_steps, help='Warmup steps')
    parser.add_argument('--num_workers', type=int, default=base_config.num_workers, help='Number of workers')
    args = parser.parse_args()

    seed_everything(args.seed)
    set_threads(args.num_workers)

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
                                 n_img_augments=args.n_img_augments)

    test_dataset = ViVQADataset(data_path=args.val_data_path,
                                data_mode="val",
                                text_processor=text_processor,
                                image_processor=image_processor,
                                label_encoder=label2idx)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              pin_memory=True,
                              shuffle=True,
                              num_workers=args.num_workers)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.val_batch_size,
                             pin_memory=True,
                             shuffle=False,
                             num_workers=args.num_workers)

    dataloaders = {
        "train": train_loader,
        "val": test_loader
    }

    # model = VQAModel()
    model = Light_ViVQAModel(
        projection_dim = 512,
        hidden_dim = 512,
        answer_space_len = answer_space_len,
        is_text_augment = False,
        use_dynamic_thresh = False,
        text_para_thresh = 0.6
    )

    total_steps = len(train_loader) * args.epochs
    num_steps = total_steps - args.warmup_steps

    optimizer = optim.Adam(model.parameters(),
                     lr=args.lr,
                     weight_decay=args.weight_decay)
    
    lr_scheduler, warmup_scheduler = None, None
    if args.use_scheduler:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=num_steps,
                                                            eta_min=args.lr_min)
        warmup_scheduler = warmup.LinearWarmup(optimizer, args.warmup_steps)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=args.use_amp)
    best_model, best_val_acc, history = train_model(model=model,
                                        scaler=scaler,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        lr_scheduler=lr_scheduler,
                                        warmup_scheduler=warmup_scheduler,
                                        dataloaders=dataloaders,
                                        epochs=args.epochs,
                                        device=device,
                                        args=args)

    if args.wandb_log:
        wandb.finish()

if __name__ == '__main__':
    main()

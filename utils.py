from pathlib import Path
import os
import copy
import torch
class DataPath:
    DATASET_DIR = Path('datasets')
    ViVQA_PATH = DATASET_DIR / 'vivqa'

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_label_encoder():
    save_path = DataPath.ViVQA_PATH / 'answer_space.txt'
    with open(save_path, 'r') as f:
        lines = f.read().splitlines()

    label2idx = {label: idx for idx, label in enumerate(lines)}
    idx2label = {idx: label for idx, label in enumerate(lines)}
    answer_space_len = len(lines)

    return label2idx, idx2label, answer_space_len


def save_model_ckpt(model, path_dir, ckpt_name='best.pt'):
    path_save = os.path.join(path_dir, "weights")
    os.makedirs(path_save, exist_ok=True)
    model_ckpt = copy.deepcopy(model.state_dict())
    models_ckpt = {
        "model_state_dict": model_ckpt,
    }
    torch.save(models_ckpt,os.path.join(path_save, ckpt_name))


def load_model_ckpt(ckpt_path, model, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0]) 
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def emojis(str=''):
    import platform
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str

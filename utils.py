from pathlib import Path

class AppPath:
    DATA_PATH = Path('datasets')
    ViVQA_PATH = DATA_PATH / 'vivqa'

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
    save_path = AppPath.ViVQA_PATH / 'answer_space.txt'
    with open(save_path, 'r') as f:
        lines = f.read().splitlines()

    label2idx = {label: idx for idx, label in enumerate(lines)}
    idx2label = {idx: label for idx, label in enumerate(lines)}
    answer_space_len = len(lines)

    return label2idx, idx2label, answer_space_len
import csv
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        pred = logits.argmax(dim=-1)

        y = batch.y.view(-1)
        correct += int((pred == y).sum())
        total += y.numel()

    return correct / total


def summarize(values):
    x = torch.tensor(values, dtype=torch.float)
    if x.numel() == 1:
        return x.item(), 0.0
    return x.mean().item(), x.std(unbiased=False).item()


def save_results_csv(path, rows):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fieldnames = [
        'method',
        'row_type',
        'run',
        'accuracy',
        'train_time',
        'acc_mean',
        'acc_std',
        'time_mean',
        'time_std',
    ]

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
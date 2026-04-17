import time

import torch

from utils import get_targets

def train_one_epoch(model, loader, optimizer, device, loss_fn, task):
    model.train()
    total_loss = 0.0

    start = time.perf_counter()

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        logits, aux_loss = model(batch)
        y = get_targets(batch, task)

        if task == "regression":
            y = y.view(-1, 1).float()
            logits = logits.view(-1, 1)
        loss = loss_fn(logits, y) + aux_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.numel()

    train_time = time.perf_counter() - start
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, train_time

@torch.no_grad()
def evaluate(model, loader, device, task):
    model.eval()

    ys, preds = [], []

    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        y = batch.y

        if task == "classification":
            pred = logits.argmax(dim=-1)

        elif task == "binary":
            pred = torch.sigmoid(logits).view(-1)

        elif task == "multilabel":
            pred = torch.sigmoid(logits)

        elif task == "regression":
            pred = logits.view_as(y)

        else:
            raise ValueError(f"Unknown task: {task}")

        ys.append(y.detach().cpu())
        preds.append(pred.detach().cpu())

    y = torch.cat(ys)
    pred = torch.cat(preds)

    if task == "classification":
        return (pred.view(-1) == y.view(-1)).float().mean().item()

    elif task == "binary":
        pred_label = (pred.view(-1) > 0.5).float()
        return (pred_label == y.view(-1)).float().mean().item()

    elif task == "multilabel":
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y.numpy(), pred.numpy(), average="macro")

    elif task == "regression":
        return torch.mean(torch.abs(pred - y)).item()
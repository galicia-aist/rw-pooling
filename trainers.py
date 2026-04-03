import time
import torch.nn.functional as F

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    start = time.perf_counter()

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        logits, aux_loss = model(batch)
        y = batch.y.view(-1)

        loss = F.cross_entropy(logits, y) + aux_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.numel()

    train_time = time.perf_counter() - start
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, train_time
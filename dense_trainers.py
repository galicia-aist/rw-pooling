import torch.nn.functional as F


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _, _ = model(data.x, data.adj, data.mask)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        total_loss += data.y.size(0) * float(loss)
        optimizer.step()
    return total_loss / len(train_loader.dataset)
def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output, _, _ = model(data.x, data.adj, data.mask)
        pred = output.max(dim=1)[1]
        correct += int(pred.eq(data.y.view(-1)).sum())
    return correct / len(loader.dataset)
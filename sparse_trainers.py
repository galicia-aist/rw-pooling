import torch.nn.functional as F
from ogb.graphproppred import Evaluator
import torch
from sklearn.metrics import average_precision_score


def train(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)
def test(loader, model, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)


def train_ogb(model, optimizer, train_loader, device, criterion):
    model.train()
    y_true = []
    y_pred = []
    total_loss = 0.
    num_g_total = 0.
    for data in train_loader:
        if data.x.shape[0] == 1 or data.batch[-1] == 0:
            continue
        num_g = data.num_graphs
        num_g_total += num_g
        data = data.to(device)
        out = model(data)
        pred = out
        is_labeled = data.y == data.y
        loss = torch.nn.BCEWithLogitsLoss()(pred.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
        total_loss += loss.item() * num_g
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        y_true.append(data.y.view(pred.shape)[is_labeled].detach().cpu())
        y_pred.append(torch.sigmoid(pred)[is_labeled].detach().cpu())
    total_loss = total_loss / num_g_total
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    metric = average_precision_score(y_true, y_pred)
    return total_loss, metric


@torch.no_grad()
def test_ogb(loader, model, device):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.
    num_g_total = 0.
    for data in loader:
        if data.x.shape[0] == 1:
            continue
        num_g = data.num_graphs
        num_g_total += num_g
        data = data.to(device)
        out = model(data)
        pred = out
        is_labeled = data.y == data.y
        loss = torch.nn.BCEWithLogitsLoss()(pred.to(torch.float32)[is_labeled], data.y.to(torch.float32)[is_labeled])
        total_loss += loss.item() * num_g
        y_true.append(data.y.view(pred.shape)[is_labeled].detach().cpu())
        y_pred.append(torch.sigmoid(pred)[is_labeled].detach().cpu())
    total_loss = total_loss / num_g_total
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    metric = average_precision_score(y_true, y_pred)
    return total_loss, metric
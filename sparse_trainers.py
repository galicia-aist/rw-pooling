import torch.nn.functional as F
from ogb.graphproppred import Evaluator
import torch

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
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        # Handle missing labels (important for molpcba)
        is_labeled = data.y == data.y

        loss = criterion(
            out[is_labeled],
            data.y.float()[is_labeled]
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test_ogb(loader, model, device, dataset_name):
    model.eval()
    evaluator = Evaluator(name=dataset_name)
    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        out = model(data)
        y_true.append(data.y.view(out.shape).detach().cpu())
        y_pred.append(out.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {
        "y_true": y_true,
        "y_pred": y_pred,
    }

    result = evaluator.eval(input_dict)
    metric_name = evaluator.eval_metric
    return result[metric_name]
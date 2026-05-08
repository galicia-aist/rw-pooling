import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
import numpy as np
import random
import time
import warnings
warnings.filterwarnings("ignore")
max_nodes = 150
dataset_sparse = PygGraphPropPredDataset(root='data', name="ogbg-molpcba")
evaluator = Evaluator("ogbg-molpcba")
eval_metric = dataset_sparse.eval_metric
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
dataset_sparse = dataset_sparse.shuffle()
num_total = len(dataset_sparse)
num_train = int(num_total * train_ratio)
num_val = int(num_total * val_ratio)
num_test = num_total - num_train - num_val
train_dataset = dataset_sparse[:num_train]
val_dataset = dataset_sparse[num_train:num_train + num_val]
test_dataset = dataset_sparse[num_train + num_val:]
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)



import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from sklearn.metrics import average_precision_score
class HierarchicalGCN_TOPK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_TOPK, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, improved = True)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.3)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, improved = True)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.3)
        self.conv3 = GCNConv(hidden_channels, out_channels, improved = True)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        self.lin1 = torch.nn.Linear(out_channels, 128)
        self.lin2 = torch.nn.Linear(128, 128)
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        init_x = data.x
        x = x.float()
        init_x = init_x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch=batch)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch=batch)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
num_classes = dataset_sparse.num_classes
in_channels = dataset_sparse.num_features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HierarchicalGCN_TOPK(in_channels=dataset_sparse.num_features, hidden_channels=512,out_channels=256, num_classes=dataset_sparse.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.BCEWithLogitsLoss()
def train():
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
def test(loader):
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
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seeds = [42, 43, 44]
times = []
memories = []
best_val_accs = []
best_test_accs = []
early_stop_patience = 50
tolerance = 0.0001
for seed in seeds:
    set_seed(seed)
    dataset_sparse = dataset_sparse.shuffle()
    num_total = len(dataset_sparse)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)
    num_test = num_total - num_train - num_val
    train_dataset = dataset_sparse[:num_train]
    val_dataset = dataset_sparse[num_train:num_train + num_val]
    test_dataset = dataset_sparse[num_train + num_val:]
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
    model = HierarchicalGCN_TOPK(in_channels=dataset_sparse.num_features, hidden_channels=512,out_channels=256, num_classes=dataset_sparse.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    start_time = time.time()
    best_val_acc = 0
    epochs_no_improve = 0
    for epoch in range(1, 151):
        loss, _ = train()
        val_loss, val_acc = test(valid_loader)
        test_loss, test_acc = test(test_loader)
        if val_acc > best_val_acc + tolerance:
            best_val_acc = val_acc
            best_test_acc = test_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        print(f'Seed: {seed}, Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        if epochs_no_improve >= early_stop_patience:
            print(f'Early stopping at epoch {epoch} for seed {seed}')
            break
    end_time = time.time()
    total_time = end_time - start_time
    memory_allocated = torch.cuda.memory_reserved(device) / (1024 ** 2)
    times.append(total_time)
    memories.append(memory_allocated)
    best_val_accs.append(best_val_acc)
    best_test_accs.append(best_test_acc)
    torch.cuda.empty_cache()
print(f'Average Time: {np.mean(times):.2f} seconds')
print(f'Var Time: {np.var(times):.2f} seconds')
print(f'Average Memory: {np.mean(memories):.2f} MB')
print(f'Average Best Val AP: {np.mean(best_val_accs):.4f}')
print(f'Std Best Test AP: {np.std(best_test_accs):.4f}')
print(f'Average Test AP: {np.mean(best_test_accs):.4f}')


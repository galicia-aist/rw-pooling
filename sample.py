import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import os
import random
import torch
import torch_geometric.transforms as T
from typing import Optional
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Actor
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.datasets import TUDataset
from sklearn.metrics import r2_score
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_networkx
from torch.nn import Linear
from sklearn.model_selection import KFold
import numpy as np
import os
import random
import time
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
max_nodes = 500
data_path = "data"
dataset_sparse = TUDataset(root=data_path, name="IMDB-MULTI", transform=T.Compose([T.OneHotDegree(88)]), use_node_attr=True)
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import BatchNorm
class HierarchicalGCN_TOPK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_TOPK, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.9)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.9)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
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
model = HierarchicalGCN_TOPK(in_channels=dataset_sparse.num_features, hidden_channels=64,out_channels=64, num_classes=dataset_sparse.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
def train():
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
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)
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
early_stop_patience = 150
tolerance = 0.0001
for seed in seeds:
    set_seed(seed)
    dataset_sparse = dataset_sparse.shuffle()
    train_ratio = 0.7
    val_ratio = 0.15
    val_ratio = 0.15
    num_total = len(dataset_sparse)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)
    num_test = num_total - num_train - num_val
    train_dataset = dataset_sparse[:num_train]
    val_dataset = dataset_sparse[num_train:num_train + num_val]
    test_dataset = dataset_sparse[num_train + num_val:]
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    model = HierarchicalGCN_TOPK(in_channels=dataset_sparse.num_features, hidden_channels=64,out_channels=64, num_classes=dataset_sparse.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    best_val_acc = 0
    epochs_no_improve = 0
    for epoch in range(1, 201):
        loss = train()
        val_acc = test(valid_loader)
        test_acc = test(test_loader)
        if val_acc > best_val_acc + tolerance:
            best_val_acc = val_acc
            best_test_acc = test_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
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
print(f'Average Best Val Acc: {np.mean(best_val_accs):.4f}')
print(f'Std Best Test Acc: {np.std(best_test_accs):.4f}')
print(f'Average Test Acc: {np.mean(best_test_accs):.4f}')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import torch
# from transformers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
import os
import random
import pandas as pd
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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import random
import pandas as pd
import time
import psutil
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.utils import softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import spspmm
from torch_sparse import coalesce
from torch_sparse import eye
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_scatter import scatter_max
class PANPooling(torch.nn.Module):
    r""" General Graph pooling layer based on PAN, which can work with all layers.
    """
    def __init__(self, in_channels, ratio=0.5, pan_pool_weight=None, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh, filter_size=3, panpool_filter_weight=None):
        super(PANPooling, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity
        self.filter_size = filter_size
        if panpool_filter_weight is None:
            self.panpool_filter_weight = torch.nn.Parameter(0.5 * torch.ones(filter_size), requires_grad=True)
        self.transform = Parameter(torch.ones(in_channels), requires_grad=True)
        if pan_pool_weight is None:
            self.pan_pool_weight = torch.nn.Parameter(0.5 * torch.ones(2), requires_grad=True)
        else:
            self.pan_pool_weight = pan_pool_weight
    def forward(self, x, edge_index, M=None, batch=None, num_nodes=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        edge_index, edge_weight = self.panentropy_sparse(edge_index, num_nodes)
        num_nodes = x.size(0)
        degree = torch.zeros(num_nodes, device=edge_index.device)
        degree = scatter_add(edge_weight, edge_index[0], out=degree)
        xtransform = torch.matmul(x, self.transform)
        x_transform_norm = xtransform
        degree_norm = degree
        score = self.pan_pool_weight[0] * x_transform_norm + self.pan_pool_weight[1] * degree_norm
        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)
        perm = self.topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x
        batch = batch[perm]
        edge_index, edge_weight = self.filter_adj(edge_index, edge_weight, perm, num_nodes=score.size(0))
        return x, edge_index, edge_weight, batch, perm, score[perm]
    def topk(self, x, ratio, batch, min_score=None, tol=1e-7):
        if min_score is not None:
            scores_max = scatter_max(x, batch)[0][batch] - tol
            scores_min = scores_max.clamp(max=min_score)
            perm = torch.nonzero(x > scores_min).view(-1)
        else:
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
            cum_num_nodes = torch.cat(
                [num_nodes.new_zeros(1),
                 num_nodes.cumsum(dim=0)[:-1]], dim=0)
            index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
            index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)
            dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
            dense_x[index] = x
            dense_x = dense_x.view(batch_size, max_num_nodes)
            _, perm = dense_x.sort(dim=-1, descending=True)
            perm = perm + cum_num_nodes.view(-1, 1)
            perm = perm.view(-1)
            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
            mask = [
                torch.arange(k[i], dtype=torch.long, device=x.device) +
                i * max_num_nodes for i in range(batch_size)
            ]
            mask = torch.cat(mask, dim=0)
            perm = perm[mask]
        return perm
    def filter_adj(self, edge_index, edge_weight, perm, num_nodes=None):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i
        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]
        if edge_weight is not None:
            edge_weight = edge_weight[mask]
        return torch.stack([row, col], dim=0), edge_weight
    def panentropy_sparse(self, edge_index, num_nodes):
        edge_value = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index, edge_value = coalesce(edge_index, edge_value, num_nodes, num_nodes)
        pan_index, pan_value = eye(num_nodes, device=edge_index.device)
        indextmp = pan_index.clone().to(edge_index.device)
        valuetmp = pan_value.clone().to(edge_index.device)
        pan_value = self.panpool_filter_weight[0] * pan_value
        for i in range(self.filter_size - 1):
            indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value, num_nodes, num_nodes, num_nodes)
            valuetmp = valuetmp * self.panpool_filter_weight[i+1]
            indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            pan_index = torch.cat((pan_index, indextmp), 1)
            pan_value = torch.cat((pan_value, valuetmp))
        return coalesce(pan_index, pan_value, num_nodes, num_nodes, op='add')

from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
max_nodes = 150
data_path = "data"
dataset_sparse = TUDataset(root=data_path, name="MUTAG", pre_filter=lambda data: data.num_nodes <= max_nodes, use_node_attr=True)
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ASAPooling
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToUndirected
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import BatchNorm
class HierarchicalGCN_PAN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_PAN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = PANPooling(hidden_channels, ratio=0.9, pan_pool_weight=None, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh, filter_size=2, panpool_filter_weight=None)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = PANPooling(hidden_channels, ratio=0.9, pan_pool_weight=None, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh, filter_size=2, panpool_filter_weight=None)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, perm, score_perm = self.pool1(x, edge_index, batch=batch, M=None)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, perm, score_perm = self.pool2(x, edge_index, batch=batch, M=None)
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
model = HierarchicalGCN_PAN(in_channels=dataset_sparse.num_features, hidden_channels=64,out_channels=64, num_classes=dataset_sparse.num_classes).to(device)
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
    model = HierarchicalGCN_PAN(in_channels=dataset_sparse.num_features, hidden_channels=64,out_channels=64, num_classes=dataset_sparse.num_classes).to(device)
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
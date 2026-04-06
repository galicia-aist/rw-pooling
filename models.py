import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    DenseGCNConv,
    GCNConv,
    GraphConv,
    SAGPooling,
    TopKPooling,
    dense_diff_pool,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    EdgePooling,
    ASAPooling,
)
from pooling import uniform_pool, countsketch_pool

class DenseGCNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activate_last=True):
        super().__init__()
        self.conv1 = DenseGCNConv(in_channels, hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, out_channels)
        self.activate_last = activate_last

    def forward(self, x, adj, mask=None):
        x = F.relu(self.conv1(x, adj, mask))
        x = self.conv2(x, adj, mask)
        if self.activate_last:
            x = F.relu(x)
        return x


class MeanPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)

        aux_loss = x.new_zeros(())
        return self.cls(x), aux_loss


class UniformPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, pool_ratio=0.5):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, batch = uniform_pool(x, edge_index, batch, self.pool_ratio)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)

        aux_loss = x.new_zeros(())
        return self.cls(x), aux_loss


class TopKPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, pool_ratio=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool = TopKPooling(hidden_channels, ratio=pool_ratio)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)

        aux_loss = x.new_zeros(())
        return self.cls(x), aux_loss


class SAGPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, pool_ratio=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool = SAGPooling(hidden_channels, ratio=pool_ratio)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)

        aux_loss = x.new_zeros(())
        return self.cls(x), aux_loss


class DiffPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, max_nodes, pool_ratio=0.5):
        super().__init__()
        num_clusters = max(1, math.ceil(max_nodes * pool_ratio))

        self.embed_gnn = DenseGCNBlock(in_channels, hidden_channels, hidden_channels)
        self.assign_gnn = DenseGCNBlock(in_channels, hidden_channels, num_clusters, activate_last=False)
        self.post_gnn = DenseGCNBlock(hidden_channels, hidden_channels, hidden_channels)
        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x = data.x.float()
        adj = data.adj.float()
        mask = data.mask

        z = self.embed_gnn(x, adj, mask)
        s = self.assign_gnn(x, adj, mask)

        x, adj, link_loss, ent_loss = dense_diff_pool(z, adj, s, mask)
        x = self.post_gnn(x, adj)
        x = x.mean(dim=1)

        aux_loss = link_loss + ent_loss
        return self.cls(x), aux_loss


class CountSketchPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, pool_ratio=0.5):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.pre = GCNConv(in_channels, hidden_channels)
        self.post = GraphConv(hidden_channels, hidden_channels)
        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = F.relu(self.pre(x, edge_index))
        x, edge_index, batch, edge_weight = countsketch_pool(
            x=x,
            edge_index=edge_index,
            batch=batch,
            ratio=self.pool_ratio,
            edge_weight=None,
        )
        x = F.relu(self.post(x, edge_index, edge_weight))
        x = global_mean_pool(x, batch)

        aux_loss = x.new_zeros(())
        return self.cls(x), aux_loss

class SumPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)

        return self.cls(x), x.new_zeros(())

class MaxPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_max_pool(x, batch)

        return self.cls(x), x.new_zeros(())

class EdgePoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool = EdgePooling(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, batch, _ = self.pool(x, edge_index, batch=batch)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)

        return self.cls(x), x.new_zeros(())

class ASAPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, pool_ratio=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool = ASAPooling(hidden_channels, ratio=pool_ratio)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool(x, edge_index, batch=batch)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)

        return self.cls(x), x.new_zeros(())

def build_model(method, in_channels, hidden_channels, num_classes, max_nodes, pool_ratio):
    if method == 'mean':
        return MeanPoolNet(in_channels, hidden_channels, num_classes)

    if method == 'uniform':
        return UniformPoolNet(in_channels, hidden_channels, num_classes, pool_ratio)

    if method == 'topk':
        return TopKPoolNet(in_channels, hidden_channels, num_classes, pool_ratio)

    if method == 'sag':
        return SAGPoolNet(in_channels, hidden_channels, num_classes, pool_ratio)

    if method == 'diffpool':
        return DiffPoolNet(in_channels, hidden_channels, num_classes, max_nodes, pool_ratio)

    if method == 'countsketch':
        return CountSketchPoolNet(in_channels, hidden_channels, num_classes, pool_ratio)

    if method == 'sum':
        return SumPoolNet(in_channels, hidden_channels, num_classes)

    if method == 'max':
        return MaxPoolNet(in_channels, hidden_channels, num_classes)

    if method == 'edge':
        return EdgePoolNet(in_channels, hidden_channels, num_classes)

    if method == 'asap':
        return ASAPoolNet(in_channels, hidden_channels, num_classes, pool_ratio)

    raise ValueError(f'Unknown method: {method}')
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
    dense_mincut_pool
)
from pooling import uniform_pool, countsketch_pool
from pooling.gsa_pooling import GSAPool
from pooling.hgpsl_pooling import HGPSLPool
from pooling.pan_pooling import PANPooling, PANConv
from pooling.co_pooling import CoPooling
from  pooling.cgi_pooling import CGIPool
from  pooling.kmis_pooling_module.kmis_pooling import KMISPooling

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
    def __init__(self, in_channels, hidden_channels, num_classes, pool_ratio=0.5, task_mode="graph"):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.pool1 = TopKPooling(hidden_channels, ratio=pool_ratio)
        self.pool2 = TopKPooling(hidden_channels, ratio=pool_ratio)

        self.lin1 = nn.Linear(hidden_channels, 32)
        self.lin2 = nn.Linear(32, num_classes)
        self.task = task_mode

    def forward(self, data):
        if self.task == "node_classification":
            logits = self.forward_node(data)
        else:
            logits = self.forward_graph(data)

        return logits, logits.new_zeros(())

    def forward_node(self, data):
        x = data.x.float()
        edge_index = data.edge_index


        x = F.relu(self.conv1(x, edge_index))
        x_before = x
        batch = None
        x, edge_index, edge_attr, batch, perm, _ = self.pool(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x_unpooled = torch.zeros_like(x_before)
        x_unpooled[perm] = x
        x = x_unpooled + x_before

        return self.cls(x)

    def forward_graph(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x


class PANPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, pool_ratio=0.5):
        super().__init__()

        self.conv1 = PANConv(in_channels, hidden_channels)
        self.pool1 = PANPooling(hidden_channels, ratio=pool_ratio)

        self.conv2 = PANConv(hidden_channels, hidden_channels)
        self.pool2 = PANPooling(hidden_channels, ratio=pool_ratio)

        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        # --- Block 1 ---
        x = F.relu(self.conv1(x, edge_index))
        M = self.conv1.m
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch, M=M)

        # --- Block 2 ---
        x = F.relu(self.conv2(x, edge_index))
        M = self.conv2.m
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch, M=M)

        # --- Readout ---
        x = global_mean_pool(x, batch)

        return self.cls(x), x.new_zeros(())

class CoPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, pool_ratio=0.5):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)

        self.pool = CoPooling(
            ratio=pool_ratio,
            nhid=hidden_channels,
            edge_ratio=0.6,
            K=10,
            alpha=0.1,
            Init='PPR'
        )

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        # ✅ Ensure edge_attr exists
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), device=x.device)

        # ---- GNN ----
        x = F.relu(self.conv1(x, edge_index))

        # ---- CoPooling ----
        x, edge_index, edge_attr, batch, _, _, _ = self.pool(
            x, edge_index, edge_attr, batch
        )

        # ---- GNN ----
        x = F.relu(self.conv2(x, edge_index))

        # ---- Readout ----
        x = global_mean_pool(x, batch)

        return self.cls(x), x.new_zeros(())


class CGIPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, pool_ratio=0.5):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = CGIPool(hidden_channels, ratio=pool_ratio)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = CGIPool(hidden_channels, ratio=pool_ratio)

        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        total_aux_loss = 0

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, loss1 = self.pool1(x, edge_index, batch=batch)
        total_aux_loss += loss1

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, loss2 = self.pool2(x, edge_index, batch=batch)
        total_aux_loss += loss2

        x = global_mean_pool(x, batch)

        return self.cls(x), total_aux_loss

class KMISPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes,
                 k=1, scorer='sagpool'):
        super().__init__()

        # --- Block 1 ---
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.pool1 = KMISPooling(
            in_channels=hidden_channels,
            k=k,
            scorer=scorer,
            reduce_x='mean',     # important!
            reduce_edge='sum'
        )

        # --- Block 2 ---
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.pool2 = KMISPooling(
            in_channels=hidden_channels,
            k=k,
            scorer=scorer,
            reduce_x='mean',
            reduce_edge='sum'
        )

        # --- Classifier ---
        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        # --- Block 1 ---
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _, _ = self.pool1(x, edge_index, batch=batch)

        # --- Block 2 ---
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _, _ = self.pool2(x, edge_index, batch=batch)

        # --- Readout ---
        x = global_mean_pool(x, batch)

        return self.cls(x), x.new_zeros(())

class GSAPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, pool_ratio=0.5):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = GSAPool(hidden_channels, pooling_ratio=pool_ratio)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = GSAPool(hidden_channels, pooling_ratio=pool_ratio)

        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, batch=batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, batch=batch)

        x = global_mean_pool(x, batch)

        return self.cls(x), x.new_zeros(())

class HGPSLPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, pool_ratio=0.5):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)

        self.pool = HGPSLPool(
            in_channels=hidden_channels,
            ratio=pool_ratio
        )

        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        edge_attr = getattr(data, 'edge_attr', None)

        # --- GNN ---
        x = F.relu(self.conv1(x, edge_index))

        # --- HGPSL Pool ---
        x, edge_index, edge_attr, batch = self.pool(
            x, edge_index, edge_attr, batch
        )

        # --- GNN ---
        x = F.relu(self.conv2(x, edge_index))

        # --- Readout ---
        x = global_mean_pool(x, batch)

        return self.cls(x), x.new_zeros(())

class MinCutPoolNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, max_nodes, pool_ratio=0.5):
        super().__init__()

        num_clusters = max(1, int(max_nodes * pool_ratio))

        # --- Embedding ---
        self.conv1 = DenseGCNConv(in_channels, hidden_channels)

        # --- Assignment matrix S ---
        self.pool1 = DenseGCNConv(hidden_channels, num_clusters)

        # --- Post-pooling conv ---
        self.conv2 = DenseGCNConv(hidden_channels, hidden_channels)

        self.cls = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x = data.x.float()        # [B, N, F]
        adj = data.adj.float()   # [B, N, N]
        mask = data.mask         # [B, N]

        # --- Embed ---
        x = F.relu(self.conv1(x, adj, mask))

        # --- Assignment matrix ---
        s = self.pool1(x, adj, mask)   # [B, N, C]

        # --- MinCut pooling ---
        x, adj, mincut_loss, ortho_loss = dense_mincut_pool(x, adj, s, mask)

        # --- Post GNN ---
        x = F.relu(self.conv2(x, adj))

        # --- Readout ---
        x = x.mean(dim=1)   # same as global mean pool

        aux_loss = mincut_loss + ortho_loss

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

def build_model(method, in_channels, hidden_channels, num_classes, max_nodes, pool_ratio, task):
    if method == 'mean':
        return MeanPoolNet(in_channels, hidden_channels, num_classes)

    if method == 'uniform':
        return UniformPoolNet(in_channels, hidden_channels, num_classes, pool_ratio)

    if method == 'topk':
        return TopKPoolNet(in_channels, hidden_channels, num_classes, pool_ratio, task_mode=task)

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

    if method == 'pan':
        return PANPoolNet(in_channels, hidden_channels, num_classes, pool_ratio)

    if method == 'cop':
        return CoPoolNet(in_channels, hidden_channels, num_classes, pool_ratio)

    if method == 'cgi':
        return CGIPoolNet(in_channels, hidden_channels, num_classes, pool_ratio)

    if method == 'kmis':
        return KMISPoolNet(in_channels, hidden_channels, num_classes)

    if method == 'gsa':
        return GSAPoolNet(in_channels, hidden_channels, num_classes, pool_ratio)

    if method == 'hgpsl':
        return HGPSLPoolNet(in_channels, hidden_channels, num_classes, pool_ratio)

    if method == 'mincut':
        return MinCutPoolNet(in_channels, hidden_channels, num_classes, max_nodes, pool_ratio)

    raise ValueError(f'Unknown method: {method}')
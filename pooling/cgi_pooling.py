from torch_geometric.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap

from torch_geometric.utils import subgraph

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_channels * 2, in_channels)
        self.fc2 = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)
        return x

class CGIPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, non_lin=torch.tanh):
        super(CGIPool, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.non_lin = non_lin
        self.hidden_dim = in_channels

        self.transform = GraphConv(in_channels, self.hidden_dim)
        self.pp_conv = GraphConv(self.hidden_dim, self.hidden_dim)
        self.np_conv = GraphConv(self.hidden_dim, self.hidden_dim)

        self.positive_pooling = GraphConv(self.hidden_dim, 1)
        self.negative_pooling = GraphConv(self.hidden_dim, 1)

        self.discriminator = Discriminator(self.hidden_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def batch_topk(self, score, batch, ratio):
        perm_list = []

        for b in batch.unique():
            mask = (batch == b)
            scores_b = score[mask]
            nodes_b = mask.nonzero(as_tuple=False).view(-1)

            if isinstance(ratio, float):
                k = max(1, int(ratio * scores_b.size(0)))
            else:
                k = min(ratio, scores_b.size(0))

            topk_b = torch.topk(scores_b, k=k).indices
            perm_list.append(nodes_b[topk_b])

        return torch.cat(perm_list, dim=0)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # --- Transform ---
        x_transform = F.leaky_relu(self.transform(x, edge_index), 0.2)
        x_tp = F.leaky_relu(self.pp_conv(x, edge_index), 0.2)
        x_tn = F.leaky_relu(self.np_conv(x, edge_index), 0.2)

        s_pp = self.positive_pooling(x_tp, edge_index).squeeze()
        s_np = self.negative_pooling(x_tn, edge_index).squeeze()

        # --- Positive / Negative sampling (1 node per graph) ---
        perm_positive = self.batch_topk(s_pp, batch, ratio=1)
        perm_negative = self.batch_topk(s_np, batch, ratio=1)

        x_pp = x_transform[perm_positive] * self.non_lin(s_pp[perm_positive]).view(-1, 1)
        x_np = x_transform[perm_negative] * self.non_lin(s_np[perm_negative]).view(-1, 1)

        x_pp_readout = gap(x_pp, batch[perm_positive])
        x_np_readout = gap(x_np, batch[perm_negative])
        x_readout = gap(x_transform, batch)

        positive_pair = torch.cat([x_pp_readout, x_readout], dim=1)
        negative_pair = torch.cat([x_np_readout, x_readout], dim=1)

        device = x.device
        real = torch.ones(positive_pair.shape[0], 1, device=device)
        fake = torch.zeros(negative_pair.shape[0], 1, device=device)

        real_loss = self.loss_fn(self.discriminator(positive_pair), real)
        fake_loss = self.loss_fn(self.discriminator(negative_pair), fake)
        discrimination_loss = (real_loss + fake_loss) / 2

        # --- Final pooling ---
        score = s_pp - s_np
        perm = self.batch_topk(score, batch, self.ratio)

        x = x_transform[perm] * self.non_lin(score[perm]).view(-1, 1)
        batch = batch[perm]

        # --- Subgraph (replaces filter_adj) ---
        edge_index, edge_attr = subgraph(
            perm,
            edge_index,
            edge_attr,
            relabel_nodes=True,
            num_nodes=score.size(0)
        )

        return x, edge_index, edge_attr, batch, perm, discrimination_loss
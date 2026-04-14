import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv, GraphConv
from torch_geometric.utils import subgraph, softmax


class GSAPool(torch.nn.Module):
    def __init__(self, in_channels, pooling_ratio=0.5, alpha=0.6,
                 pooling_conv="GCNConv", fusion_conv="false",
                 min_score=None, multiplier=1, non_linearity=torch.tanh):
        super(GSAPool, self).__init__()

        self.in_channels = in_channels
        self.ratio = pooling_ratio
        self.alpha = alpha

        # --- Score branches ---
        self.sbtl_layer = self.conv_selection(pooling_conv, in_channels)
        self.fbtl_layer = nn.Linear(in_channels, 1)

        # ✅ FIX: only create fusion if needed
        self.fusion_flag = fusion_conv != "false"
        if self.fusion_flag:
            self.fusion = self.conv_selection(fusion_conv, in_channels, conv_type=1)

        self.min_score = min_score
        self.multiplier = multiplier
        self.non_linearity = non_linearity

    def conv_selection(self, conv, in_channels, conv_type=0):
        out_channels = 1 if conv_type == 0 else in_channels

        if conv == "GCNConv":
            return GCNConv(in_channels, out_channels)
        elif conv == "ChebConv":
            return ChebConv(in_channels, out_channels, K=1)
        elif conv == "SAGEConv":
            return SAGEConv(in_channels, out_channels)
        elif conv == "GATConv":
            return GATConv(in_channels, out_channels, heads=1, concat=True)
        elif conv == "GraphConv":
            return GraphConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown conv type: {conv}")

    # ✅ replacement for topk (batch-wise)
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

        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # --- SBTL ---
        score_s = self.sbtl_layer(x, edge_index).squeeze()

        # --- FBTL ---
        score_f = self.fbtl_layer(x).squeeze()

        # --- Combine ---
        score = score_s * self.alpha + score_f * (1 - self.alpha)

        if score.dim() == 0:
            score = score.unsqueeze(-1)

        # --- Normalize ---
        if self.min_score is None:
            score = self.non_linearity(score)
        else:
            score = softmax(score, batch)

        # --- TopK selection ---
        perm = self.batch_topk(score, batch, self.ratio)

        # --- Fusion (optional) ---
        if self.fusion_flag:
            x = self.fusion(x, edge_index)

        # --- Apply selection ---
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x
        batch = batch[perm]

        # --- Subgraph (replaces filter_adj) ---
        edge_index, edge_attr = subgraph(
            perm,
            edge_index,
            edge_attr,
            relabel_nodes=True,
            num_nodes=score.size(0)
        )

        return x, edge_index, edge_attr, batch, perm
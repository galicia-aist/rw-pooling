import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import (
    softmax,
    dense_to_sparse,
    add_remaining_self_loops,
    subgraph
)
from torch_scatter import scatter_add
from torch_sparse import spspmm, coalesce

from .spare_softmax import Sparsemax


class TwoHopNeighborhood(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        n = data.num_nodes

        fill = 1e16
        value = edge_index.new_full((edge_index.size(1),), fill, dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, n, n, n, True)

        edge_index = torch.cat([edge_index, index], dim=1)

        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, n, n)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)

            data.edge_index, edge_attr = coalesce(
                edge_index, edge_attr, n, n, op='min', fill_value=fill
            )
            edge_attr[edge_attr >= fill] = 0
            data.edge_attr = edge_attr

        return data


class NodeInformationScore(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight):
        row, col = edge_index

        if edge_weight is None:
            edge_weight = torch.ones(row.size(0), device=x.device)

        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class HGPSLPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8,
                 sample=False, sparse=False, sl=True,
                 lamb=1.0, negative_slop=0.2):
        super(HGPSLPool, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.negative_slop = negative_slop
        self.lamb = lamb

        self.att = Parameter(torch.Tensor(1, self.in_channels * 2))
        nn.init.xavier_uniform_(self.att.data)

        self.sparse_attention = Sparsemax()
        self.neighbor_augment = TwoHopNeighborhood()
        self.calc_information_score = NodeInformationScore()

    # ✅ replacement for topk
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

        # --- Node scoring ---
        x_info = self.calc_information_score(x, edge_index, edge_attr)
        score = torch.sum(torch.abs(x_info), dim=1)

        original_x = x

        # ✅ NEW topk
        perm = self.batch_topk(score, batch, self.ratio)

        x = x[perm]
        batch = batch[perm]

        # ✅ NEW filter_adj replacement
        induced_edge_index, induced_edge_attr = subgraph(
            perm,
            edge_index,
            edge_attr,
            relabel_nodes=True,
            num_nodes=score.size(0)
        )

        # --- No structure learning ---
        if not self.sl:
            return x, induced_edge_index, induced_edge_attr, batch

        # -------------------------------
        # Structure Learning
        # -------------------------------
        if edge_attr is None:
            induced_edge_attr = torch.ones(
                induced_edge_index.size(1),
                dtype=x.dtype,
                device=x.device
            )

        row, col = induced_edge_index

        weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
        weights = F.leaky_relu(weights, self.negative_slop)

        weights = weights + induced_edge_attr * self.lamb

        if self.sparse:
            new_edge_attr = self.sparse_attention(weights, row)
        else:
            new_edge_attr = softmax(weights, row, num_nodes=x.size(0))

        new_edge_index = induced_edge_index

        return x, new_edge_index, new_edge_attr, batch
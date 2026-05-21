import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import torch
# from transformers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
import torch_geometric.transforms as T
# from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
import os
import random
# import pandas as pd
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
# import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np
import os
import random
# import pandas as pd
import time
# import psutil
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import spspmm
from torch_sparse import coalesce
from torch_sparse import eye
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_scatter import scatter_max
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn import GCNConv, GATConv, LEConv, SAGEConv, GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, add_self_loops
from typing import Callable, Optional, Union
from torch_sparse import coalesce, transpose
from torch_scatter import scatter
from torch import Tensor
from typing import List, Optional, Tuple, Union
import math
import torch
from torch import Tensor
from torch_geometric.nn.models.mlp import Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn import BatchNorm
class AsymCheegerCutPool(torch.nn.Module):
    r"""
    The asymmetric cheeger cut pooling layer from the `"Total Variation Graph Neural Networks"
    <https://arxiv.org/abs/2211.06218>`_ paper.
    Args:
        k (int):
            Number of clusters or output nodes
        mlp_channels (int, list of int):
            Number of hidden units for each hidden layer in the MLP used to
            compute cluster assignments. First integer must match the number
            of input channels.
        mlp_activation (any):
            Activation function between hidden layers of the MLP.
            Must be compatible with `torch_geometric.nn.resolver`.
        return_selection (bool):
            Whether to return selection matrix. Cannot not  be False
            if `return_pooled_graph` is False. (default: :obj:`False`)
        return_pooled_graph (bool):
            Whether to return pooled node features and adjacency.
            Cannot be False if `return_selection` is False. (default: :obj:`True`)
        bias (bool):
            whether to add a bias term to the MLP layers. (default: :obj:`True`)
        totvar_coeff (float):
            Coefficient for graph total variation loss component. (default: :obj:`1.0`)
        balance_coeff (float):
            Coefficient for asymmetric norm loss component. (default: :obj:`1.0`)
    """
    def __init__(self,
                 k: int,
                 mlp_channels: Union[int, List[int]],
                 mlp_activation="relu",
                 return_selection: bool = False,
                 return_pooled_graph: bool = True,
                 bias: bool = True,
                 totvar_coeff: float = 1.0,
                 balance_coeff: float = 1.0,
                 ):
        super().__init__()
        if not return_selection and not return_pooled_graph:
            raise ValueError("return_selection and return_pooled_graph can not both be False")
        if isinstance(mlp_channels, int):
            mlp_channels = [mlp_channels]
        act = activation_resolver(mlp_activation)
        in_channels = mlp_channels[0]
        self.mlp = torch.nn.Sequential()
        for channels in mlp_channels[1:]:
            self.mlp.append(Linear(in_channels, channels, bias=bias))
            in_channels = channels
            self.mlp.append(act)
        self.mlp.append(Linear(in_channels, k))
        self.k = k
        self.return_selection = return_selection
        self.return_pooled_graph = return_pooled_graph
        self.totvar_coeff = totvar_coeff
        self.balance_coeff = balance_coeff
        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Args:
            x (Tensor):
                Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`
                with batch-size :math:`B`, (maximum) number of nodes :math:`N` for each graph,
                and feature dimension :math:`F`. Note that the cluster assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}` is
                being created within this method.
            adj (Tensor):
                Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            mask (BoolTensor, optional):
                Mask matrix :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}`
                indicating the valid nodes for each graph. (default: :obj:`None`)
        :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
            :class:`Tensor`, :class:`Tensor`, :class:`Tensor`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = self.mlp(x)
        s = torch.softmax(s, dim=-1)
        batch_size, n_nodes, _ = x.size()
        if mask is not None:
            mask = mask.view(batch_size, n_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask
        if self.return_pooled_graph:
            x_pool = torch.matmul(s.transpose(1, 2), x)
            adj_pool = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        tv_loss = self.totvar_coeff*torch.mean(self.totvar_loss(adj, s))
        bal_loss = self.balance_coeff*torch.mean(self.balance_loss(s))
        if self.return_selection and self.return_pooled_graph:
            return s, x_pool, adj_pool, tv_loss, bal_loss
        elif self.return_selection and not self.return_pooled_graph:
            return s, tv_loss, bal_loss
        else:
            return x_pool, adj_pool, tv_loss, bal_loss
    def totvar_loss(self, adj, s):
        l1_norm = torch.sum(torch.abs(s[..., None, :] - s[:, None, ...]), dim=-1)
        loss = torch.sum(adj * l1_norm, dim=(-1, -2))
        n_edges = torch.count_nonzero(adj, dim=(-1, -2))
        loss *= 1 / (2 * n_edges)
        return loss
    def balance_loss(self, s):
        n_nodes = s.size()[-2]
        idx = int(math.floor(n_nodes / self.k))
        quant = torch.sort(s, dim=-2, descending=True)[0][:, idx, :]
        loss = s - torch.unsqueeze(quant, dim=1)
        loss = (loss >= 0) * (self.k - 1) * loss + (loss < 0) * loss * -1
        loss = torch.sum(loss, dim=(-1, -2))
        loss = 1 / (n_nodes * (self.k - 1)) * (n_nodes * (self.k - 1) - loss)
        return loss



from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
import random
from torch_geometric.nn import GCNConv
import os.path as osp
import time
from math import ceil
import torch
import torch.nn.functional as F


from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseGCNConv, dense_diff_pool

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

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False, lin=True):
        super().__init__()
        self.conv1 = DenseGCNConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseGCNConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        if lin:
            self.lin = torch.nn.Linear(out_channels, out_channels)
        else:
            self.lin = None
    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x
    def forward(self, x, adj, mask=None):
        x = self.bn(1, self.conv1(x, adj, mask).relu())
        x = self.bn(2, self.conv2(x, adj, mask).relu())
        x = self.bn(3, self.conv3(x, adj, mask).relu())
        if self.lin is not None:
            x = self.lin(x).relu()
        return x


class Net_AsymCheegerCut(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        mp_channels = 64
        mlp_hidden_layers = 2
        mlp_hidden_channels = 128
        mlp_activation = "relu"
        totvar_coeff = 0.5
        balance_coeff = 0.5
        N = 150
        num_nodes = 64
        self.gnn1_pool = GNN(num_features, 64, num_nodes)
        self.gnn1_embed = DenseGCNConv(num_features, 64)
        num_nodes = 64
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.gnn2_embed = DenseGCNConv(64, 64)
        self.gnn3_embed = DenseGCNConv(64, 64)
        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)
        self.pool1 = AsymCheegerCutPool(int(N//2),
                           mlp_channels=[mp_channels] +
                                [mlp_hidden_channels for _ in range(mlp_hidden_layers)],
                           mlp_activation=mlp_activation,
                           totvar_coeff=totvar_coeff,
                           balance_coeff=balance_coeff,
                           return_selection=False,
                           return_pooled_graph=True)
        self.pool2 = AsymCheegerCutPool(int(N//2),
                           mlp_channels=[mp_channels] +
                                [mlp_hidden_channels for _ in range(mlp_hidden_layers)],
                           mlp_activation=mlp_activation,
                           totvar_coeff=totvar_coeff,
                           balance_coeff=balance_coeff,
                           return_selection=False,
                           return_pooled_graph=True)
    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x = F.relu(x)
        x, adj, tv1, bal1 = self.pool1(x, adj, mask=None)
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x = F.relu(x)
        x, adj, tv1, bal1 = self.pool2(x, adj, mask=None)
        x = self.gnn3_embed(x, adj)
        x = F.relu(x)
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

class Net_Diff(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        num_nodes = 64
        self.gnn1_pool = GNN(num_features, 64, num_nodes)
        self.gnn1_embed = DenseGCNConv(num_features, 64)
        num_nodes = 64
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.gnn2_embed = DenseGCNConv(64, 64)
        self.gnn3_embed = DenseGCNConv(64, 64)
        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)
    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x = F.relu(x)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x = F.relu(x)
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        x = self.gnn3_embed(x, adj)
        x = F.relu(x)
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2
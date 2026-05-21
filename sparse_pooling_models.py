import torch
from torch.autograd import Function
from torch_geometric.data import Data
from torch_geometric.nn import (GCNConv, TopKPooling, SAGPooling, ASAPooling, GraphConv, ChebConv, GATConv, DenseGCNConv,
                                dense_diff_pool)
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, dense_to_sparse
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, add_self_loops
import numpy as np
import torch.nn as nn
from torch_sparse import coalesce, transpose, spspmm
from typing import Callable, Optional, Union, Tuple
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.typing import Adj, OptTensor, PairTensor, Tensor
from torch_scatter import scatter_max, scatter, scatter_add, scatter_min
from torch.nn import Module
from torch_sparse import SparseTensor, remove_diag
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.dense import Linear
Scorer = Callable[[Tensor, Adj, OptTensor, OptTensor], Tensor]
from torch.nn import Linear
from torch_geometric.utils import softmax


class HierarchicalGCN_TOPK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, pool_ratio):
        super(HierarchicalGCN_TOPK, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=pool_ratio)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=pool_ratio)
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

class HierarchicalGCN_SAG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, pool_ratio):
        super(HierarchicalGCN_SAG, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = SAGPooling(hidden_channels, ratio=pool_ratio)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = SAGPooling(hidden_channels, ratio=pool_ratio)
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


class HierarchicalGCN_ASA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, pool_ratio):
        super(HierarchicalGCN_ASA, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = ASAPooling(hidden_channels, ratio=pool_ratio)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = ASAPooling(hidden_channels, ratio=pool_ratio)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, batch=batch)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, batch=batch)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

def cumsum(x: Tensor, dim: int = 0) -> Tensor:
    r"""Returns the cumulative sum of elements of :obj:`x`.
    In contrast to :meth:`torch.cumsum`, prepends the output with zero.
    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension to do the operation over.
            (default: :obj:`0`)
    Example:
        >>> x = torch.tensor([2, 4, 1])
        >>> cumsum(x)
        tensor([0, 2, 6, 7])
    """
    size = x.size()[:dim] + (x.size(dim) + 1, ) + x.size()[dim + 1:]
    out = x.new_empty(size)
    out.narrow(dim, 0, 1).zero_()
    torch.cumsum(x, dim=dim, out=out.narrow(dim, 1, x.size(dim)))
    return out

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i
    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]
    return torch.stack([row, col], dim=0), edge_attr

def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)
        perm = (x > scores_min).nonzero().view(-1)
        return perm
    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)
        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)
        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]
        return x_perm[batch_perm[mask]]
    raise ValueError("At least one of the 'ratio' and 'min_score' parameters "
                     "must be specified")
class graph_attention(torch.nn.Module):
    src_nodes_dim = 0
    trg_nodes_dim = 1
    nodes_dim = 0
    head_dim = 1
    def __init__(self, num_in_features, num_out_features, num_of_heads, dropout_prob=0.6, log_attention_weights=False):
        super().__init__()
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.init_params()
    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow
        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.
        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
    def forward(self, x, edge_index):
        in_nodes_features = x
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = scores_source_lifted + scores_target_lifted
        return torch.sigmoid(scores_per_edge)
    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)
        return scores_source, scores_target, nodes_features_matrix_proj_lifted

class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''
    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            TEMP = Gamma
        self.temp = Parameter(torch.tensor(TEMP))
    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K
    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                           self.temp)

class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super(NodeInformationScore, self).__init__(aggr='add', **kwargs)
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, num_nodes)
        edge_index = edge_index.type(torch.long)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        expand_deg = torch.zeros((edge_weight.size(0),), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes,), dtype=dtype, device=edge_index.device)
        return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    def forward(self, x, edge_index, edge_weight):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    def update(self, aggr_out):
        return aggr_out



class CoPooling(torch.nn.Module):
    def __init__(self, ratio=0.5, K=0.05, edge_ratio=0.6, nhid=64, alpha=0.1, Init='Random', Gamma=None):
        super(CoPooling, self).__init__()
        self.ratio = ratio
        self.calc_information_score = NodeInformationScore()
        self.edge_ratio = edge_ratio
        self.prop1 = GPR_prop(K, alpha, Init, Gamma)
        score_dim = 32
        self.G_att = graph_attention(num_in_features=nhid, num_out_features=score_dim, num_of_heads=1)
        self.weight = Parameter(torch.Tensor(2*nhid, nhid))
        nn.init.xavier_uniform_(self.weight.data)
        self.bias = Parameter(torch.Tensor(nhid))
        nn.init.zeros_(self.bias.data)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.zeros_(self.bias.data)
        self.prop1.reset_parameters()
        self.G_att.init_params()
    def forward(self, x, edge_index, edge_attr, batch=None, nodes_index=None, node_attr=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        ori_batch = batch.clone()
        device = x.device
        num_nodes = x.shape[0]
        x_cut = self.prop1(x, edge_index)
        attention = self.G_att(x_cut, edge_index)
        attention = attention.sum(dim=1)
        edge_index, attention = add_self_loops(edge_index, attention, 1.0, num_nodes)
        edge_index_t, attention_t = transpose(edge_index, attention, num_nodes, num_nodes)
        edge_tmp = torch.cat((edge_index, edge_index_t), 1)
        att_tmp = torch.cat((attention, attention_t),0)
        edge_index, attention = coalesce(edge_tmp, att_tmp, num_nodes, num_nodes, 'mean')
        attention_np = attention.cpu().data.numpy()
        cut_val = np.percentile(attention_np, int(100*(1-self.edge_ratio)))
        attention = attention * (attention >= cut_val)
        kep_idx = attention > 0.0
        cut_edge_index, cut_edge_attr = edge_index[:, kep_idx], attention[kep_idx]
        x_information_score = self.calc_information_score(x, cut_edge_index, cut_edge_attr)
        score = torch.sum(torch.abs(x_information_score), dim=1)
        perm = topk(score, self.ratio, batch)
        x_topk = x[perm]
        batch = batch[perm]
        if nodes_index is not None:
            nodes_index = nodes_index[perm]
        if node_attr is not None:
            node_attr = node_attr[perm]
        if cut_edge_index is not None or cut_edge_index.nelement() != 0:
            induced_edge_index, induced_edge_attr = filter_adj(cut_edge_index, cut_edge_attr, perm, num_nodes=num_nodes)
        else:
            print('All edges are cut!')
            induced_edge_index, induced_edge_attr = cut_edge_index, cut_edge_attr
        attention_dense = (to_dense_adj(cut_edge_index, edge_attr=cut_edge_attr, max_num_nodes=num_nodes)).squeeze()
        x = F.relu(torch.matmul(torch.cat((x_topk, torch.matmul(attention_dense[perm],x)), 1), self.weight) + self.bias)
        return x, induced_edge_index, perm, induced_edge_attr, batch, nodes_index, node_attr, attention_dense
class HierarchicalGCN_CO(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, pool_ratio):
        super(HierarchicalGCN_CO, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = CoPooling(ratio=pool_ratio, K=1, edge_ratio=0.6, nhid=64, alpha=0.1, Init='Random', Gamma=1.0)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = CoPooling(ratio=pool_ratio, K=1, edge_ratio=0.6, nhid=64, alpha=0.1, Init='Random', Gamma=1.0)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, perm, _, batch, _, _, _ = self.pool1(x, edge_index, edge_attr=None, batch=batch)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, perm, _, batch, _, _, _ = self.pool2(x, edge_index, edge_attr=None, batch=batch)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_channels * 2, in_channels)
        self.fc2 = nn.Linear(in_channels, 1)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.sigmoid(self.fc2(x))
        return x

class CGIPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.5, non_lin=torch.tanh):
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
        self.loss_fn = torch.nn.BCELoss()
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        device = x.device
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x_transform = F.leaky_relu(self.transform(x, edge_index), 0.2)
        x_tp = F.leaky_relu(self.pp_conv(x, edge_index), 0.2)
        x_tn = F.leaky_relu(self.np_conv(x, edge_index), 0.2)
        s_pp = self.positive_pooling(x_tp, edge_index).squeeze()
        s_np = self.negative_pooling(x_tn, edge_index).squeeze()
        perm_positive = topk(s_pp, 1, batch)
        perm_negative = topk(s_np, 1, batch)
        x_pp = x_transform[perm_positive] * self.non_lin(s_pp[perm_positive]).view(-1, 1)
        x_np = x_transform[perm_negative] * self.non_lin(s_np[perm_negative]).view(-1, 1)
        x_pp_readout = gap(x_pp, batch[perm_positive])
        x_np_readout = gap(x_np, batch[perm_negative])
        x_readout = gap(x_transform, batch)
        positive_pair = torch.cat([x_pp_readout, x_readout], dim=1)
        negative_pair = torch.cat([x_np_readout, x_readout], dim=1)
        real = torch.ones(positive_pair.shape[0], device=device)
        fake = torch.zeros(negative_pair.shape[0], device=device)
        score = (s_pp - s_np)
        perm = topk(score, self.ratio, batch)
        x = x_transform[perm] * self.non_lin(score[perm]).view(-1, 1)
        batch = batch[perm]
        filter_edge_index, filter_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, filter_edge_index, filter_edge_attr, batch, perm

class HierarchicalGCN_CGI(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, pool_ratio):
        super(HierarchicalGCN_CGI, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = CGIPool(hidden_channels, ratio=pool_ratio)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = CGIPool(hidden_channels, ratio=pool_ratio)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, perm = self.pool1(x, edge_index, None, batch)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, perm = self.pool2(x, edge_index, None, batch)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

def maximal_independent_set(edge_index: Adj, k: int = 1,
                            perm: OptTensor = None) -> Tensor:
    r"""Returns a Maximal :math:`k`-Independent Set of a graph, i.e., a set of
    nodes (as a :class:`ByteTensor`) such that none of them are :math:`k`-hop
    neighbors, and any node in the graph has a :math:`k`-hop neighbor in the
    returned set.
    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.
    This method follows `Blelloch's Alogirithm
    <https://arxiv.org/abs/1202.3205>`_ for :math:`k = 1`, and its
    generalization by `Bacciu et al. <https://arxiv.org/abs/2208.03523>`_ for
    higher values of :math:`k`.
    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).
    :rtype: :class:`ByteTensor`
    """
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        device = edge_index.device()
        n = edge_index.size(0)
    else:
        row, col = edge_index[0], edge_index[1]
        device = row.device
        n = edge_index.max().item() + 1
    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)
    mis = torch.zeros(n, dtype=torch.bool, device=device)
    mask = mis.clone()
    min_rank = rank.clone()
    while not mask.all():
        for _ in range(k):
            min_neigh = torch.full_like(min_rank, fill_value=n)
            scatter_min(min_rank[row], col, out=min_neigh)
            torch.minimum(min_neigh, min_rank, out=min_rank)
        mis = mis | torch.eq(rank, min_rank)
        mask = mis.clone().byte()
        for _ in range(k):
            max_neigh = torch.full_like(mask, fill_value=0)
            scatter_max(mask[row], col, out=max_neigh)
            torch.maximum(max_neigh, mask, out=mask)
        mask = mask.to(dtype=torch.bool)
        min_rank = rank.clone()
        min_rank[mask] = n
    return mis
def maximal_independent_set_cluster(edge_index: Adj, k: int = 1,
                                    perm: OptTensor = None) -> PairTensor:
    r"""Computes the Maximal :math:`k`-Independent Set (:math:`k`-MIS)
    clustering of a graph, as defined in `"Generalizing Downsampling from
    Regular Data to Graphs" <https://arxiv.org/abs/2208.03523>`_.
    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.
    This method returns both the :math:`k`-MIS and the clustering, where the
    :math:`c`-th cluster refers to the :math:`c`-th element of the
    :math:`k`-MIS.
    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).
    :rtype: (:class:`ByteTensor`, :class:`LongTensor`)
    """
    mis = maximal_independent_set(edge_index=edge_index, k=k, perm=perm)
    n, device = mis.size(0), mis.device
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index[0], edge_index[1]
    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)
    min_rank = torch.full((n, ), fill_value=n, dtype=torch.long, device=device)
    rank_mis = rank[mis]
    min_rank[mis] = rank_mis
    for _ in range(k):
        min_neigh = torch.full_like(min_rank, fill_value=n)
        scatter_min(min_rank[row], col, out=min_neigh)
        torch.minimum(min_neigh, min_rank, out=min_rank)
    _, clusters = torch.unique(min_rank, return_inverse=True)
    perm = torch.argsort(rank_mis)
    return mis, perm[clusters]

class KMISPooling(Module):
    _heuristics = {None, 'greedy', 'w-greedy'}
    _passthroughs = {None, 'before', 'after'}
    _scorers = {
        'linear',
        'random',
        'constant',
        'canonical',
        'first',
        'last',
    }
    def __init__(self, in_channels: Optional[int] = None, k: int = 1,
                 scorer: Union[Scorer, str] = 'linear',
                 score_heuristic: Optional[str] = 'greedy',
                 score_passthrough: Optional[str] = 'before',
                 aggr_x: Optional[Union[str, Aggregation]] = None,
                 aggr_edge: str = 'sum',
                 aggr_score: Callable[[Tensor, Tensor], Tensor] = torch.mul,
                 remove_self_loops: bool = True) -> None:
        super(KMISPooling, self).__init__()
        assert score_heuristic in self._heuristics, \
            "Unrecognized `score_heuristic` value."
        assert score_passthrough in self._passthroughs, \
            "Unrecognized `score_passthrough` value."
        if not callable(scorer):
            assert scorer in self._scorers, \
                "Unrecognized `scorer` value."
        self.k = k
        self.scorer = scorer
        self.score_heuristic = score_heuristic
        self.score_passthrough = score_passthrough
        self.aggr_x = aggr_x
        self.aggr_edge = aggr_edge
        self.aggr_score = aggr_score
        self.remove_self_loops = remove_self_loops
        if scorer == 'linear':
            assert self.score_passthrough is not None, \
                "`'score_passthrough'` must not be `None`" \
                " when using `'linear'` scorer"
            self.lin = Linear(in_features=in_channels, out_features=1)
    def _apply_heuristic(self, x: Tensor, adj: SparseTensor) -> Tensor:
        if self.score_heuristic is None:
            return x
        row, col, _ = adj.coo()
        x = x.view(-1)
        if self.score_heuristic == 'greedy':
            k_sums = torch.ones_like(x)
        else:
            k_sums = x.clone()
        for _ in range(self.k):
            scatter_add(k_sums[row], col, out=k_sums)
        return x / k_sums
    def _scorer(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                batch: OptTensor = None) -> Tensor:
        if self.scorer == 'linear':
            return self.lin(x).sigmoid()
        if self.scorer == 'random':
            return torch.rand((x.size(0), 1), device=x.device)
        if self.scorer == 'constant':
            return torch.ones((x.size(0), 1), device=x.device)
        if self.scorer == 'canonical':
            return -torch.arange(x.size(0), device=x.device).view(-1, 1)
        if self.scorer == 'first':
            return x[..., [0]]
        if self.scorer == 'last':
            return x[..., [-1]]
        return self.scorer(x, edge_index, edge_attr, batch)
    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                batch: OptTensor = None) \
            -> Tuple[Tensor, Adj, OptTensor, OptTensor, Tensor, Tensor]:
        """"""
        edge_index = edge_index.long()
        adj, n = edge_index, x.size(0)
        if not isinstance(edge_index, SparseTensor):
            adj = SparseTensor.from_edge_index(edge_index, edge_attr, (n, n))
        score = self._scorer(x, edge_index, edge_attr, batch)
        updated_score = self._apply_heuristic(score, adj)
        perm = torch.argsort(updated_score.view(-1), 0, descending=True)
        mis, cluster = maximal_independent_set_cluster(adj, self.k, perm)
        row, col, val = adj.coo()
        c = mis.sum()
        if val is None:
            val = torch.ones_like(row, dtype=torch.float)
        adj = SparseTensor(row=cluster[row], col=cluster[col], value=val,
                           is_sorted=False,
                           sparse_sizes=(c, c)).coalesce(self.aggr_edge)
        if self.remove_self_loops:
            adj = remove_diag(adj)
        if self.score_passthrough == 'before':
            x = self.aggr_score(x, score)
        if self.aggr_x is None:
            x = x[mis]
        elif isinstance(self.aggr_x, str):
            x = scatter(x, cluster, dim=0, dim_size=mis.sum(),
                        reduce=self.aggr_x)
        else:
            x = self.aggr_x(x, cluster, dim_size=c)
        if self.score_passthrough == 'after':
            x = self.aggr_score(x, score[mis])
        if isinstance(edge_index, SparseTensor):
            edge_index, edge_attr = adj, None
        else:
            row, col, edge_attr = adj.coo()
            edge_index = torch.stack([row, col])
        if batch is not None:
            batch = batch[mis]
        return x, edge_index, edge_attr, batch, mis, cluster
    def __repr__(self):
        if self.scorer == 'linear':
            channels = f"in_channels={self.lin.in_channels}, "
        else:
            channels = ""
        return f'{self.__class__.__name__}({channels}k={self.k})'

class HierarchicalGCN_KMIS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(HierarchicalGCN_KMIS, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = KMISPooling(64, k=1, aggr_x='sum')
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = KMISPooling(64, k=1, aggr_x='sum')
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

class GSAPool(torch.nn.Module):
    def __init__(self, in_channels, pooling_ratio=0.5, alpha=0.6,
                        min_score=None, multiplier=1,
                        non_linearity=torch.tanh,
                        cus_drop_ratio =0):
        super(GSAPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = pooling_ratio
        self.alpha = alpha
        self.sbtl_layer = GCNConv(in_channels,1)
        self.fbtl_layer = nn.Linear(in_channels, 1)
        self.fusion = GCNConv(in_channels,in_channels)
        self.min_score = min_score
        self.multiplier = multiplier
        self.fusion_flag = 0
        self.non_linearity = non_linearity
        self.dropout = torch.nn.Dropout(cus_drop_ratio)
    def conv_selection(self, conv, in_channels, conv_type=0):
        if(conv_type == 0):
            out_channels = 1
        elif(conv_type == 1):
            out_channels = in_channels
        if(conv == "GCNConv"):
            return GCNConv(in_channels,out_channels)
        elif(conv == "ChebConv"):
            return ChebConv(in_channels,out_channels,1)
        elif(conv == "GCNConv"):
            return GCNConv(in_channels,out_channels)
        elif(conv == "GATConv"):
            return GATConv(in_channels,out_channels, heads=1, concat=True)
        elif(conv == "GraphConv"):
            return GraphConv(in_channels,out_channels)
        else:
            raise ValueError
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        score_s = self.sbtl_layer(x,edge_index).squeeze()
        score_f = self.fbtl_layer(x).squeeze()
        score = score_s*self.alpha + score_f*(1-self.alpha)
        score = score.unsqueeze(-1) if score.dim()==0 else score
        if self.min_score is None:
            score = self.non_linearity(score)
        else:
            score = softmax(score, batch)
        sc = self.dropout(score)
        perm = topk(sc, self.ratio, batch)
        if(self.fusion_flag == 1):
            x = self.fusion(x, edge_index)
        x_ae = x[perm]
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm, x_ae
class HierarchicalGCN_GSA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, pool_ratio):
        super(HierarchicalGCN_GSA, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = GSAPool(64, pooling_ratio=pool_ratio, alpha = 0.6, cus_drop_ratio = 0)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = GSAPool(64, pooling_ratio=pool_ratio, alpha = 0.6, cus_drop_ratio = 0)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, perm_1, x_ae1 = self.pool1(x, edge_index, None, batch)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, perm_1, x_ae1 = self.pool2(x, edge_index, None, batch)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

def scatter_sort(x, batch, fill_value=-1e16):
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)
    dense_x = x.new_full((batch_size * max_num_nodes,), fill_value)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)
    sorted_x, _ = dense_x.sort(dim=-1, descending=True)
    cumsum_sorted_x = sorted_x.cumsum(dim=-1)
    cumsum_sorted_x = cumsum_sorted_x.view(-1)
    sorted_x = sorted_x.view(-1)
    filled_index = sorted_x != fill_value
    sorted_x = sorted_x[filled_index]
    cumsum_sorted_x = cumsum_sorted_x[filled_index]
    return sorted_x, cumsum_sorted_x
def _make_ix_like(batch):
    num_nodes = scatter_add(batch.new_ones(batch.size(0)), batch, dim=0)
    idx = [torch.arange(1, i + 1, dtype=torch.long, device=batch.device) for i in num_nodes]
    idx = torch.cat(idx, dim=0)
    return idx
def _threshold_and_support(x, batch):
    """Sparsemax building block: compute the threshold
    Args:
        x: input tensor to apply the sparsemax
        batch: group indicators
    Returns:
        the threshold value
    """
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
    sorted_input, input_cumsum = scatter_sort(x, batch)
    input_cumsum = input_cumsum - 1.0
    rhos = _make_ix_like(batch).to(x.dtype)
    support = rhos * sorted_input > input_cumsum
    support_size = scatter_add(support.to(batch.dtype), batch)
    idx = support_size + cum_num_nodes - 1
    mask = idx < 0
    idx[mask] = 0
    tau = input_cumsum.gather(0, idx)
    tau /= support_size.to(x.dtype)
    return tau, support_size
class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, x, batch):
        """sparsemax: normalizing sparse transform
        Parameters:
            ctx: context object
            x (Tensor): shape (N, )
            batch: group indicator
        Returns:
            output (Tensor): same shape as input
        """
        max_val, _ = scatter_max(x, batch)
        x -= max_val[batch]
        tau, supp_size = _threshold_and_support(x, batch)
        output = torch.clamp(x - tau[batch], min=0)
        ctx.save_for_backward(supp_size, output, batch)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output, batch = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        v_hat = scatter_add(grad_input, batch) / supp_size.to(output.dtype)
        grad_input = torch.where(output != 0, grad_input - v_hat[batch], grad_input)
        return grad_input, None
sparsemax = SparsemaxFunction.apply
class Sparsemax(nn.Module):
    def __init__(self):
        super(Sparsemax, self).__init__()
    def forward(self, x, batch):
        return sparsemax(x, batch)

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
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, n, n, op='min', fill_value=fill)
            edge_attr[edge_attr >= fill] = 0
            data.edge_attr = edge_attr
        return data
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super(NodeInformationScore, self).__init__(aggr='add', **kwargs)
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, num_nodes)
        row, col = edge_index
        expand_deg = torch.zeros((edge_weight.size(0),), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes,), dtype=dtype, device=edge_index.device)
        return edge_index, expand_deg - deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    def forward(self, x, edge_index, edge_weight):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    def update(self, aggr_out):
        return aggr_out

class HGPSLPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.5, sample=False, sparse=False, sl=True, lamb=1.0, negative_slop=0.2):
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
    def forward(self, x, edge_index, edge_attr, batch):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        x_information_score = self.calc_information_score(x, edge_index, edge_attr)
        score = torch.sum(torch.abs(x_information_score), dim=1)
        original_x = x
        perm = topk(score, self.ratio, batch)
        x = x[perm]
        batch = batch[perm]
        induced_edge_index, induced_edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))
        if self.sl is False:
            return x, induced_edge_index, induced_edge_attr, batch
        if self.sample:
            k_hop = 3
            if edge_attr is None:
                edge_attr = torch.ones((edge_index.size(1),), dtype=torch.float, device=edge_index.device)
            hop_data = Data(x=original_x, edge_index=edge_index, edge_attr=edge_attr)
            for _ in range(k_hop - 1):
                hop_data = self.neighbor_augment(hop_data)
            hop_edge_index = hop_data.edge_index
            hop_edge_attr = hop_data.edge_attr
            new_edge_index, new_edge_attr = filter_adj(hop_edge_index, hop_edge_attr, perm, num_nodes=score.size(0))
            new_edge_index, new_edge_attr = add_remaining_self_loops(new_edge_index, new_edge_attr, 0, x.size(0))
            row, col = new_edge_index
            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop) + new_edge_attr * self.lamb
            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            adj[row, col] = weights
            new_edge_index, weights = dense_to_sparse(adj)
            row, col = new_edge_index
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, x.size(0))
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)
            del adj
            torch.cuda.empty_cache()
        else:
            if edge_attr is None:
                induced_edge_attr = torch.ones((induced_edge_index.size(1),), dtype=x.dtype,
                                               device=induced_edge_index.device)
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            shift_cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)
            cum_num_nodes = num_nodes.cumsum(dim=0)
            adj = torch.zeros((x.size(0), x.size(0)), dtype=torch.float, device=x.device)
            for idx_i, idx_j in zip(shift_cum_num_nodes, cum_num_nodes):
                adj[idx_i:idx_j, idx_i:idx_j] = 1.0
            new_edge_index, _ = dense_to_sparse(adj)
            row, col = new_edge_index
            weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop)
            adj[row, col] = weights
            induced_row, induced_col = induced_edge_index
            adj[induced_row, induced_col] += induced_edge_attr * self.lamb
            weights = adj[row, col]
            if self.sparse:
                new_edge_attr = self.sparse_attention(weights, row)
            else:
                new_edge_attr = softmax(weights, row, num_nodes=x.size(0))
            adj[row, col] = new_edge_attr
            new_edge_index, new_edge_attr = dense_to_sparse(adj)
            del adj
            torch.cuda.empty_cache()
        return x, new_edge_index, new_edge_attr, batch

class HierarchicalGCN_HGPSL(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, pool_ratio):
        super(HierarchicalGCN_HGPSL, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool1 = HGPSLPool(hidden_channels, ratio=pool_ratio, sample=False, sparse=False, sl=True, lamb=1.0, negative_slop=0.2)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool2 = HGPSLPool(hidden_channels, ratio=pool_ratio, sample=False, sparse=False, sl=True, lamb=1.0, negative_slop=0.2)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        self.lin1 = torch.nn.Linear(out_channels, 32)
        self.lin2 = torch.nn.Linear(32, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr=None
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch = self.pool1(x, edge_index, edge_attr, batch)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch = self.pool2(x, edge_index, edge_attr, batch)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x, mask = to_dense_batch(x, batch)
        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

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
class Net_Diff(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, pool_ratio):
        super().__init__()
        num_nodes = 64
        self.gnn1_pool = GNN(in_channels, 64, num_nodes)
        self.gnn1_embed = DenseGCNConv(in_channels, 64)
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
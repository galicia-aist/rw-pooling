import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import subgraph
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, add_self_loops
from torch_scatter import scatter_add
from torch_sparse import coalesce, transpose
import numpy as np


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
            # SGC-like
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
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

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0,
                                                           num_nodes)  # in case all the edges are removed

        edge_index = edge_index.type(torch.long)
        row, col = edge_index
        # print(row, col)
        # print(edge_weight.shape, row.shape, num_nodes)
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # row, col = edge_index
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


class graph_attention(torch.nn.Module):
    # reference: https://github.com/gordicaleksa/pytorch-GAT/blob/39c8f0ee634477033e8b1a6e9a6da3c7ed71bbd1/models/definitions/GAT.py#L324
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0  # node dimension/axis
    head_dim = 1  # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, dropout_prob=0.6, log_attention_weights=False):
        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
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
        #
        # Step 1: Linear Projection + regularization
        #

        in_nodes_features = x  # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target,
                                                                                           nodes_features_proj,
                                                                                           edge_index)
        scores_per_edge = scores_source_lifted + scores_target_lifted

        return torch.sigmoid(scores_per_edge)

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted


class CoPooling(torch.nn.Module):
    def __init__(self, ratio=0.8, K=0.05, edge_ratio=0.6,
                 nhid=64, alpha=0.1, Init=None, Gamma=None):
        super(CoPooling, self).__init__()

        self.ratio = ratio
        self.edge_ratio = edge_ratio

        self.calc_information_score = NodeInformationScore()
        self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        score_dim = 32
        self.G_att = graph_attention(
            num_in_features=nhid,
            num_out_features=score_dim,
            num_of_heads=1
        )

        self.weight = Parameter(torch.Tensor(2 * nhid, nhid))
        self.bias = Parameter(torch.Tensor(nhid))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        self.prop1.reset_parameters()
        self.G_att.init_params()

    def forward(self, x, edge_index, edge_attr=None,
                batch=None, nodes_index=None, node_attr=None):

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        num_nodes = x.size(0)

        # -------------------------------------------------
        # 1. Feature propagation (GPR)
        # -------------------------------------------------
        x_cut = self.prop1(x, edge_index)

        # -------------------------------------------------
        # 2. Edge attention scoring
        # -------------------------------------------------
        attention = self.G_att(x_cut, edge_index)
        attention = attention.sum(dim=1)

        edge_index, attention = add_self_loops(
            edge_index, attention, fill_value=1.0, num_nodes=num_nodes
        )

        # Symmetrize adjacency
        edge_index_t, attention_t = transpose(
            edge_index, attention, num_nodes, num_nodes
        )

        edge_index = torch.cat([edge_index, edge_index_t], dim=1)
        attention = torch.cat([attention, attention_t], dim=0)

        edge_index, attention = coalesce(
            edge_index, attention, num_nodes, num_nodes, op='mean'
        )

        # -------------------------------------------------
        # 3. Edge pruning
        # -------------------------------------------------
        cut_val = torch.quantile(
            attention, 1 - self.edge_ratio
        )

        mask = attention >= cut_val
        cut_edge_index = edge_index[:, mask]
        cut_edge_attr = attention[mask]

        # -------------------------------------------------
        # 4. Node scoring
        # -------------------------------------------------
        x_info = self.calc_information_score(
            x, cut_edge_index, cut_edge_attr
        )

        score = torch.sum(torch.abs(x_info), dim=1)

        # -------------------------------------------------
        # 5. Batch-safe TopK pooling  ✅ FIXED
        # -------------------------------------------------
        perm_list = []

        for b in batch.unique():
            mask_b = (batch == b)
            scores_b = score[mask_b]
            nodes_b = mask_b.nonzero(as_tuple=False).view(-1)

            k = max(1, int(self.ratio * scores_b.size(0)))
            topk_b = torch.topk(scores_b, k=k).indices

            perm_list.append(nodes_b[topk_b])

        perm = torch.cat(perm_list, dim=0)

        # Apply node selection
        x_topk = x[perm]
        batch = batch[perm]

        if nodes_index is not None:
            nodes_index = nodes_index[perm]

        if node_attr is not None:
            node_attr = node_attr[perm]

        # -------------------------------------------------
        # 6. Induced subgraph  ✅ modern PyG
        # -------------------------------------------------
        if cut_edge_index is not None and cut_edge_index.numel() > 0:
            induced_edge_index, induced_edge_attr = subgraph(
                perm,
                cut_edge_index,
                cut_edge_attr,
                relabel_nodes=True,
                num_nodes=num_nodes
            )
        else:
            print("All edges are cut!")
            induced_edge_index = cut_edge_index
            induced_edge_attr = cut_edge_attr

        # -------------------------------------------------
        # 7. Feature update
        # -------------------------------------------------
        # ⚠️ Dense adjacency (OK for small graphs)
        attention_dense = to_dense_adj(
            cut_edge_index,
            edge_attr=cut_edge_attr,
            max_num_nodes=num_nodes
        ).squeeze(0)

        x = F.relu(
            torch.matmul(
                torch.cat(
                    (x_topk, torch.matmul(attention_dense[perm], x)),
                    dim=1
                ),
                self.weight
            ) + self.bias
        )

        return (
            x,
            induced_edge_index,
            induced_edge_attr,
            batch,
            nodes_index,
            node_attr,
            attention_dense
        )
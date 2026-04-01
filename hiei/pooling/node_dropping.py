import torch
from torch_geometric.utils import to_dense_batch


def uniform_sample_perm(batch, ratio):
    """
    Uniformly sample exactly ceil(ratio * N_g) nodes for each graph g in the batch,
    without any Python loop over graphs.

    This is vectorized:
      1) generate i.i.d. random scores
      2) convert to dense batch
      3) take per-graph top-k random scores
    """
    device = batch.device
    num_nodes = batch.numel()

    node_ids = torch.arange(num_nodes, device=device)
    rand_scores = torch.rand(num_nodes, device=device)

    node_ids_dense, mask = to_dense_batch(node_ids, batch, fill_value=-1)
    rand_dense, _ = to_dense_batch(rand_scores, batch, fill_value=-1.0)

    rand_dense = rand_dense.masked_fill(~mask, -1.0)

    num_nodes_per_graph = mask.sum(dim=1)
    k = torch.ceil(num_nodes_per_graph.float() * ratio).long().clamp(min=1)
    max_k = int(k.max().item())

    topk_pos = rand_dense.topk(k=max_k, dim=1, largest=True, sorted=False).indices
    chosen_node_ids = node_ids_dense.gather(1, topk_pos)

    chosen_mask = torch.arange(max_k, device=device).unsqueeze(0) < k.unsqueeze(1)
    perm = chosen_node_ids[chosen_mask]

    return perm


def uniform_pool(x, edge_index, batch, ratio):
    """
    Fast vectorized uniform node dropping.
    No 'for graph in batch'.
    """
    perm = uniform_sample_perm(batch, ratio)

    keep_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    keep_mask[perm] = True

    edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
    pooled_edge_index = edge_index[:, edge_mask]

    new_index = -torch.ones(x.size(0), dtype=torch.long, device=x.device)
    new_index[perm] = torch.arange(perm.numel(), device=x.device)

    pooled_edge_index = new_index[pooled_edge_index]

    x_pool = x[perm]
    batch_pool = batch[perm]

    return x_pool, pooled_edge_index, batch_pool
import torch
from torch_geometric.utils import coalesce


def countsketch_pool(x, edge_index, batch, ratio, edge_weight=None):
    """
    Fast CountSketch graph pooling without explicitly constructing S.

    For each node i:
      - sample a cluster hash h(i)
      - sample a sign s(i) in {-1, +1}

    Then:
      X' = S X       via index_add_
      A' = S A S^T   via edge scatter + coalesce

    This is sparse, vectorized, and loop-free over graphs.
    """
    device = x.device
    num_nodes = x.size(0)
    num_graphs = int(batch.max().item()) + 1

    if edge_weight is None:
        edge_weight = x.new_ones(edge_index.size(1))

    num_nodes_per_graph = torch.bincount(batch, minlength=num_graphs)
    num_clusters_per_graph = torch.ceil(num_nodes_per_graph.float() * ratio).long().clamp(min=1)

    cluster_offsets = torch.cat([
        num_clusters_per_graph.new_zeros(1),
        num_clusters_per_graph.cumsum(dim=0)[:-1],
    ], dim=0)

    local_num_clusters = num_clusters_per_graph[batch]
    local_offsets = cluster_offsets[batch]

    # Uniform hash into per-graph clusters:
    h_local = torch.floor(torch.rand(num_nodes, device=device) * local_num_clusters.float()).long()
    signs = torch.where(
        torch.rand(num_nodes, device=device) < 0.5,
        x.new_full((num_nodes,), -1.0),
        x.new_full((num_nodes,), 1.0),
    )

    cluster = local_offsets + h_local
    total_clusters = int(num_clusters_per_graph.sum().item())

    # X' = S X
    x_pool = x.new_zeros(total_clusters, x.size(-1))
    x_pool.index_add_(0, cluster, x * signs.unsqueeze(-1))

    # cluster sizes, for light normalization
    cluster_sizes = x.new_zeros(total_clusters)
    cluster_sizes.index_add_(0, cluster, x.new_ones(num_nodes))
    nonempty = cluster_sizes > 0

    x_pool[nonempty] = x_pool[nonempty] / cluster_sizes[nonempty].sqrt().unsqueeze(-1)

    # A' = S A S^T, done edge-wise
    row, col = edge_index
    pooled_row = cluster[row]
    pooled_col = cluster[col]
    pooled_weight = signs[row] * signs[col] * edge_weight

    pooled_edge_index = torch.stack([pooled_row, pooled_col], dim=0)
    pooled_edge_index, pooled_weight = coalesce(
        pooled_edge_index,
        pooled_weight,
        num_nodes=total_clusters,
        reduce='sum',
    )

    cluster_batch = torch.repeat_interleave(
        torch.arange(num_graphs, device=device),
        num_clusters_per_graph,
    )

    if not bool(nonempty.all()):
        remap = -torch.ones(total_clusters, dtype=torch.long, device=device)
        remap[nonempty] = torch.arange(int(nonempty.sum().item()), device=device)

        pooled_edge_index = remap[pooled_edge_index]
        valid = (pooled_edge_index[0] >= 0) & (pooled_edge_index[1] >= 0)
        pooled_edge_index = pooled_edge_index[:, valid]
        pooled_weight = pooled_weight[valid]

        x_pool = x_pool[nonempty]
        cluster_batch = cluster_batch[nonempty]
        cluster_sizes = cluster_sizes[nonempty]

    # Mild normalization of pooled weights
    if pooled_edge_index.numel() > 0:
        row, col = pooled_edge_index
        norm = (cluster_sizes[row].sqrt() * cluster_sizes[col].sqrt()).clamp(min=1.0)
        pooled_weight = pooled_weight / norm

    return x_pool, pooled_edge_index, cluster_batch, pooled_weight
import torch
import torch_geometric.data
import torch_geometric.transforms as T
from ogb.graphproppred import PygGraphPropPredDataset

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader, DenseDataLoader


def _register_safe_globals():
    safe = [torch_geometric.data.data.Data]

    for name in ['DataEdgeAttr', 'DataTensorAttr']:
        if hasattr(torch_geometric.data.data, name):
            safe.append(getattr(torch_geometric.data.data, name))

    if hasattr(torch_geometric.data, 'storage'):
        for name in ['GlobalStorage', 'NodeStorage', 'EdgeStorage']:
            if hasattr(torch_geometric.data.storage, name):
                safe.append(getattr(torch_geometric.data.storage, name))

    # torch.serialization.add_safe_globals(safe)


_register_safe_globals()


def load_dataset(args):
    # =========================
    # OGB
    # =========================
    if args.dataset.startswith("ogbg-"):
        dataset = PygGraphPropPredDataset(name=args.dataset, root=args.root)

        sparse_graphs, dense_graphs, in_channels, max_nodes = build_graphs(
            dataset,
            max_nodes=args.max_nodes,
            quantile=args.quantile,
        )

        if dataset.task_type == "multiclass classification":
            num_classes = dataset.num_classes
        else:
            num_classes = dataset.num_tasks

        split_list = make_ogb_split_list(dataset, args.runs)

        return (
            sparse_graphs,
            dense_graphs,
            in_channels,
            num_classes,
            max_nodes,
            split_list,
        )

    # =========================
    # TU
    # =========================
    else:
        dataset = TUDataset(
            root=args.root,
            name=args.dataset,
            use_node_attr=True,
        )

        sparse_graphs, dense_graphs, in_channels, max_nodes = build_graphs(
            dataset,
            max_nodes=args.max_nodes,
            quantile=args.quantile,
        )

        num_classes = dataset.num_classes

        split_list = make_TU_split_list(
            num_graphs=len(sparse_graphs),
            seed=args.seed,
            runs=args.runs
        )

        return (
            sparse_graphs,
            dense_graphs,
            in_channels,
            num_classes,
            max_nodes,
            split_list,
        )

def build_graphs(dataset, max_nodes=None, quantile=0.95):
    add_const = T.Constant(value=1.0, cat=False) if dataset.num_features == 0 else None

    sizes = torch.tensor([data.num_nodes for data in dataset], dtype=torch.float)

    if max_nodes is None:
        max_nodes = max(1, int(torch.quantile(sizes, quantile).item()))

    to_dense = T.ToDense(max_nodes)

    sparse_graphs = []
    dense_graphs = []

    for data in dataset:
        data = data.clone()

        if add_const is not None:
            data = add_const(data)

        if data.num_nodes > max_nodes:
            continue

        sparse_graphs.append(data)

        dense_data = data.clone()

        # remove edge_attr for dense
        if getattr(dense_data, 'edge_attr', None) is not None:
            dense_data.edge_attr = None

        dense = to_dense(dense_data)

        # ensure adjacency is 2D
        if dense.adj.dim() == 3:
            dense.adj = (dense.adj.abs().sum(dim=-1) > 0).float()

        dense_graphs.append(dense)

    if len(sparse_graphs) == 0:
        raise ValueError(f'No graph left after filtering with max_nodes={max_nodes}.')

    in_channels = sparse_graphs[0].num_features

    return sparse_graphs, dense_graphs, in_channels, max_nodes

def make_ogb_split_list(dataset, runs):
    split_idx = dataset.get_idx_split()

    split = (
        split_idx["train"].tolist(),
        split_idx["valid"].tolist(),
        split_idx["test"].tolist(),
    )

    # same split repeated for each run
    return [split for _ in range(runs)]


def make_TU_split_list(num_graphs, seed, runs, train_ratio=0.8, val_ratio=0.1):
    split_list = []

    for run_id in range(runs):
        g = torch.Generator().manual_seed(seed + run_id)
        perm = torch.randperm(num_graphs, generator=g).tolist()

        n_train = max(1, int(num_graphs * train_ratio))
        n_val = max(1, int(num_graphs * val_ratio))

        if n_train + n_val >= num_graphs:
            n_train = max(1, num_graphs - 2)
            n_val = 1

        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        split_list.append((train_idx, val_idx, test_idx))

    return split_list


def _select(graphs, indices):
    return [graphs[i] for i in indices]


def make_loaders(
    sparse_graphs,
    dense_graphs,
    split_indices,
    batch_size=32,
):
    train_idx, val_idx, test_idx = split_indices

    sparse_train = _select(sparse_graphs, train_idx)
    sparse_val = _select(sparse_graphs, val_idx)
    sparse_test = _select(sparse_graphs, test_idx)

    dense_train = _select(dense_graphs, train_idx)
    dense_val = _select(dense_graphs, val_idx)
    dense_test = _select(dense_graphs, test_idx)

    sparse_loaders = (
        DataLoader(sparse_train, batch_size=batch_size, shuffle=True),
        DataLoader(sparse_val, batch_size=batch_size, shuffle=False),
        DataLoader(sparse_test, batch_size=batch_size, shuffle=False),
    )

    dense_loaders = (
        DenseDataLoader(dense_train, batch_size=batch_size, shuffle=True),
        DenseDataLoader(dense_val, batch_size=batch_size, shuffle=False),
        DenseDataLoader(dense_test, batch_size=batch_size, shuffle=False),
    )

    return sparse_loaders, dense_loaders
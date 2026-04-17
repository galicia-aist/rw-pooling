import torch
import torch_geometric.data
import torch_geometric.transforms as T
from ogb.graphproppred import PygGraphPropPredDataset

from torch_geometric.datasets import TUDataset, MoleculeNet
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


def load_dataset(args, use_dense):
    # =========================
    # OGB
    # =========================
    if args.dataset.startswith("ogbg-"):
        dataset = PygGraphPropPredDataset(name=args.dataset, root=args.root)

        if use_dense:
            graphs, in_channels, max_nodes = build_dense_graphs(dataset, max_nodes=args.max_nodes, quantile=args.quantile)
        else:
            graphs, in_channels = build_sparse_graphs(dataset)
            max_nodes = None

        if dataset.task_type == "multiclass classification":
            num_classes = dataset.num_classes
        else:
            num_classes = dataset.num_tasks

        split_list = make_ogb_split_list(dataset, args.runs)

        return (
            graphs,
            in_channels,
            num_classes,
            max_nodes,
            split_list,
        )
    elif args.dataset in ["ESOL", "FreeSolv", "Lipo", "QM7", "QM8"]:
        dataset = MoleculeNet(root=args.root, name=args.dataset)

        if use_dense:
            graphs, in_channels, max_nodes = build_dense_graphs(dataset, max_nodes=args.max_nodes,
                                                                quantile=args.quantile)
        else:
            graphs, in_channels = build_sparse_graphs(dataset)
            max_nodes = None

        # ✅ regression → always 1 output
        num_classes = 1  # usually 1, but safer than hardcoding

        split_list = make_TU_split_list(
            num_graphs=len(graphs),
            seed=args.seed,
            runs=args.runs
        )

        return (
            graphs,
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

        if use_dense:
            graphs, in_channels, max_nodes = build_dense_graphs(dataset, max_nodes=args.max_nodes,
                                                                quantile=args.quantile)
        else:
            graphs, in_channels = build_sparse_graphs(dataset)
            max_nodes = None

        if args.dataset in ["TRIANGLES", "ZINC_full"]:
            num_classes = 1
        else:
            num_classes = dataset.num_classes

        split_list = make_TU_split_list(
            num_graphs=len(graphs),
            seed=args.seed,
            runs=args.runs
        )

        return (
            graphs,
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

def build_sparse_graphs(dataset):
    import torch_geometric.transforms as T

    add_const = T.Constant(value=1.0, cat=False) if dataset.num_features == 0 else None

    sparse_graphs = []

    for data in dataset:
        data = data.clone()

        if add_const is not None:
            data = add_const(data)

        sparse_graphs.append(data)

    if len(sparse_graphs) == 0:
        raise ValueError("Dataset is empty.")

    in_channels = sparse_graphs[0].num_features

    return sparse_graphs, in_channels

def build_dense_graphs(dataset, max_nodes=None, quantile=0.95):

    add_const = T.Constant(value=1.0, cat=False) if dataset.num_features == 0 else None

    sizes = torch.tensor([data.num_nodes for data in dataset], dtype=torch.float)

    if max_nodes is None:
        max_nodes = max(1, int(torch.quantile(sizes, quantile).item()))

    to_dense = T.ToDense(max_nodes)

    dense_graphs = []

    for data in dataset:
        if data.num_nodes > max_nodes:
            continue  # ✅ REQUIRED for dense

        data = data.clone()

        if add_const is not None:
            data = add_const(data)

        # remove edge_attr for dense
        if getattr(data, 'edge_attr', None) is not None:
            data.edge_attr = None

        dense = to_dense(data)

        # ensure adjacency is 2D
        if dense.adj.dim() == 3:
            dense.adj = (dense.adj.abs().sum(dim=-1) > 0).float()

        dense_graphs.append(dense)

    if len(dense_graphs) == 0:
        raise ValueError(f"No graph left after filtering with max_nodes={max_nodes}.")

    in_channels = dense_graphs[0].x.size(-1)

    return dense_graphs, in_channels, max_nodes

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
    graphs,
    split_indices,
    batch_size=32,
    dense=False,
):
    train_idx, val_idx, test_idx = split_indices

    train_graphs = _select(graphs, train_idx)
    val_graphs = _select(graphs, val_idx)
    test_graphs = _select(graphs, test_idx)

    Loader = DenseDataLoader if dense else DataLoader

    train_loader = Loader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = Loader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = Loader(test_graphs, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
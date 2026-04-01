import argparse
import time

import torch
import torch.nn.functional as F
import torch_geometric.data
torch.serialization.add_safe_globals([torch_geometric.data.data.Data])

from data import load_tu_graphs, make_loaders, make_split_indices
from models import AVAILABLE_METHODS, DENSE_METHODS, build_model
from utils import evaluate, save_results_csv, set_seed, summarize


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def parse_methods(text):
    text = text.strip().lower()
    if text == 'all':
        return list(AVAILABLE_METHODS)

    methods = [x.strip() for x in text.split(',') if x.strip()]
    unknown = [m for m in methods if m not in AVAILABLE_METHODS]
    if unknown:
        raise ValueError(f'Unknown methods: {unknown}. Available: {AVAILABLE_METHODS}')
    return methods


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    start = time.perf_counter()

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        logits, aux_loss = model(batch)
        y = batch.y.view(-1)

        loss = F.cross_entropy(logits, y) + aux_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.numel()

    train_time = time.perf_counter() - start
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--methods', type=str, default='all')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--pool-ratio', type=float, default=0.5)
    parser.add_argument('--max-nodes', type=int, default=None)
    parser.add_argument('--quantile', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--out-csv', type=str, default='results_proteins.csv')
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    device = get_device()

    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

    sparse_graphs, dense_graphs, in_channels, num_classes, max_nodes = load_tu_graphs(
        root=args.root,
        name=args.dataset,
        max_nodes=args.max_nodes,
        quantile=args.quantile,
        use_node_attr=True,
    )

    print(
        f'Dataset loaded | name={args.dataset} | graphs={len(sparse_graphs)} | '
        f'in_channels={in_channels} | num_classes={num_classes} | '
        f'max_nodes={max_nodes} | device={device}'
    )

    split_list = [
        make_split_indices(len(sparse_graphs), seed=args.seed + run_id)
        for run_id in range(args.runs)
    ]

    rows = []

    for method in methods:
        print(f'\nTraining method: {method}')

        run_accs = []
        run_train_times = []

        for run_id in range(args.runs):
            print(f'Run {run_id + 1}/{args.runs}')

            seed = args.seed + run_id
            set_seed(seed)

            sparse_loaders, dense_loaders = make_loaders(
                sparse_graphs=sparse_graphs,
                dense_graphs=dense_graphs,
                split_indices=split_list[run_id],
                batch_size=args.batch_size,
            )

            if method in DENSE_METHODS:
                train_loader, val_loader, test_loader = dense_loaders
            else:
                train_loader, val_loader, test_loader = sparse_loaders

            model = build_model(
                method=method,
                in_channels=in_channels,
                hidden_channels=args.hidden,
                num_classes=num_classes,
                max_nodes=max_nodes,
                pool_ratio=args.pool_ratio,
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            best_val = 0.0
            best_test = 0.0
            cumulative_train_time = 0.0

            for epoch in range(1, args.epochs + 1):
                loss, epoch_train_time = train_one_epoch(model, train_loader, optimizer, device)
                cumulative_train_time += epoch_train_time

                val_acc = evaluate(model, val_loader, device)

                if val_acc >= best_val:
                    best_val = val_acc
                    best_test = evaluate(model, test_loader, device)

                if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                    print(
                        f'  Epoch {epoch:03d}/{args.epochs:03d} | '
                        f'loss={loss:.4f} | val={val_acc:.4f} | '
                        f'best_test={best_test:.4f} | '
                        f'epoch_train_time={epoch_train_time:.2f}s | '
                        f'cum_train_time={cumulative_train_time:.2f}s'
                    )

            run_accs.append(best_test)
            run_train_times.append(cumulative_train_time)

            rows.append({
                'method': method,
                'row_type': 'run',
                'run': run_id + 1,
                'accuracy': f'{best_test:.6f}',
                'train_time': f'{cumulative_train_time:.6f}',
                'acc_mean': '',
                'acc_std': '',
                'time_mean': '',
                'time_std': '',
            })

            print(
                f'  Run done | method={method} | '
                f'accuracy={best_test:.4f} | train_time={cumulative_train_time:.2f}s'
            )

        acc_mean, acc_std = summarize(run_accs)
        time_mean, time_std = summarize(run_train_times)

        rows.append({
            'method': method,
            'row_type': 'summary',
            'run': f'mean_over_{args.runs}_runs',
            'accuracy': '',
            'train_time': '',
            'acc_mean': f'{acc_mean:.6f}',
            'acc_std': f'{acc_std:.6f}',
            'time_mean': f'{time_mean:.6f}',
            'time_std': f'{time_std:.6f}',
        })

        print(
            f'Summary | method={method} | '
            f'acc_mean={acc_mean:.4f} | acc_std={acc_std:.4f} | '
            f'time_mean={time_mean:.2f}s | time_std={time_std:.2f}s'
        )

    save_results_csv(args.out_csv, rows)
    print(f'\nSaved CSV: {args.out_csv}')


if __name__ == '__main__':
    main()
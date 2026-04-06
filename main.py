# torch.serialization.add_safe_globals([torch_geometric.data.data.Data])
from datetime import datetime

from data import make_loaders, load_dataset
from models import build_model
from utils import *
from trainers import train_one_epoch

def main(args, device, method, timestamp, logger=None):

    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

    logger.debug("start loading dataset")

    sparse_graphs, dense_graphs, in_channels, num_classes, max_nodes, split_list = load_dataset(args)

    logger.debug("start loading dataset")

    logger.info(
        f'Dataset loaded | name={args.dataset} | graphs={len(sparse_graphs)} | '
        f'in_channels={in_channels} | num_classes={num_classes} | '
        f'max_nodes={max_nodes} | device={device}'
    )

    rows = []


    logger.info(f'\nTraining pooling method: {method}')

    DENSE_METHODS = {'diffpool'}

    run_accs = []
    run_train_times = []

    for run_id in range(args.runs):
        logger.info(f'Run {run_id + 1}/{args.runs}')

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
                logger.info(
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

        logger.info(
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

    logger.info(
        f'Summary | method={method} | '
        f'acc_mean={acc_mean:.4f} | acc_std={acc_std:.4f} | '
        f'time_mean={time_mean:.2f}s | time_std={time_std:.2f}s'
    )

    save_results_csv(args.pmethod, args.dataset, args.exp_name, timestamp, rows, logger=logger)


if __name__ == '__main__':
    args = get_args()
    device = get_device()
    method = args.pmethod
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger_settings = {
        "logger": {
            "model": args.pmethod,
            "log_path": args.log_path,
            "dataset": args.dataset,
            "log_level": args.log_level.upper()
        },
        # "ddp": args.ddp
    }

    with open("global_settings.json", "w") as file:
        json.dump(logger_settings, file, indent=4)

    logger = get_logger(args.exp_name, timestamp)
    log_experiment_settings(logger, args)



    main(args, device, method, timestamp, logger=logger)
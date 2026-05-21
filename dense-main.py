import json
import os
from datetime import datetime

from dense_pooling_models import Net_Diff
from dense_trainers import train, test
from utils import get_args, get_logger, log_experiment_settings

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import random
import time
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader


args = get_args()
dense_methods = [
    "asym",
    "diff",
    "mincut",
    "dmon",
    "hosc",
    "justb",
    "sep",
    "pars"
]
if args.model not in dense_methods:
    raise Exception("Not dense pooling method. Please run main_sparse")
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logger_settings = {
        "logger": {
            "model": args.model,
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

logger = get_logger("sample", timestamp)

max_nodes = 700
data_path = "data"
if args.dataset == "PROTEINS":
    dataset_dense = TUDataset(
        data_path,
        name="PROTEINS",
        transform=T.Compose([T.ToDense(max_nodes)]),
        use_node_attr=True,
        pre_filter=lambda data: data.num_nodes <= max_nodes,
    )
else:
    raise Exception("wrong dataset")

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
model = Net_Diff(dataset_dense.num_features, dataset_dense.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seeds = args.seeds
times = []
memories = []
best_val_accs = []
best_test_accs = []
early_stop_patience = args.early_stop
tolerance = args.tolerance
for seed in seeds:
    set_seed(seed)
    dataset_dense = dataset_dense.shuffle()
    train_ratio = 0.7
    val_ratio = 0.15
    val_ratio = 0.15
    num_total = len(dataset_dense)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)
    num_test = num_total - num_train - num_val
    train_dataset = dataset_dense[:num_train]
    val_dataset = dataset_dense[num_train:num_train + num_val]
    test_dataset = dataset_dense[num_train + num_val:]
    train_loader = DenseDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DenseDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DenseDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = Net_Diff(dataset_dense.num_features, dataset_dense.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_time = time.time()
    best_val_acc = 0
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        loss = train(model, train_loader, optimizer, device)
        val_acc = test(model, valid_loader, device)
        test_acc = test(model, test_loader, device)
        if val_acc > best_val_acc + tolerance:
            best_val_acc = val_acc
            best_test_acc = test_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        logger.info(f'Seed: {seed}, Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        if epochs_no_improve >= early_stop_patience:
            logger.info(f'Early stopping at epoch {epoch} for seed {seed}')
            break
    end_time = time.time()
    total_time = end_time - start_time
    memory_allocated = torch.cuda.memory_reserved(device) / (1024 ** 2)
    times.append(total_time)
    memories.append(memory_allocated)
    best_val_accs.append(best_val_acc)
    best_test_accs.append(best_test_acc)
    torch.cuda.empty_cache()
logger.info(f'Average Time: {np.mean(times):.2f} seconds')
logger.info(f'Var Time: {np.var(times):.2f} seconds')
logger.info(f'Average Memory: {np.mean(memories):.2f} MB')
logger.info(f'Average Best Val Acc: {np.mean(best_val_accs):.4f}')
logger.info(f'Std Best Test Acc: {np.std(best_test_accs):.4f}')
logger.info(f'Average Test Acc: {np.mean(best_test_accs):.4f}')
import os
from datetime import datetime

import numpy as np
import time
import warnings
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import torch
from utils import *
from torch_geometric.data import DataLoader
from sparse_pooling_models import *
from sparse_trainers import train, test, train_ogb, test_ogb
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

args = get_args()
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
max_nodes = args.max_nodes
data_path = "data"
if args.dataset == "COLLAB":
    dataset_sparse = TUDataset(root=data_path, name="COLLAB", transform=T.Compose([T.OneHotDegree(491)]),
                               use_node_attr=True)
elif args.dataset == "IMDB-MULTI":
    dataset_sparse = TUDataset(root=data_path, name="IMDB-MULTI", transform=T.Compose([T.OneHotDegree(88)]),
                               use_node_attr=True)
elif args.dataset in ["ppa", "molpcba", "molhiv"]:
    args.dataset = "ogbg-" + args.dataset
    dataset_sparse = PygGraphPropPredDataset(name=args.dataset, root=data_path)
else:
    dataset_sparse = TUDataset(root=data_path, name=args.dataset, pre_filter=lambda data: data.num_nodes <= max_nodes, use_node_attr=True)
num_classes = dataset_sparse.num_classes
in_channels = dataset_sparse.num_features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.CrossEntropyLoss()


seeds = args.seeds
times = []
memories = []
best_val_accs = []
best_test_accs = []
early_stop_patience = args.early_stop
tolerance = args.tolerance

# Read JSON file
with open("dataset_config.json", "r") as f:
    DATASET_CONFIGS = json.load(f)

# Example access
cfg = DATASET_CONFIGS[args.dataset]


for seed in seeds:
    set_seed(seed)
    dataset_sparse = dataset_sparse.shuffle()
    if "ogb" in args.dataset:
        split_idx = dataset_sparse.get_idx_split()
        dataset_sparse.shuffle()
        train_dataset = dataset_sparse[split_idx["train"]]
        val_dataset = dataset_sparse[split_idx["valid"]]
        test_dataset = dataset_sparse[split_idx["test"]]
        evaluator = Evaluator(name=args.dataset)
    else:
        train_ratio = 0.7
        val_ratio = 0.15
        num_total = len(dataset_sparse)
        num_train = int(num_total * train_ratio)
        num_val = int(num_total * val_ratio)
        num_test = num_total - num_train - num_val
        train_dataset = dataset_sparse[:num_train]
        val_dataset = dataset_sparse[num_train:num_train + num_val]
        test_dataset = dataset_sparse[num_train + num_val:]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if args.model == "topk":
        model = HierarchicalGCN_TOPK(in_channels=dataset_sparse.num_features, hidden_channels=cfg["hidden_channels"],
                                     out_channels=cfg["out_channels"], mlp_hidden=cfg["mlp_hidden"],
                                     num_classes=dataset_sparse.num_classes,pool_ratio=args.pratio,
                                     dataset_name=args.dataset).to(device)
    elif args.model == "sag":
        model = HierarchicalGCN_SAG(in_channels=dataset_sparse.num_features, hidden_channels=64, out_channels=64,
                                    num_classes=dataset_sparse.num_classes, pool_ratio=args.pratio).to(device)
    elif args.model == "asap":
        model = HierarchicalGCN_ASA(in_channels=dataset_sparse.num_features, hidden_channels=64, out_channels=64,
                                    num_classes=dataset_sparse.num_classes, pool_ratio=args.pratio).to(device)
    elif args.model == "pan":
        model = HierarchicalGCN_PAN(in_channels=dataset_sparse.num_features, hidden_channels=64, out_channels=64,
                                    num_classes=dataset_sparse.num_classes, pool_ratio=args.pratio).to(device)
    elif args.model == "cop":
        model = HierarchicalGCN_CO(in_channels=dataset_sparse.num_features, hidden_channels=64, out_channels=64,
                                   num_classes=dataset_sparse.num_classes, pool_ratio=args.pratio).to(device)
    elif args.model == "cgi":
        model = HierarchicalGCN_CGI(in_channels=dataset_sparse.num_features, hidden_channels=64,out_channels=64,
                                    num_classes=dataset_sparse.num_classes, pool_ratio=args.pratio).to(device)
    elif args.model == "kmis":
        model = HierarchicalGCN_KMIS(in_channels=dataset_sparse.num_features, hidden_channels=64,out_channels=64,
                                     num_classes=dataset_sparse.num_classes).to(device)
    elif args.model == "gsa":
        model = HierarchicalGCN_GSA(in_channels=dataset_sparse.num_features, hidden_channels=64, out_channels=64,
                                    num_classes=dataset_sparse.num_classes, pool_ratio=args.pratio).to(device)
    elif args.model == "hgpsl":
        model = HierarchicalGCN_HGPSL(in_channels=dataset_sparse.num_features, hidden_channels=64, out_channels=64,
                                      num_classes=dataset_sparse.num_classes, pool_ratio=args.pratio).to(device)
    elif args.model == "ndp":
        model = HierarchicalGCN_NDP(in_channels=dataset_sparse.num_features, hidden_channels=64, out_channels=64,
                                      num_classes=dataset_sparse.num_classes, pool_ratio=args.pratio).to(device)
    elif args.model == "graclus":
        model = HierarchicalGCN_GRACLUS(in_channels=dataset_sparse.num_features, hidden_channels=64, out_channels=64,
                                      num_classes=dataset_sparse.num_classes).to(device)
    else:
        raise Exception("Incorrect model")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_time = time.time()
    best_val_acc = 0
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        logger.debug(f"Current epoch: {epoch}")
        if "ogb" in args.dataset:
            loss = train_ogb(model, optimizer, train_loader, device, criterion)
            val_acc = test_ogb(valid_loader, model, device)
            test_acc = test_ogb(test_loader, model, device)
        else:
            loss = train(model, optimizer, train_loader, device)
            val_acc = test(valid_loader, model, device)
            test_acc = test(test_loader, model, device)
        if val_acc > best_val_acc + tolerance:
            best_val_acc = val_acc
            best_test_acc = test_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            logger.debug(f'Early stopping at epoch {epoch} for seed {seed}')
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
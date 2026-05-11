import os
import numpy as np
import time
import warnings
from torch_geometric.datasets import TUDataset
import torch
from utils import *
from torch_geometric.data import DataLoader
from pooling_models import HierarchicalGCN_TOPK
from trainers import train, test

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
max_nodes = 500
data_path = "data"
dataset_sparse = TUDataset(root=data_path, name=args.dataset, pre_filter=lambda data: data.num_nodes <= max_nodes, use_node_attr=True)
num_classes = dataset_sparse.num_classes
in_channels = dataset_sparse.num_features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HierarchicalGCN_TOPK(in_channels=dataset_sparse.num_features, hidden_channels=64,out_channels=64, num_classes=dataset_sparse.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


seeds = [42, 43, 44]
times = []
memories = []
best_val_accs = []
best_test_accs = []
early_stop_patience = 150
tolerance = 0.0001
for seed in seeds:
    set_seed(seed)
    dataset_sparse = dataset_sparse.shuffle()
    train_ratio = 0.7
    val_ratio = 0.15
    val_ratio = 0.15
    num_total = len(dataset_sparse)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)
    num_test = num_total - num_train - num_val
    train_dataset = dataset_sparse[:num_train]
    val_dataset = dataset_sparse[num_train:num_train + num_val]
    test_dataset = dataset_sparse[num_train + num_val:]
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    model = HierarchicalGCN_TOPK(in_channels=dataset_sparse.num_features, hidden_channels=64,out_channels=64, num_classes=dataset_sparse.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    best_val_acc = 0
    epochs_no_improve = 0
    for epoch in range(1, 201):
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
            print(f'Early stopping at epoch {epoch} for seed {seed}')
            break
    end_time = time.time()
    total_time = end_time - start_time
    memory_allocated = torch.cuda.memory_reserved(device) / (1024 ** 2)
    times.append(total_time)
    memories.append(memory_allocated)
    best_val_accs.append(best_val_acc)
    best_test_accs.append(best_test_acc)
    torch.cuda.empty_cache()
print(f'Average Time: {np.mean(times):.2f} seconds')
print(f'Var Time: {np.var(times):.2f} seconds')
print(f'Average Memory: {np.mean(memories):.2f} MB')
print(f'Average Best Val Acc: {np.mean(best_val_accs):.4f}')
print(f'Std Best Test Acc: {np.std(best_test_accs):.4f}')
print(f'Average Test Acc: {np.mean(best_test_accs):.4f}')
import argparse
import csv
import json
import logging
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

DATASET_TASK = {

    # =========================
    # Graph Multi-class Classification
    # =========================
    "ENZYMES": "graph_multiclass",
    "MUTAG": "graph_multiclass",
    "PROTEINS": "graph_multiclass",
    "NCI1": "graph_multiclass",
    "NCI109": "graph_multiclass",
    "PTC_MR": "graph_multiclass",
    "FRANKENSTEIN": "graph_multiclass",
    "D&D": "graph_multiclass",
    "MUTAGENICITY": "graph_multiclass",
    "IMDB-MULTI": "graph_multiclass",
    "COLLAB": "graph_multiclass",
    "COLORS-3": "graph_multiclass",
    "TRIANGLES": "graph_regression",
    "ogbg-ppa": "graph_multiclass",

    # =========================
    # Graph Binary Classification
    # =========================
    "IMDB-BINARY": "graph_binary",
    "REDDIT-BINARY": "graph_binary",
    "REDDIT-M5K": "graph_binary",
    "REDDIT-M12K": "graph_binary",
    "ogbg-moltox21": "graph_binary",
    "ogbg-moltoxcast": "graph_binary",
    "ogbg-molpcba": "graph_binary",

    # OGB molecular datasets
    "ogbg-molhiv": "graph_binary",
    "ogbg-molbbbp": "graph_binary",

    # =========================
    # Graph Regression
    # =========================
    "QM7": "graph_regression",
    "QM8": "graph_regression",
    "ZINC_full": "graph_regression",
    "ESOL": "graph_regression",
    "FREESOLV": "graph_regression",
    "Lipo": "graph_regression",
    "FreeSolv": "graph_regression",
    "ogbg-molesol": "graph_regression",

    # =========================
    # Node Classification
    # =========================
    "Cora": "node_classification",
    "CiteSeer": "node_classification",
    "PubMed": "node_classification",
    "ogbn-proteins": "node_classification",
    "ogbn-products": "node_classification",
    "ogbn-arxiv": "node_classification",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def summarize(values):
    x = torch.tensor(values, dtype=torch.float)
    if x.numel() == 1:
        return x.item(), 0.0
    return x.mean().item(), x.std(unbiased=False).item()


def save_results_csv(pmethod, dataset, exp_name, timestamp, rows, filename_prefix="results", logger=None):
    # Build directory: results/<exp_name>/
    directory = os.path.join("results", exp_name)
    os.makedirs(directory, exist_ok=True)

    # Build file path: placeholdername_YYYYMMDD-HHMMSS.csv
    filename = f"{timestamp}_{filename_prefix}_{pmethod}_{dataset}.csv"
    path = os.path.join(directory, filename)

    fieldnames = [
        'method',
        'row_type',
        'run',
        'accuracy',
        'train_time',
        'acc_mean',
        'acc_std',
        'time_mean',
        'time_std',
    ]

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Results CSV saved to {path}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PROTEINS', choices=DATASET_TASK.keys())
    parser.add_argument('--pmethod', type=str, choices=['mean', 'uniform', 'topk', 'sag', 'diffpool',
                        'countsketch', 'sum', 'asap', 'max', 'edge', 'pan', 'cop', 'cgi', 'kmis', 'gsa', 'hgpsl', 'mincut'],
                        help='Pooling method to use', default='sum')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--pool_ratio', type=float, default=0.5)
    parser.add_argument('--max-nodes', type=int, default=None)
    parser.add_argument('--quantile', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--exp_name', type=str, default="lorem")
    parser.add_argument('--log_path', type=str, default='local', help='Path to store logs. Default is "local".')
    parser.add_argument('--log_level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'], help='Logging level.')

    args = parser.parse_args()

    return args

class CustomFormatter(logging.Formatter):
    """Custom formatter to include the current GPU in log messages with colors."""

    # ANSI color codes
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Log format with GPU info
    format = "%(asctime)s - %(gpu_info)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    # Different colors for different log levels
    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        # Get current GPU info
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_id)
            record.gpu_info = f"GPU: {gpu_id} ({gpu_name})"
        else:
            record.gpu_info = "GPU: CPU"

        # Select the appropriate format based on log level
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger(exp_name, timestamp):
    """Sets up the logger with GPU info and color-coded formatting."""
    with open("global_settings.json", "r") as file:
        loaded_data = json.load(file)

    logger_settings = loaded_data["logger"]
    model = logger_settings["model"]
    dataset_name = logger_settings["dataset"]
    log_level = logger_settings["log_level"]

    # Build logs directory: logs/<exp_name>/
    logs_dir = os.path.join(os.getcwd(), "logs", exp_name)
    os.makedirs(logs_dir, exist_ok=True)

    # Build log filename: model_dataset_expname_timestamp.log
    filename = f"{model}_dataset-{dataset_name}_{exp_name}_{timestamp}.log"
    log_path = os.path.join(logs_dir, filename)

    logger = logging.getLogger(f"{model}_{exp_name}_{timestamp}")

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(CustomFormatter())

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter())

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Set the logger level dynamically
        logger.setLevel(logging.DEBUG if log_level == "DEBUG" else logging.INFO)

    return logger

def log_experiment_settings(logger, args):
    """
    Logs all experiment settings in a nicely formatted table as a single log entry,
    including the PyTorch version.

    Args:
        logger: The logger instance to use.
        args: An argparse.Namespace or any object with attributes to log.
    """
    # Find the longest argument name for alignment
    max_len = max(len(k) for k in vars(args).keys())

    # Build the table as a string
    lines = []
    lines.append("-" * (max_len + 30))
    lines.append(f"{'Argument'.ljust(max_len)} | Value")
    lines.append("-" * (max_len + 30))

    for k, v in vars(args).items():
        lines.append(f"{k.ljust(max_len)} | {v}")

    # Add PyTorch version
    lines.append(f"{'torch_version'.ljust(max_len)} | {torch.__version__}")

    lines.append("-" * (max_len + 30))

    # Join everything into a single string and log once
    logger.info("Experiment settings:\n" + "\n".join(lines))


def get_task_config(dataset_name):
    task = DATASET_TASK[dataset_name]

    # -------------------------
    # Graph Regression
    # -------------------------
    if task == "graph_regression":
        return {
            "task": "regression",
            "loss_fn": F.mse_loss,
            "metric": "mae",
            "out_dim_mode": "regression"
        }

    # -------------------------
    # Graph Binary Classification
    # -------------------------
    elif task == "graph_binary":
        return {
            "task": "binary",
            "loss_fn": nn.BCEWithLogitsLoss(),
            "metric": "rocauc",   # important for OGB
            "out_dim_mode": "binary"
        }

    # -------------------------
    # Graph Multi-class Classification
    # -------------------------
    elif task == "graph_multiclass":
        return {
            "task": "classification",
            "loss_fn": F.cross_entropy,
            "metric": "accuracy",
            "out_dim_mode": "classification"
        }

    # -------------------------
    # Node Classification
    # -------------------------
    elif task == "node_classification":
        return {
            "task": "node_classification",
            "loss_fn": F.cross_entropy,
            "metric": "accuracy",
            "out_dim_mode": "classification"
        }

    else:
        raise ValueError(f"Unknown task type: {task}")

def get_targets(data, task):
    if task == "classification":
        return data.y.view(-1).long()

    elif task == "binary":
        return data.y.float()

    elif task == "regression":
        return data.y.float()

    elif task == "multilabel":
        return data.y.float()

    else:
        raise ValueError(f"Unknown task: {task}")
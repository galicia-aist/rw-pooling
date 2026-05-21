import argparse
import json
import logging
import os

import numpy as np
import random
import torch

def get_args():
    """
    Builds and parses command-line arguments from a JSON configuration file
    (`args-config.json`). It reads each argument’s properties (such as type, default, help text, and flags) from the
     file, adds them to an `ArgumentParser`, and returns the parsed arguments, allowing flexible and easily
     maintainable experiment setups without changing the code.
    """

    with open("args-config.json", 'r') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()

    for arg, props in config.items():
        kwargs = {}

        # Add common fields
        if "help" in props:
            kwargs["help"] = props["help"]
        if "default" in props:
            kwargs["default"] = props["default"]
        if "choices" in props:
            kwargs["choices"] = props["choices"]

        # Determine type
        if "type" in props:
            type_map = {"int": int, "float": float, "str": str, "bool": bool}
            kwargs["type"] = type_map.get(props["type"], str)

        # Handle flags
        if "action" in props:
            kwargs["action"] = props["action"]

        # Handle required flag
        if props.get("required", False):
            kwargs["required"] = True

        parser.add_argument(f'--{arg}', **kwargs)

    args = parser.parse_args()

    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
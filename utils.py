import argparse
import json
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
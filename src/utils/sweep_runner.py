"""Sweep Runner.

Runs the run_training function from a set of command line arguments instead of
from a yaml file. This is useful since sweep agents run from a
"""
import sys

from experiments.train_runner import run_training
from utils.config_parser import train_config, data_config, encoder_config, \
    mt_config, embed_config, optim_loss_config, debug_config


if __name__ == '__main__':
    # Read command line arguments
    args = sys.argv[1:]

    # Set up data_dir config based on system (i.e., mac or windows)
    if sys.platform == "win32":
        data_dir = "F:\\Offline Docs\\Master Thesis\\data"
    elif sys.platform == "darwin":
        data_dir = "../data"
    else:
        raise ValueError("This currently only supports Windows and macOS.")

    # Set up constant configs explicitly, in case defaults change
    config = {"train": {"checkpoint_dir": "../checkpoints",
                        "epochs": 150,
                        "ui": "tqdm"},
              "data": {"data_dir": data_dir},
              "encoder": {},
              "mt": {},
              "embed": {},
              "optim_loss": {},
              "debug": {"is_debug": False,
                        "on_cpu": False}}

    for arg in args:
        if arg.startswith("--"):
            arg = arg[2:]
            arg_split = arg.split("=")
            if len(arg_split) != 2:
                raise ValueError(f"Invalid argument {arg}.")
            config_key, config_value = arg_split
            config_type, config_key = config_key.split(".")
            if config_type not in config:
                raise ValueError(f"Invalid config key "
                                 f"{config_key}.{config_key}.")
            if config_value == "true":
                config_value = True
            elif config_value == "false":
                config_value = False
            elif config_value == "null":
                config_value = None

            # Try converting config_values from string to int
            try:
                config_value = int(config_value)
            except ValueError:
                # Try converting config_values from string to float
                try:
                    config_value = float(config_value)
                except ValueError:
                    pass

            config[config_type][config_key] = config_value
        else:
            raise ValueError(f"Invalid argument {arg}.")

    config_dict = {
        "train": train_config(**config["train"]),
        "data": data_config(**config["data"]),
        "encoder": encoder_config(**config["encoder"]),
        "mt": mt_config(**config["mt"]),
        "embed": embed_config(**config["embed"]),
        "optim_loss": optim_loss_config(**config["optim_loss"]),
        "debug": debug_config(**config["debug"])
    }

    run_training(config_dict)

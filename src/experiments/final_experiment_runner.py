"""Final Experiment Runner.

This script runs all final experiments. These are basically all the config
files that are found in run_configs/final-experiments. The script runs each
config file in order and saves the results to the wandb project, but only
runs those for the correct system. If the system is Windows, it only runs those
config files with suffix `_win.yaml`. If the system is Mac, it only runs does
config files with suffix `_mac.yaml`.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
    GitHub Copilot
"""
from pathlib import Path
import os

from data_utils.pregenerate_transform import pregenerate_transforms
from experiments.train_runner import run_training
from utils.config_parser import parse_config_file


def main():
    run_config_dir = Path(__file__).parent.parent.parent / "run_configs"
    final_configs = run_config_dir / "final-experiments"

    # Get system
    system = os.name

    # Get all config files
    config_files = [f for f in final_configs.iterdir() if f.suffix == ".yaml"]

    # Sort config files and only choose those that are for the correct system
    config_files = sorted(config_files)
    if system == "nt":
        config_files = [f for f in config_files if f.name.endswith("_win.yaml")]
    elif system == "posix":
        config_files = [f for f in config_files if f.name.endswith("_mac.yaml")]
    else:
        raise ValueError("Unsupported system.")

    last_run_transform = None

    # Run each config file
    for config_file in config_files:
        print(f"Running config file: {config_file.name}")
        transform_name = config_file.name.split("_")[0]
        if transform_name != last_run_transform:
            # Do pregeneration of transform
            pregenerate_transforms(config_file)
            last_run_transform = transform_name

        run_training(parse_config_file(config_file))


if __name__ == "__main__":
    main()

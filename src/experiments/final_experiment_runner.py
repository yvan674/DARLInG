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
import subprocess
import warnings

from data_utils.pregenerate_transform import pregenerate_transforms


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
    failed_runs = []

    # Run each config file
    for config_file in config_files:
        print(f"Running config file: {config_file.name}")
        transform_name = config_file.name.split("_")[0]
        if transform_name != last_run_transform:
            # Do pregeneration of transform
            print(f"Pregenerating transform {transform_name}...")
            pregenerate_transforms(config_file)
            last_run_transform = transform_name

        # noinspection PyBroadException
        try:
            # Run training using subprocess
            environ_vars = os.environ.copy()
            training_fp = Path(__file__).parent / "train_runner.py"
            subprocess.Popen(["python", str(training_fp), str(config_file)],
                             env=environ_vars).wait()
        except:
            # We catch ALL exceptions
            warnings.warn(f"{config_file.name} Failed!")
            failed_runs.append(config_file.name)
    print("=================")
    if len(failed_runs) > 0:
        print("All runs attempted.")
        print("Failed runs:")
        for failed_run in failed_runs:
            print(f"- {failed_run}")
    else:
        print("All runs completed!")


if __name__ == "__main__":
    main()

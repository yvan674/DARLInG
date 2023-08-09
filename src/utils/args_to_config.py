"""Args to Config.

Converts arguments to a YAML config file.
"""
import sys
import yaml
from pathlib import Path

from utils.sweep_runner import prepare_config


def main():
    save_fp = sys.argv[1]
    config = prepare_config(start_argument_idx=2)

    # If there are any path objects, convert them to strings
    for key, value in config.items():
        for k, v in value.items():
            if isinstance(v, Path):
                config[key][k] = str(v)

    # Save config to file as a YAML
    with open(save_fp, "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    main()

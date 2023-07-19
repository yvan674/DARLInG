"""Pregenerate Transform.

Generates transformed data before running training.
"""
import pickle
from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree

from tqdm import tqdm

from data_utils.widar_dataset import WidarDataset
from utils.config_parser import parse_config_file


def parse_args():
    p = ArgumentParser()
    p.add_argument("CONFIG", type=Path,
                   help="Path to the config file.")
    return p.parse_args()


def open_and_generate(config: dict[str, any], split: str,
                      out_dir: Path):
    """Opens a dataset, generates the appropriate transformation, and saves it.

    Args:
        config: Data config dictionary.
        split: Data split to handle.
        out_dir: The directory for where the pregenerated files should be
            generated in.
    """
    # Create the directory if it doesn't exist
    (out_dir / split).mkdir(exist_ok=True, parents=True)

    # Remove the pytorch to tensor step
    amp_pipeline = config["amp_pipeline"]
    amp_pipeline.processors = amp_pipeline.processors[:-1]
    phase_pipeline = config["phase_pipeline"]
    phase_pipeline.processors = phase_pipeline.processors[:-1]

    dataset = WidarDataset(
        root_path=config["data_dir"],
        split_name=split,
        dataset_type=config["dataset_type"],
        downsample_multiplier=config["downsample_multiplier"],
        return_bvp=False,
        return_csi=True,
        amp_pipeline=amp_pipeline,
        phase_pipeline=phase_pipeline
    )

    for x_amp, x_phase, _, info in dataset:
        data = (x_amp, x_phase)

        # Since we get an aggregated all receiver view of the data, we
        # can save it without the receiver identifier. We also add
        # file type .pkl since this is a pickle file.
        out_name = info["csi_fps"][0].name.split("-r")[0] + ".pkl"
        out_fp = out_dir / split / out_name

        with open(out_fp, "wb") as f:
            pickle.dump(data, f)

        print("hello")


def pregenerate_transforms(config_file: Path):
    """Pregenerates transforms.

    Args:
        config_file: Path to the config file to use. This is used to figure out
            what dataset type to use, downsample multiplier, transformation,
            pipelines, etc.
    """
    config = parse_config_file(config_file)["data"]

    pregenerated_dir = config["data_dir"] / "pregenerated"

    if pregenerated_dir.exists():
        # Delete all file recursively inside of this directory
        rmtree(pregenerated_dir)
    else:
        pregenerated_dir.mkdir()

    for split in ["train", "validation", "test"]:
        open_and_generate(config, split, pregenerated_dir)


if __name__ == '__main__':
    args = parse_args()
    pregenerate_transforms(args.CONFIG)

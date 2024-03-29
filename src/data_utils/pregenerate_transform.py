"""Pregenerate Transform.

Generates transformed data before running training.
"""
from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree

import numpy as np
from tqdm import tqdm

from data_utils.widar_dataset import WidarDataset
from signal_processing.pipeline import Pipeline
from utils.config_parser import parse_config_file


def parse_args():
    p = ArgumentParser()
    p.add_argument("CONFIG", type=Path,
                   help="Path to the config file.")
    return p.parse_args()


def open_and_generate(config: dict[str, any],
                      split: str,
                      out_dir: Path,
                      amp_pipeline: Pipeline,
                      phase_pipeline: Pipeline):
    """Opens a dataset, generates the appropriate transformation, and saves it.

    Args:
        config: Data config dictionary.
        split: Data split to handle.
        out_dir: The directory for where the pregenerated files should be
            generated in.
        amp_pipeline: Pipeline to push through the amp CSI signal.
        phase_pipeline: Pipeline to push through the phase CSI signal.
    """
    # Create the directory if it doesn't exist
    (out_dir / split).mkdir(exist_ok=True, parents=True)

    dataset = WidarDataset(root_path=config["data_dir"], split_name=split,
                           dataset_type=config["dataset_type"],
                           return_bvp=False, return_csi=True,
                           amp_pipeline=amp_pipeline,
                           phase_pipeline=phase_pipeline,
                           pregenerated=False)

    for x_amp, x_phase, _, info in tqdm(dataset):
        # Since we get an aggregated all receiver view of the data, we
        # can save it without the receiver identifier. We also add
        # file type .npz since this is a compressed npz file.
        out_name = info["csi_fps"][0].name.split("-r")[0] + ".npz"
        out_fp = out_dir / split / out_name

        # Note that we tried also to save only the upper triangle since
        # the matrices are symmetric. This did not result in disk footprint
        # reduction, probably due to the compression algorithm.

        with open(out_fp, "wb") as f:
            np.savez_compressed(f, x_amp=x_amp, x_phase=x_phase)


def pregenerate_transforms(config_file: Path):
    """Pregenerates transforms.

    Args:
        config_file: Path to the config file to use. This is used to figure out
            what dataset type to use, downsample multiplier, transformation,
            pipelines, etc.
    """
    config = parse_config_file(config_file)["data"]

    pregenerated_dir = (config["data_dir"] /
                        f"pregenerated_{config['dataset_type']}")

    data_dir = config["data_dir"] / ("widar_" + config["dataset_type"])

    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir {data_dir} does not exist.")

    if pregenerated_dir.exists():
        # Delete all file recursively inside of this directory
        rmtree(pregenerated_dir)
    else:
        pregenerated_dir.mkdir()

    # Remove the pytorch to tensor step
    amp_pipe = config["amp_pipeline"]
    amp_pipe.processors = amp_pipe.processors[:-1]
    phase_pipe = config["phase_pipeline"]
    phase_pipe.processors = phase_pipe.processors[:-1]

    for split in ["train", "validation", "test"]:
        if config["dataset_type"] == "full":
            raise NotImplementedError("Full not yet implemented.")
        elif (data_dir / split).exists():
            open_and_generate(config, split, pregenerated_dir,
                              amp_pipe, phase_pipe)


if __name__ == '__main__':
    args = parse_args()
    pregenerate_transforms(args.CONFIG)

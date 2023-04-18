"""Train Runner.

Provides all necessary objects and parameters to make training possible.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import yaml

from data_utils.widar_dataset import WidarDataset
from signal_processing.pipeline import Pipeline
from signal_processing.lowpass_filter import LowPassFilter
from signal_processing.phase_unwrap import PhaseUnwrap
from signal_processing.phase_filter import PhaseFilter
from signal_processing.standard_scaler import StandardScaler
from signal_to_image.deepinsight_transform import DeepInsight
from signal_to_image.gaf_transform import GAF
from signal_to_image.mtf_transform import MTF
from signal_to_image.recurrence_plot_transform import RP
from ui.tqdm_ui import TqdmUI


def parse_args():
    p = ArgumentParser(description="Runner for DARLInG Training")
    p.add_argument("CONFIG_FP", type=Path,
                   help="Path to the config yaml file.")
    return p.parse_args()


def parse_config_file(config_fp: Path) -> dict[str, any]:
    """Parses the yaml config file."""
    with open(config_fp, "r") as f:
        return yaml.safe_load(f)


def run_training(batch_size: int = 64,
                 lr: float = 0.001,
                 epochs: int = 15,
                 alpha: float = 0.5,
                 dropout: float = 0.3,
                 optimizer: str = "Adam",
                 enc_ac_fn: nn.Module,
                 mt_ac_fn: nn.Module,
                 ui: str = "tqdm",
                 checkpoint_dir: Path = Path("checkpoints/"),
                 data_dir: Path = Path("data/"),
                 small_dataset: bool = True,
                 transformation: str = None,
                 is_debug: bool = False,
                 config: dict[str, any] = None):
    """Runs the training.

    Args:
        batch_size: The batch size to use for training.
        lr: The learning rate to use for training.
        epochs: The number of epochs to train for.
        alpha: The alpha value to use for the Triple loss.
        dropout: The dropout rate to use for the VAE.
        optimizer: The optimizer to use for training.
        enc_ac_fn: The activation function to use for the encoder.
        mt_ac_fn: The activation function to use for the MT heads.
        ui: The UI to use for training. Can be "tqdm" or "gui".
        checkpoint_dir: The dir to save the checkpoints to.
        data_dir: The dir to the data to use for training.
        small_dataset: Whether to use the small version of the dataset.
        transformation: The signal-to-image transformation to apply during
            this training run. Options are ["deepinsight", "gaf", "mtf", "rp"].
        is_debug: Whether to consider this run a debugging run.
        config: The yaml configuration as a dict.
    """
    # SECTION Initial stuff
    # Set tags
    tags = [transformation, "training"]
    if is_debug:
        tags.append("debug")

    # Init wandb
    wandb.init(project="master-thesis", entity="yvan674",
               config=config, tags=tags)
    # Validate checkpoints
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Set UI
    match ui:
        case "tqdm":
            ui = TqdmUI
        case "gui":
            raise NotImplementedError("GUI has not been implemented yet.")
        case _:
            raise ValueError(f"{ui} is not one of the available options."
                             f"Available options are [`tqdm`, `gui`]")

    # SECTION Data
    # Set the signal to image transformation to use.
    match transformation:
        case "deepinsight":
            transform = DeepInsight()
        case "gaf":
            transform = GAF()
        case "mtf":
            transform = MTF()
        case "rp":
            transform = RP()
        case _:
            raise ValueError(f"Chosen transformation {transformation}"
                             f"is not one of the valid options. Valid options "
                             f"are [`deepinsight`, `gaf`, `mtf`, `rp`]")

    # Set up the pipeline
    amp_pipeline = Pipeline([LowPassFilter(250, 1000),
                             StandardScaler(data_dir, "amp"),
                             transform,
                             torch.from_numpy])
    phase_pipeline = Pipeline([PhaseUnwrap(),
                               PhaseFilter([3, 3, 1], [3, 3, 1]),
                               LowPassFilter(250, 1000),
                               StandardScaler(data_dir, "phase"),
                               transform,
                               torch.from_numpy])

    train_dataset = WidarDataset(data_dir,
                                 "train",
                                 small_dataset,
                                 amp_pipeline=amp_pipeline,
                                 phase_pipeline=phase_pipeline)
    valid_dataset = WidarDataset(data_dir,
                                 "validation",
                                 small_dataset,
                                 amp_pipeline=amp_pipeline,
                                 phase_pipeline=phase_pipeline)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size,
                                  num_workers=(torch.get_num_threads() - 2) / 2)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size,
                                  num_workers=(torch.get_num_threads() - 2) / 2)

    activation_fn_map = {
        "relu": nn.ReLU,
        "leaky": nn.LeakyReLU,
        "selu": nn.SELU
    }



if __name__ == '__main__':
    args = parse_args()
    conf = parse_config_file(args.CONFIG_FP)
    run_training(**conf, config=conf)

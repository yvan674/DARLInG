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
from torch.optim import Adam, SGD
import wandb
import yaml

from data_utils.widar_dataset import WidarDataset
from data_utils.dataloader_collate import widar_collate_fn
from experiments.train import Training
from models.encoder import Encoder
from models.multi_task import MultiTaskHead
from models.null_agent import NullAgent
from models.known_domain_agent import KnownDomainAgent
from loss.triple_loss import TripleLoss
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
                 latent_dim: int = 10,
                 optimizer: str = "adam",
                 enc_ac_fn: str = "relu",
                 mt_ac_fn: str = "relu",
                 embed_agent_value: str = "known",
                 embed_agent_size: int = None,
                 bvp_pipeline: bool = False,
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
        latent_dim: Size of the latent embedding produced by the encoder.
        optimizer: The optimizer to use for training. Options are [`adam`,
            `sgd`]
        enc_ac_fn: The activation function to use for the encoder as a str.
            Options are [`relu`, `leaky`, `selu`].
        mt_ac_fn: The activation function to use for the MT heads as a str.
            Options are [`relu`, `leaky`, `selu`].
        embed_agent_value: The type of embedding agent to used. Options are
            [`known`, `one-hot`, `probability`].
        embed_agent_size: The size of the embedding provided by the embedding
            agent. If embed_agent_value is `known`, this value is ignored.
        bvp_pipeline: Whether the signal-processing pipeline should be replaced
            with simply providing the pre-calculated BVP.
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
    if embed_agent_size is None and embed_agent_value is not "known":
        raise ValueError("A value must be provided for parameter "
                         f"embed_agent_size if "
                         f"embed_agent_value={embed_agent_value}.")
    # Set tags
    tags = [transformation, "training"]
    if is_debug:
        tags.append("debug")

    # Init wandb
    run = wandb.init(project="master-thesis", entity="yvan674",
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
    if bvp_pipeline:
        amp_pipeline = Pipeline([])
        phase_pipeline = Pipeline([])
    else:
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
                                 phase_pipeline=phase_pipeline,
                                 return_csi=not bvp_pipeline)
    valid_dataset = WidarDataset(data_dir,
                                 "validation",
                                 small_dataset,
                                 amp_pipeline=amp_pipeline,
                                 phase_pipeline=phase_pipeline,
                                 return_csi=not bvp_pipeline)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size,
                                  num_workers=(torch.get_num_threads() - 2) / 2,
                                  collate_fn=widar_collate_fn)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size,
                                  num_workers=(torch.get_num_threads() - 2) / 2,
                                  collate_fn=widar_collate_fn)

    # SECTION Set up models
    # Activation functions
    activation_fn_map = {"relu": nn.ReLU,
                         "leaky": nn.LeakyReLU,
                         "selu": nn.SELU}
    enc_ac_fn = activation_fn_map[enc_ac_fn]
    mt_ac_fn = activation_fn_map[mt_ac_fn]

    # VAE based model
    domain_embedding_size = 33 if embed_agent_size is None else embed_agent_size

    encoder = Encoder(enc_ac_fn, dropout, latent_dim, bvp_pipeline)

    null_head = MultiTaskHead(mt_ac_fn, dropout, latent_dim, mt_ac_fn, dropout,
                              domain_embedding_size)
    embed_head = MultiTaskHead(mt_ac_fn, dropout, latent_dim, mt_ac_fn, dropout,
                               domain_embedding_size)

    # Embedding agents
    null_value = 1. if embed_agent_value in ("known", "one-hot") else None
    null_agent = NullAgent(domain_embedding_size, null_value)
    if embed_agent_value == "known":
        embed_agent = KnownDomainAgent()
    else:
        raise NotImplementedError("Actual RL agent is not yet implemented.")

    # Loss and optimizers
    loss_fn = TripleLoss(alpha)
    optimizer_map = {"adam": Adam, "sgd": SGD}
    encoder_optimizer = optimizer_map[optimizer](encoder.parameters(), lr=lr)
    null_optimizer = optimizer_map[optimizer](null_head.parameters(), lr=lr)
    embed_optimizer = optimizer_map[optimizer](embed_head.parameters(), lr=lr)

    # SECTION UI
    initial_data = {"train_loss": float("nan"),
                    "train_kl_loss": float("nan"),
                    "train_null_loss": float("nan"),
                    "train_embed_loss": float("nan"),
                    "valid_loss": float("nan"),
                    "loss_diff": float("nan"),
                    "epoch": "0",
                    "batch": "0",
                    "rate": float("nan")}
    match ui:
        case "tqdm":
            ui = TqdmUI(len(train_dataset), len(valid_dataset), epochs,
                        initial_data)
        case "gui":
            raise NotImplementedError("GUI has not been implemented yet.")
        case _:
            raise ValueError(f"{ui} is not one of the available options."
                             f"Available options are [`tqdm`, `gui`]")

    ui.update_status("Preparation complete. Starting training...")

    # SECTION Run training
    training = Training(
        bvp_pipeline,                                         # BVP Pipeline
        encoder, null_head, embed_head,                       # Models
        embed_agent, null_agent,                              # Embed agents
        encoder_optimizer, null_optimizer, embed_optimizer,   # Optimizers
        loss_fn,                                              # Loss function
        run, checkpoint_dir, ui                               # Utils
    )

    training.train(train_embedding_agent=embed_agent_value != "known",
                   train_loader=train_dataloader,
                   valid_loader=valid_dataloader,
                   epochs=epochs,
                   device=device)


if __name__ == '__main__':
    args = parse_args()
    conf = parse_config_file(args.CONFIG_FP)
    run_training(**conf, config=conf)

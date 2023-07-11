"""Train Runner.

Provides all necessary objects and parameters to make training possible.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from argparse import ArgumentParser
from pathlib import Path
import os
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import wandb


from data_utils.widar_dataset import WidarDataset
from data_utils.dataloader_collate import widar_collate_fn
from utils.config_parser import parse_config_file
from experiments.train import Training
from models.model_builder import build_model
from models.null_agent import NullAgent
from models.known_domain_agent import KnownDomainAgent
from models.ppo_agent import PPOAgent
from loss.multi_joint_loss import MultiJointLoss
from signal_processing.pipeline import Pipeline
from signal_processing.standard_scaler import StandardScaler
from ui.tqdm_ui import TqdmUI


def parse_args():
    p = ArgumentParser(description="Runner for DARLInG Training")
    p.add_argument("CONFIG_FP", type=Path,
                   help="Path to the config yaml file.")
    return p.parse_args()


CONV_NUM_FEATURES_MAPS = {
    "bvp_stack": 4608,     # 28x20x20 dimensional BVP
    "bvp_1d": 102400,      # 1x28x400 dimensional BVP
    "bvp_sum": 4608,       # 1x20x20 dimensional BVP
    "sti_transform": 0     # 540x1024x1024 dimensional signal-to-image transform
}





def run_training(config: dict[str, dict[str, any]]):
    """Runs the training.

    """
    # SECTION Initial stuff
    # Set tags
    if config["data"]["transformation"] is None:
        if config["train"]["bvp_pipeline"]:
            transformation = "bvp_pipeline"
        else:
            transformation = "no_transform"
    else:
        transformation = config["data"]["transformation"]
    tags = [transformation, "training"]
    if config["debug"]["is_debug"]:
        warnings.warn("Running a debug run, `debug` will be appended to tags.")
        tags.append("debug")

    # Init wandb
    if config["debug"]["offline"]:
        os.environ["WANDB_MODE"] = "dryrun"
        warnings.warn("Running WandB in offline mode.")
    run = wandb.init(project="master-thesis", entity="yvan674",
                     config=config, tags=tags)
    # Validate checkpoints dir
    if not config["train"]["checkpoint_dir"].exists():
        config["train"]["checkpoint_dir"].mkdir(parents=True)

    # Set device
    if config["debug"]["on_cpu"]:
        # Debugging on CPU is easier
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # SECTION Data
    data_dir = config["data"]["data_dir"]
    bvp_pipeline = config["train"]["bvp_pipeline"]
    bvp_agg = config["data"]["bvp_agg"]
    dataset_type = config["data"]["dataset_type"]
    # Set up the pipeline
    if bvp_pipeline:
        warnings.warn("Running with bvp_pipeline=True; no transformation will "
                      "be applied.")
        amp_pipeline = Pipeline([])
        phase_pipeline = Pipeline([])
    else:
        # Set the signal to image transformation to use.


        amp_pipeline = Pipeline.from_str_list(
            config["data"]["amp_pipeline"],
            config["transformation"],
            StandardScaler(config["data"]["data_dir"], "amp"),
            config["data"]["downsample_multiplier"]
        )
        phase_pipeline = Pipeline.from_str_list(
            config["data"]["phase_pipeline"],
            config["transformation"],
            StandardScaler(config["data"]["data_dir"], "phase"),
            config["data"]["downsample_multiplier"]
        )

    train_dataset = WidarDataset(
        data_dir,
        "train",
        dataset_type,
        downsample_multiplier=config["data"]["downsample_multiplier"],
        amp_pipeline=amp_pipeline,
        phase_pipeline=phase_pipeline,
        return_csi=not bvp_pipeline,
        return_bvp=True,
        bvp_agg=bvp_agg
    )
    valid_dataset = WidarDataset(
        data_dir,
        "validation",
        dataset_type,
        downsample_multiplier=config["data"]["downsample_multiplier"],
        amp_pipeline=amp_pipeline,
        phase_pipeline=phase_pipeline,
        return_csi=not bvp_pipeline,
        return_bvp=True,
        bvp_agg=bvp_agg
    )

    # 1 worker ensure no multithreading so we can debug easily
    # num_workers = 1 if is_debug else (torch.get_num_threads() - 2) // 2
    # Trying to get a process lock for the dataloader takes way too long if
    # num_workers > 0 (~3x longer) so we set num_workers to always be 0
    num_workers = 0
    train_dataloader = DataLoader(train_dataset,
                                  config["train"]["batch_size"],
                                  num_workers=num_workers,
                                  collate_fn=widar_collate_fn)
    valid_dataloader = DataLoader(valid_dataset,
                                  config["train"]["batch_size"],
                                  num_workers=num_workers,
                                  collate_fn=widar_collate_fn)

    # SECTION Set up models
    encoder, null_head, embed_head, null_agent, embed_agent = build_model(
        config,
        train_dataset
    )

    # Move models to device
    encoder.to(device)
    null_head.to(device)
    embed_head.to(device)
    null_agent.to(device)
    embed_agent.to(device)

    # Loss and optimizers
    optimizer = config["optim_loss"]["optimizer"]
    lr = config["optim_loss"]["lr"]
    loss_fn = MultiJointLoss(config["optim_loss"]["alpha"],
                             config["optim_loss"]["beta"])
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
    train_steps = len(train_dataset)
    if config["embed"]["value_type"] != "known":
        # Then we're training the embedding agent as well, so we go through
        # the train dataset twice
        train_steps *= 2
    match config["train"]["ui"]:
        case "tqdm":
            ui = TqdmUI(train_steps, len(valid_dataset),
                        config["train"]["epochs"], initial_data)
        case "gui":
            raise NotImplementedError("GUI has not been implemented yet.")
        case _:
            raise ValueError(f"{config['train']['ui']} is not one of the "
                             f"available options."
                             f"Available options are [`tqdm`, `gui`]")

    ui.update_status("Preparation complete. Starting training...")

    # SECTION Run training
    checkpoint_dir = config["train"]["checkpoint_dir"]
    training = Training(
        bvp_pipeline,                                         # BVP Pipeline
        encoder, null_head, embed_head,                       # Models
        embed_agent, null_agent,                              # Embed agents
        encoder_optimizer, null_optimizer, embed_optimizer,   # Optimizers
        loss_fn,                                              # Loss function
        run, checkpoint_dir, ui                               # Utils
    )
    training.train(
        train_embedding_agent=config["embed"]["value_type"] != "known",
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        epochs=config["train"]["epochs"],
        device=device,
        agent_epochs=config["embed"]["epochs"]
    )


if __name__ == '__main__':
    args = parse_args()
    print("Running from config file:")
    print(f"{args.CONFIG_FP}")
    run_training(parse_config_file(args.CONFIG_FP))

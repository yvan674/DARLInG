from pathlib import Path
from typing import Optional

import yaml


def train_config(batch_size: int = 64,
                 epochs: int = 15,
                 ui: str = "tqdm",
                 checkpoint_dir: Path = Path("../../checkpoints/"),
                 bvp_pipeline: bool = False) -> dict[str, any]:
    """Training configuration.

    A function is used to ensure defaults and clear documentation of possible
    options.

    Args:
        batch_size: The batch size to use for training.
        epochs: Number of epochs to train for.
        ui: UI type to use during training.
        checkpoint_dir: Where to save checkpoints to.
        bvp_pipeline: Whether to use the BVP pipeline. When BVP pipeline is
            used, then CSI is ignored as the input data.
    """
    return {"batch_size": batch_size,
            "epochs": epochs,
            "ui": ui,
            "checkpoint_dir": checkpoint_dir,
            "bvp_pipeline": bvp_pipeline}


def data_config(data_dir: Path = Path("../../data/"),
                small_dataset: bool = True,
                downsample_multiplier: int = 2,
                transformation: str = None,
                bvp_agg: str = "stack",
                amp_pipeline: Optional[list[str]] = None,
                phase_pipeline: Optional[list[str]] = None) -> dict[str, any]:
    """Data configuration for training.

    A function is used to ensure defaults and clear documentation of possible
    options.

    Args:
        data_dir: Path to the data directory.
        small_dataset: Whether to use the small dataset or not.
        downsample_multiplier: How much to downsample the data by.
        transformation: Transformation to use on the data. If None, no
            transformation is used.
        bvp_agg: How to aggregate the BVP data. Options are
            [`stack`, `1d`, `sum`].
        amp_pipeline: Pipeline to use for the amplitude data. This is provided
            as a list of strings, where each string is a function to call on
            the data. The possible functions are:
                [`lowpass_filter`, `phase_derivative`, `phase_filter`,
                `phase_unwrap`, `standard_scalar`, `torch.from_numpy`].
            The default is [`torch.from_numpy`].
        phase_pipeline: Pipeline to use for the phase data. This is provided
            as a list of strings, where each string is a function to call on
            the data. The possible functions are:
                [`lowpass_filter`, `phase_derivative`, `phase_filter`,
                `phase_unwrap`, `standard_scalar`, `torch.from_numpy`].
            The default is [`torch.from_numpy`].
    """
    if amp_pipeline is None:
        amp_pipeline = ["torch.from_numpy"]
    if phase_pipeline is None:
        phase_pipeline = ["torch.from_numpy"]

    return {"data_dir": data_dir,
            "small_dataset": small_dataset,
            "downsample_multiplier": downsample_multiplier,
            "transformation": transformation,
            "bvp_agg": bvp_agg,
            "amp_pipeline": amp_pipeline,
            "phase_pipeline": phase_pipeline}



def encoder_config(dropout: float = 0.3,
                   latent_dim: int = 10,
                   activation_fn: str = "relu") -> dict[str, any]:
    """Encoder configuration for training.

    Args:
        dropout: Dropout rate to use.
        latent_dim: Dimension of the latent space.
        activation_fn: Activation function to use. Options are
            [`relu`, `leaky`, `selu`].
    """
    return {"dropout": dropout,
            "latent_dim": latent_dim,
            "activation_fn": activation_fn}


def mt_config(decoder_dropout: float = 0.3,
              decoder_activation_fn: str = "relu",
              predictor_dropout: float = 0.3,
              predictor_activation_fn: str = "relu") -> dict[str, any]:
    """Multitask configuration for training.

    Args:
        decoder_dropout: Dropout rate to use for the decoder.
        decoder_activation_fn: Activation function to use for the decoder.
            Options are [`relu`, `leaky`, `selu`].
        predictor_dropout: Dropout rate to use for the predictor.
        predictor_activation_fn: Activation function to use for the predictor.
            Options are [`relu`, `leaky`, `selu`].
    """
    return {"decoder_dropout": decoder_dropout,
            "decoder_activation_fn": decoder_activation_fn,
            "predictor_dropout": predictor_dropout,
            "predictor_activation_fn": predictor_activation_fn}


def embed_config(embed_agent_value: str = "known",
                 embed_agent_size: Optional[int] = None) -> dict[str, any]:
    """Embedding configuration for training.

    Args:
        embed_agent_value: How to embed the agent. Options are
            [`known`, `one-hot`, `probability-measure`].
        embed_agent_size: Size of the embedding. None is only allowed if the
            embed_agent_value is `known`.
    """
    return {"embed_agent_value": embed_agent_value,
            "embed_agent_size": embed_agent_size}


def optim_loss_config(optimizer: str = "adam",
                      lr: float = 0.001,
                      alpha: float = 0.5,
                      beta: float = 0.5) -> dict[str, any]:
    """Optimizer and loss configuration for training.

    Args:
        optimizer: Optimizer to use. Options are [`adam`, `sgd`].
        lr: Learning rate to use.
        alpha: Weight for the reconstruction vs classification loss.
        beta: Weight for the KL divergence vs mt_head loss.
    """
    return {"optimizer": optimizer,
            "lr": lr,
            "alpha": alpha,
            "beta": beta}


def debug_config(is_debug: bool = False,
                 on_cpu: bool = False):
    """Debug configuration for training.

    Args:
        is_debug: Whether to run in debug mode.
        on_cpu: Whether to force running on the CPU.
    """
    return {"is_debug": is_debug,
            "on_cpu": on_cpu}


def parse_config_file(config_fp: Path) -> dict[str, dict[str, any]]:
    """Parses the yaml config file."""
    with open(config_fp, "r") as f:
        yaml_dict = yaml.safe_load(f)

    config_dict = {}

    if "train" in yaml_dict:
        config_dict["train"] = train_config(**yaml_dict["train"])
    else:
        config_dict["training"] = train_config()
    if "data" in yaml_dict:
        config_dict["data"] = data_config(**yaml_dict["data"])
    else:
        config_dict["data"] = data_config()
    if "encoder" in yaml_dict:
        config_dict["encoder"] = encoder_config(**yaml_dict["encoder"])
    else:
        config_dict["encoder"] = encoder_config()
    if "mt" in yaml_dict:
        config_dict["mt"] = mt_config(**yaml_dict["mt"])
    else:
        config_dict["mt"] = mt_config()
    if "embed" in yaml_dict:
        config_dict["embed"] = embed_config(**yaml_dict["embed"])
    else:
        config_dict["embed"] = embed_config()
    if "optim_loss" in yaml_dict:
        config_dict["optim_loss"] = optim_loss_config(**yaml_dict["optim_loss"])
    else:
        config_dict["optim_loss"] = optim_loss_config()
    if "debug" in yaml_dict:
        config_dict["debug"] = debug_config(**yaml_dict["debug"])
    else:
        config_dict["debug"] = debug_config()

    # Verify configuration
    if config_dict["embed"]["embed_agent_size"] is None and \
            config_dict["embed"]["embed_agent_value"] != "known":
        raise ValueError("A value must be provided for parameter "
                         f"embed_agent_size if embed_agent_value="
                         f"{config_dict['embed']['embed_agent_value']}.")

    if (config_dict["bvp"]["bvp_agg"] is not None) \
            and (config_dict["bvp"]["bvp_agg"] not in ("stack", "1d", "sum")):
        raise ValueError(f"Parameter bvp_agg is "
                         f"{config_dict['bvp']['bvp_agg']} but must be one of"
                         f"[`stack`, `1d`, `sum`].")

    return config_dict

import warnings
from pathlib import Path
from typing import Optional

import yaml

# from signal_to_image.deepinsight_transform import DeepInsight
from signal_to_image.gaf_transform import GAF
from signal_to_image.mtf_transform import MTF
from signal_to_image.recurrence_plot_transform import RP
from signal_processing.pipeline import Pipeline
from signal_processing.standard_scaler import StandardScaler


def train_config(batch_size: int = 64,
                 epochs: int = 15,
                 ui: str = "tqdm",
                 checkpoint_dir: str | Path = Path("../../checkpoints/")
                 ) -> dict[str, any]:
    """Training configuration.

    A function is used to ensure defaults and clear documentation of possible
    options.

    Args:
        batch_size: The batch size to use for training.
        epochs: Number of epochs to train for.
        ui: UI type to use during training.
        checkpoint_dir: Where to save checkpoints to.
    """
    if type(checkpoint_dir) is str:
        checkpoint_dir = Path(checkpoint_dir)

    return {"batch_size": batch_size,
            "epochs": epochs,
            "ui": ui,
            "checkpoint_dir": checkpoint_dir}


def data_config(data_dir: str | Path = Path("../../data/"),
                dataset_type: str = None,
                downsample_multiplier: int = 2,
                transformation: str = None,
                bvp_agg: str = "stack",
                bvp_pipeline: bool = False,
                amp_pipeline: Optional[list[str]] = None,
                phase_pipeline: Optional[list[str]] = None) -> dict[str, any]:
    """Data configuration for training.

    A function is used to ensure defaults and clear documentation of possible
    options.

    Args:
        data_dir: Path to the data directory.
        dataset_type: Type of the dataset. Options are [`small`,
                `single_domain`, `single_user`, `full`, `single_domain_small`,
                `single_user_small`].
        downsample_multiplier: How much to downsample the data by.
        transformation: Transformation to use on the data. If None, no
            transformation is used.
        bvp_agg: How to aggregate the BVP data. Options are
            [`stack`, `1d`, `sum`].
        bvp_pipeline: Whether to use the BVP pipeline. When BVP pipeline is
            used, then CSI is ignored as the input data.
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
    if type(data_dir) is str:
        data_dir = Path(data_dir)

    if dataset_type is None:
        raise ValueError("`dataset_type` parameter must be filled.")

    # Set the signal to image transformation to use.
    match transformation:
        case "deepinsight":
            raise NotImplementedError
            # transform = DeepInsight()
        case "gaf":
            transform = GAF()
        case "mtf":
            transform = MTF()
        case "rp":
            transform = RP()
        case None:
            transform = None
        case _:
            raise ValueError(
                f"Chosen transformation {transformation} is not one of the "
                f"valid options. Valid options are "
                f"[`deepinsight`, `gaf`, `mtf`, `rp`]"
            )
    if bvp_pipeline:
        warnings.warn("Running with bvp_pipeline=True; no transformation will "
                      "be applied.")
        amp_pipeline = Pipeline([])
        phase_pipeline = Pipeline([])
    else:
        if amp_pipeline is None:
            amp_pipeline = ["torch.from_numpy"]
        else:
            amp_pipeline = Pipeline.from_str_list(
                amp_pipeline,
                transform,
                StandardScaler(data_dir, "amp"),
                downsample_multiplier
            )
        if phase_pipeline is None:
            phase_pipeline = ["torch.from_numpy"]
        else:
            phase_pipeline = Pipeline.from_str_list(
                phase_pipeline,
                transform,
                StandardScaler(data_dir, "phase"),
                downsample_multiplier
            )

    return {"data_dir": data_dir,
            "dataset_type": dataset_type,
            "downsample_multiplier": downsample_multiplier,
            "transformation": transform,
            "bvp_agg": bvp_agg,
            "amp_pipeline": amp_pipeline,
            "phase_pipeline": phase_pipeline,
            "bvp_pipeline": bvp_pipeline}



def encoder_config(dropout: float = 0.3,
                   latent_dim: int = 10,
                   activation_fn: str = "relu",
                   num_conv_layers: int = None,
                   initial_kernel_size: int = 3) -> dict[str, any]:
    """Encoder configuration for training.

    Args:
        dropout: Dropout rate to use.
        latent_dim: Dimension of the latent space.
        activation_fn: Activation function to use. Options are
            [`relu`, `leaky`, `selu`].
        num_conv_layers: Number of convolutional layers to use.
        initial_kernel_size: Initial kernel size to use for the convolutional
            layers.
    """
    return {"dropout": dropout,
            "latent_dim": latent_dim,
            "activation_fn": activation_fn,
            "num_conv_layers": num_conv_layers,
            "initial_kernel_size": initial_kernel_size}


def mt_config(decoder_dropout: float = 0.3,
              decoder_activation_fn: str = "relu",
              predictor_dropout: float = 0.3,
              predictor_activation_fn: str = "relu",
              predictor_num_layers: int = 3) -> dict[str, any]:
    """Multitask configuration for training.

    Args:
        decoder_dropout: Dropout rate to use for the decoder.
        decoder_activation_fn: Activation function to use for the decoder.
            Options are [`relu`, `leaky`, `selu`].
        predictor_dropout: Dropout rate to use for the predictor.
        predictor_activation_fn: Activation function to use for the predictor.
            Options are [`relu`, `leaky`, `selu`].
        predictor_num_layers: Number of layers to use for the GesturePredictor.
    """
    return {"decoder_dropout": decoder_dropout,
            "decoder_activation_fn": decoder_activation_fn,
            "predictor_dropout": predictor_dropout,
            "predictor_activation_fn": predictor_activation_fn,
            "predictor_num_layers": predictor_num_layers}


def embed_config(value_type: str = "known",
                 embed_size: Optional[int] = None,
                 start_epoch: int = 0,
                 epochs: int = 1,
                 critic_num_layers: int = 3,
                 critic_dropout: float = 0.3,
                 actor_num_layers: int = 3,
                 actor_dropout: float = 0.3,
                 lr: float = 1e-4,
                 anneal_lr: bool = True,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 norm_advantage: bool = True,
                 clip_coef: float = 0.2,
                 clip_value_loss: bool = True,
                 entropy_coef: float = 0.0,
                 value_func_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: float = None) -> dict[str, any]:
    """Embedding agent configuration for training.

    Args:
        value_type: How to embed the agent. Options are
            [`known`, `one-hot`, `probability-measure`].
        embed_size: Size of the embedding. None is only allowed if the
            embed_agent_value is `known` and is automatically replaced by 33.
        epochs: Number of epochs to train the embedding agent for.
        start_epoch: Start training the agent after how many epochs. This is
            because the output of the decoder may not make sense before a few
            epochs of training and it does not make sense to train the agent
            on nonsensical inputs.
        critic_num_layers: Number of layers to use for the critic.
        critic_dropout: Dropout rate to use for the critic.
        actor_num_layers: Number of layers to use for the actor.
        actor_dropout: Dropout rate to use for the actor.
        lr: Learning rate for the agent optimizer.
        anneal_lr: Whether to use learning rate annealing.
        gamma: Discount factor gamma in the PPO algorithm.
        gae_lambda: General advantage estimation lambda value.
        norm_advantage: Whether to normalize the advantage value.
        clip_coef: Surrogate clipping coefficient.
        clip_value_loss: Whether to use a clipped value function. PPO paper
            uses a clipped value function.
        entropy_coef: Coefficient for entropy.
        value_func_coef: Coefficient for the value function.
        max_grad_norm: Maximum norm for gradient clipping
        target_kl: Target KL divergence threshold.
    """
    if embed_size is None:
        embed_size = 33
    return {"value_type": value_type,
            "embed_size": embed_size,
            "start_epoch": start_epoch,
            "epochs": epochs,
            "critic_num_layers": critic_num_layers,
            "critic_dropout": critic_dropout,
            "actor_num_layers": actor_num_layers,
            "actor_dropout": actor_dropout,
            "lr": lr,
            "anneal_lr": anneal_lr,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "norm_advantage": norm_advantage,
            "clip_coef": clip_coef,
            "clip_value_loss": clip_value_loss,
            "entropy_coef": entropy_coef,
            "value_func_coef": value_func_coef,
            "max_grad_norm": max_grad_norm,
            "target_kl": target_kl}


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
                 on_cpu: bool = False,
                 offline: bool = False):
    """Debug configuration for training.

    Args:
        is_debug: Whether to run in debug mode.
        on_cpu: Whether to force running on the CPU.
        offline: Whether to run wandb in offline mode.
    """
    return {"is_debug": is_debug,
            "on_cpu": on_cpu,
            "offline": offline}


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
    if config_dict["embed"]["embed_size"] is None and \
            config_dict["embed"]["value_type"] != "known":
        raise ValueError("A value must be provided for parameter "
                         f"embed_size if value_type="
                         f"{config_dict['embed']['value_type']}.")

    if (config_dict["data"]["bvp_agg"] is not None) \
            and (config_dict["data"]["bvp_agg"] not in ("stack", "1d", "sum")):
        raise ValueError(f"Parameter bvp_agg is "
                         f"{config_dict['data']['bvp_agg']} but must be one of"
                         f"[`stack`, `1d`, `sum`].")

    return config_dict

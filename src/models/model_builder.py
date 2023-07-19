"""Model Builder.

The models are now much more variable so requires
"""
from time import perf_counter

import torch
import torch.nn as nn

from data_utils.widar_dataset import WidarDataset
from models.encoder import Encoder, BVPEncoder, AmpPhaseEncoder
from models.multi_task import MultiTaskHead
from models.known_domain_agent import KnownDomainAgent
from models.null_agent import NullAgent
from models.ppo_agent import PPOAgent


ACTIVATION_FN_MAP = {"relu": nn.ReLU,
                     "leaky": nn.LeakyReLU,
                     "selu": nn.SELU}


def calc_encoder_fc_size(input_img: torch.Tensor,
                         initial_kernel_size: int,
                         conv_output_sizes: list[int],
                         encoder_input_dim) -> int:
    """Given an input image, calculates the size of the FC layer of the encoder.

    Args:
        input_img: The BVP or Amp/Phase transformed image desired.
        initial_kernel_size: Initial size of the kernel of the encoder
            convolutional layers.
        conv_output_sizes: Size of each output of the convolutional layers.
        encoder_input_dim: Calculated dimension of the input_img.

    Returns:
        The size of the input of the fully connected layer.
    """
    temp_encoder = Encoder(input_dim=encoder_input_dim,
                           initial_kernel_size=initial_kernel_size,
                           conv_output_sizes=conv_output_sizes)
    temp_encoder.eval()
    with torch.no_grad():
        h = temp_encoder.convnet(input_img.unsqueeze(0))
        h = h.flatten(1)
        fc_input_size = h.shape[1]

    del temp_encoder

    return fc_input_size


def build_model(config: dict[str, any],
                train_dataset: WidarDataset
                ) -> tuple:
    """Creates the DARLInG model.

    VAE based model.

    Returns:
        The encoder, null head, embed head, null agent, and embed agent.
    """
    print("Building models...")
    start_time = perf_counter()
    # Activation functions
    enc_ac_fn = ACTIVATION_FN_MAP[config["encoder"]["activation_fn"]]
    mt_dec_ac_fn = ACTIVATION_FN_MAP[config["mt"]["decoder_activation_fn"]]
    mt_pred_ac_fn = ACTIVATION_FN_MAP[config["mt"]["predictor_activation_fn"]]

    # There are 33 possible domain factors if the domain factors that are in the
    # ground-truth data is encoded in one-hot. If embed_size is None, we assume
    # we want to use the ground-truth domain factors
    if config["embed"]["embed_size"] is not None:
        domain_embedding_size = config["embed"]["embed_size"]
    else:
        domain_embedding_size = 33

    # Figure out BVP aggregration structure
    bvp_pipeline = config["data"]["bvp_pipeline"]

    x_amp, x_phase, x_bvp, x_info = train_dataset[0]

    # SECTION Encoder setup
    if bvp_pipeline:
        encoder_input_dim = x_bvp.shape[0]
        input_img = x_bvp
    else:
        encoder_input_dim = x_amp.shape[0]
        input_img = x_amp
    num_conv_layers = config["encoder"]["num_conv_layers"]
    if num_conv_layers is None:
        conv_output_sizes = None
    else:
        conv_output_sizes = [2 ** (i + 6) for i in range(num_conv_layers)]
    encoder_fc_input_size = calc_encoder_fc_size(
        input_img,
        config["encoder"]["initial_kernel_size"],
        conv_output_sizes,
        encoder_input_dim
    )

    if bvp_pipeline:
        encoder = BVPEncoder(
            enc_ac_fn,
            config["encoder"]["dropout"],
            config["encoder"]["latent_dim"],
            fc_input_size=encoder_fc_input_size,
            input_dim=encoder_input_dim,
            initial_kernel_size=config["encoder"]["initial_kernel_size"],
            conv_output_sizes=conv_output_sizes
        )
        mt_input_head_dim = config["encoder"]["latent_dim"]
    else:
        encoder = AmpPhaseEncoder(
            enc_ac_fn,
            config["encoder"]["dropout"],
            config["encoder"]["latent_dim"],
            fc_input_size=encoder_fc_input_size,
            input_dim=encoder_input_dim,
            initial_kernel_size=config["encoder"]["initial_kernel_size"],
            conv_output_sizes=conv_output_sizes
        )
        mt_input_head_dim = 2 * config["encoder"]["latent_dim"]

    # SECTION Multitask heads
    null_head = MultiTaskHead(
        decoder_ac_func=mt_dec_ac_fn,
        decoder_dropout=config["mt"]["decoder_dropout"],
        decoder_output_layers=encoder_input_dim,
        decoder_output_size=input_img.shape[1:],
        encoder_latent_dim=mt_input_head_dim,
        predictor_num_layers=config["mt"]["predictor_num_layers"],
        predictor_ac_func=mt_pred_ac_fn,
        predictor_dropout=config["mt"]["predictor_dropout"],
        domain_label_size=domain_embedding_size
    )

    embed_head = MultiTaskHead(
        decoder_ac_func=mt_dec_ac_fn,
        decoder_dropout=config["mt"]["decoder_dropout"],
        decoder_output_layers=encoder_input_dim,
        decoder_output_size=input_img.shape[1:],
        encoder_latent_dim=mt_input_head_dim,
        predictor_num_layers=config["mt"]["predictor_num_layers"],
        predictor_ac_func=mt_pred_ac_fn,
        predictor_dropout=config["mt"]["predictor_dropout"],
        domain_label_size=domain_embedding_size
    )

    # SECTION Embed Agents
    if config["embed"]["value_type"] in ("known", "one-hot"):
        null_value = 0.
    else:
        null_value = None

    null_agent = NullAgent(domain_embedding_size, null_value)
    if config["embed"]["value_type"] == "known":
        embed_agent = KnownDomainAgent(domain_embedding_size)
    else:
        embed_agent = PPOAgent(
            input_size=config["encoder"]["latent_dim"],
            domain_embedding_size=domain_embedding_size,
            critic_num_layers=config["embed"]["critic_num_layers"],
            critic_dropout=config["embed"]["critic_dropout"],
            actor_num_layers=config["embed"]["actor_num_layers"],
            actor_dropout=config["embed"]["actor_dropout"],
            lr=config["embed"]["lr"],
            anneal_lr=config["embed"]["anneal_lr"],
            gamma=config["embed"]["gamma"],
            gae_lambda=config["embed"]["gae_lambda"],
            norm_advantage=config["embed"]["norm_advantage"],
            clip_coef=config["embed"]["clip_coef"],
            clip_value_loss=config["embed"]["clip_value_loss"],
            entropy_coef=config["embed"]["entropy_coef"],
            value_func_coef=config["embed"]["value_func_coef"],
            max_grad_norm=config["embed"]["max_grad_norm"],
            target_kl=config["embed"]["target_kl"],
        )

    print(f"Completed model building. "
          f"Took {perf_counter() - start_time:.2f} s.")

    return encoder, null_head, embed_head, null_agent, embed_agent

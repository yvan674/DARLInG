"""Latent Space Exploration."""
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data_utils.widar_dataset import WidarDataset
from models.model_builder import build_model
from utils.config_parser import parse_config_file


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("checkpoint_fp", type=Path,
                   help="Path to the checkpoint")
    p.add_argument("config_fp", type=Path,
                   help="Path to the config file")
    p.add_argument("data_dir_fp", type=Path,
                   help="Path to the data directory")

    return p.parse_args()


def forward_pass(encoder, embed_agent, amp, phase, bvp, bvp_pipeline, device):
    if not bvp_pipeline:
        amp = amp.to(device)
        amp = amp.unsqueeze(0)
        phase = phase.to(device)
        phase = phase.unsqueeze(0)

    bvp = bvp.to(device)
    bvp = bvp.unsqueeze(0)

    with torch.no_grad():
        z = encoder(amp, phase, bvp)[0]
        action = embed_agent(z)[0]

    return z, action


def main(checkpoint_fp: Path, config_fp: Path, data_dir_fp: Path):
    config = parse_config_file(config_fp)
    device = torch.device("cpu")

    # SECTION Data
    bvp_pipeline = config["data"]["bvp_pipeline"]
    train_dataset = WidarDataset(
        data_dir_fp,
        "train",
        config["data"]["dataset_type"],
        downsample_multiplier=config["data"]["downsample_multiplier"],
        amp_pipeline=config["data"]["amp_pipeline"],
        phase_pipeline=config["data"]["phase_pipeline"],
        return_csi=not bvp_pipeline,
        return_bvp=True,
        bvp_agg=config["data"]["bvp_agg"]
    )
    valid_dataset = WidarDataset(
        data_dir_fp,
        "validation",
        config["data"]["dataset_type"],
        downsample_multiplier=config["data"]["downsample_multiplier"],
        amp_pipeline=config["data"]["amp_pipeline"],
        phase_pipeline=config["data"]["phase_pipeline"],
        return_csi=not bvp_pipeline,
        return_bvp=True,
        bvp_agg=config["data"]["bvp_agg"]
    )

    # SECTION Model
    encoder, _, _, embed_agent = build_model(config, train_dataset)

    encoder.to(device)
    embed_agent.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_fp, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])

    # Temporary Workaround
    embed_agent_state_dict = checkpoint["embed_agent_state_dict"]
    if "critic_num_layers" not in embed_agent_state_dict:
        embed_agent_state_dict.update({"critic_num_layers": 4,
                                       "critic_dropout": 0.562851983704799,
                                       "actor_num_layers": 6,
                                       "actor_dropout": 0.47659867398346056})
    embed_agent.load_state_dict(checkpoint["embed_agent_state_dict"])

    encoder.eval()
    embed_agent.eval()

    # SECTION Latent space exploration
    latent_space = []
    embed_space = []
    infos = []
    for i, (amp, phase, bvp, info) in tqdm(enumerate(train_dataset),
                                           total=len(train_dataset)):
        z, action = forward_pass(encoder, embed_agent, amp, phase, bvp,
                                 bvp_pipeline, device)
        latent_space.append(z)
        embed_space.append(action)
        info["is_train"] = True
        infos.append(info)

    for i, (amp, phase, bvp, info) in tqdm(enumerate(valid_dataset),
                                           total=len(valid_dataset)):
        z, action = forward_pass(encoder, embed_agent, amp, phase, bvp,
                                 bvp_pipeline, device)
        latent_space.append(z)
        embed_space.append(action)
        info["is_train"] = False
        infos.append(info)

    latent_space = torch.cat(latent_space, dim=0)
    embed_space = torch.cat(embed_space, dim=0)

    # Get all unique users from infos
    users = set([info["user"] for info in infos])


    # SECTION Visualize spaces
    for space, space_name in zip([latent_space, embed_space],
                                 ["latent_space", "embed_space"]):
        tsne = TSNE(n_components=2, random_state=0)
        space = tsne.fit_transform(space)

        # Color based on train/valid, showing legend
        fig, ax = plt.subplots()
        scatter = ax.scatter(space[:, 0], space[:, 1], c=[info["is_train"]
                                                          for info in infos])
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="lower left", title="Is Train")
        ax.add_artist(legend1)
        plt.title(f"{space_name} (train/valid)")
        plt.savefig(f"{space_name}_train_valid.png")

        # Color based on gesture
        fig, ax = plt.subplots()
        scatter = ax.scatter(space[:, 0], space[:, 1],
                             c=[info["gesture"] for info in infos])
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="lower left", title="Gesture")
        ax.add_artist(legend1)
        plt.title(f"{space_name} (gesture)")
        plt.savefig(f"{space_name}_gesture.png")

        # Color based on user
        fig, ax = plt.subplots()
        scatter = ax.scatter(space[:, 0], space[:, 1],
                             c=[info["user"] for info in infos])
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="lower left", title="User")
        ax.add_artist(legend1)
        plt.title(f"{space_name} (user)")
        plt.savefig(f"{space_name}_user.png")


if __name__ == '__main__':
    args = parse_args()
    main(args.checkpoint_fp, args.config_fp, args.data_dir_fp)

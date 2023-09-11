"""Latent Space Exploration."""
import copy
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data_utils.widar_dataset import WidarDataset
from models.model_builder import build_model
from rl.rl_builder import build_rl
from rl.reward_functions import func_from_str

from utils.config_parser import parse_config_file


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("checkpoint_fp", type=Path,
                   help="Path to the checkpoint")
    p.add_argument("agent_checkpoint_fp", type=Path,
                   help="Path to the agent checkpoint")
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
        action = embed_agent.predict(z.detach().cpu())[0]

    return z, action


def visualize_space(space, colors, legend_ncols, space_name, plot_title,
                    legend_title=""):
    fig, ax = plt.subplots()
    # set dpi to 300 for high quality
    fig.set_dpi(300)
    scatter = ax.scatter(space[:, 0], space[:, 1],
                         c=colors)

    # Shrink axis so it doesn't overlap
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15,
                     box.width, box.height * 0.85])

    # Set both x and y axis min max to -35, 35
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)

    # Put legend below the axis
    if legend_ncols > 0:
        ax.legend(*scatter.legend_elements(),
                  loc="upper center", bbox_to_anchor=(0.5, -0.05),
                  ncol=legend_ncols, title=legend_title)
    plt.title(f"{space_name} ({plot_title})")
    plt.savefig(f"figures/{space_name}_{plot_title}.png")


def main(checkpoint_fp: Path, agent_checkpoint_fp: Path,
         config_fp: Path, data_dir_fp: Path):
    config = parse_config_file(config_fp)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")

    # SECTION Data
    bvp_pipeline = config["data"]["bvp_pipeline"]
    train_dataset = WidarDataset(data_dir_fp, "train",
                                 config["data"]["dataset_type"],
                                 return_bvp=True,
                                 bvp_agg=config["data"]["bvp_agg"],
                                 return_csi=not bvp_pipeline,
                                 amp_pipeline=config["data"]["amp_pipeline"],
                                 phase_pipeline=config["data"][
                                     "phase_pipeline"])
    valid_dataset = WidarDataset(data_dir_fp, "validation",
                                 config["data"]["dataset_type"],
                                 return_bvp=True,
                                 bvp_agg=config["data"]["bvp_agg"],
                                 return_csi=not bvp_pipeline,
                                 amp_pipeline=config["data"]["amp_pipeline"],
                                 phase_pipeline=config["data"][
                                     "phase_pipeline"])

    train_loader = DataLoader(train_dataset)

    # SECTION Model
    reward_function = func_from_str(config["embed"]["reward_function"])
    encoder, null_head, null_agent = build_model(config, train_dataset)
    embed_head = copy.deepcopy(null_head)

    env, embed_agent, _ = build_rl(encoder, null_head, embed_head, null_agent,
                                   config["embed"]["agent_type"],
                                   bvp_pipeline, device, train_loader,
                                   reward_function, 1)

    encoder.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_fp, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])

    embed_agent.load(agent_checkpoint_fp)

    encoder.eval()

    # SECTION Latent space exploration
    latent_space = []
    embed_space = []
    infos = []
    users = set()

    for dataset, is_train in zip((train_dataset, valid_dataset),
                                 (True, False)):
        for i, (amp, phase, bvp, info) in tqdm(enumerate(dataset),
                                               total=len(dataset)):
            z, action = forward_pass(encoder, embed_agent, amp, phase, bvp,
                                     bvp_pipeline, device)
            latent_space.append(z)
            embed_space.append(action)
            info["is_train"] = is_train
            info["user"] += 1
            infos.append(info)
            users.add(info["user"])

    latent_space = torch.cat(latent_space, dim=0).detach().cpu().numpy()
    embed_space = np.concatenate(embed_space, axis=0)
    users = sorted(list(users))


    # SECTION Visualize spaces
    users_list = [info["user"] for info in infos]
    gesture_list = [info["gesture"] for info in infos]
    for space, space_name in tqdm(zip([latent_space, embed_space],
                                      ["Latent Space t-SNE",
                                       "Domain Embedding t-SNE"]),
                                  total=2):
        tsne = TSNE(n_components=2, random_state=0)
        space = tsne.fit_transform(space)

        # # Color based on train/valid, showing legend
        # visualize_space(space, [info["is_train"] for info in infos],
        #                 2, space_name,
        #                 "Train vs. Validation set",
        #                 "Is Train")

        visualize_space(space, gesture_list,
                        6, space_name, "Gesture",
                        "Gesture")

        visualize_space(space, users_list,
                        6, space_name, "User",
                        "User")

        # SECTION Visualize User Spaces
        # we want here to visualize gesture by user for both the latent space
        # and the domain embedding
        # First prep data to visualize

        users_list = np.array(users_list, dtype=int)
        gesture_list = np.array(gesture_list)
        for user in users:
            user_filter = users_list == user
            user_space = space[user_filter]
            user_gesture = gesture_list[user_filter]

            visualize_space(user_space, user_gesture.tolist(),
                            6,
                            space_name,
                            f"user {user} by gesture",
                            "Gesture")


if __name__ == '__main__':
    args = parse_args()
    main(args.checkpoint_fp, args.agent_checkpoint_fp,
         args.config_fp, args.data_dir_fp)

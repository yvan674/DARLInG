"""RL Builder.

Builds the environment and agent from config.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise

from models.null_agent import NullAgent
from rl.latent_environment import LatentEnvironment
from models.known_domain_agent import KnownDomainAgent


def build_rl(encoder: nn.Module,
             null_head: nn.Module,
             embed_head: nn.Module,
             null_agent: NullAgent,
             embedding_agent: str,
             bvp_pipeline: bool,
             device: torch.device,
             train_loader: DataLoader,
             reward_function: callable,
             agent_epochs: int) -> tuple:
    """Builds both the agent and environment for RL.

    Returns:
        The latent environment, the agent, and the total agent timesteps.
    """
    # Create the environment
    env = LatentEnvironment(
        encoder=encoder,
        null_head=null_head,
        embed_head=embed_head,
        null_agent=null_agent,
        bvp_pipeline=bvp_pipeline,
        device=device,
        dataset=train_loader.dataset,
        reward_function=reward_function
    )
    total_agent_timesteps = agent_epochs

    # Set up the agent
    if embedding_agent == "known":
        embedding_agent = KnownDomainAgent(
            embed_head.domain_label_size
        )
        embedding_agent.to(device)
    elif embedding_agent == "ddpg":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=np.full(n_actions, 0.1)
        )
        embedding_agent = DDPG("MlpPolicy",
                               env,
                               action_noise=action_noise,
                               device=device,
                               verbose=1)
    elif embedding_agent == "ppo":
        embedding_agent = PPO("MlpPolicy",
                              env,
                              device=device,
                              verbose=1)
        total_agent_timesteps *= len(train_loader.dataset)
    return env, embedding_agent, total_agent_timesteps

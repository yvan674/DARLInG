"""Latent Environment.

The gym environment wrapper for the latent space.
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from gymnasium import spaces

from models.null_agent import NullAgent


class LatentEnvironment(gym.Env):
    metadata = {"render_modes": ["none"]}

    def __init__(self,
                 encoder: nn.Module,
                 null_head: nn.Module,
                 embed_head: nn.Module,
                 null_agent: NullAgent,
                 bvp_pipeline: bool,
                 device: torch.device,
                 dataset: Dataset,
                 reward_function: callable):
        self.encoder = encoder
        self.null_head = null_head
        self.embed_head = embed_head
        self.null_agent = null_agent
        self.dataset = dataset
        self.bvp_pipeline = bvp_pipeline
        self.device = device
        self.reward_function = reward_function

        # obs is the cpu version, z is the device version.
        self.last_obs = None
        self.last_z = None
        self.last_info = None

        self.current_step = 0

        if bvp_pipeline:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=[self.encoder.latent_dim]
            )
        else:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=[2, self.encoder.latent_dim]
            )

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=[self.null_head.domain_label_size]
        )

    def _set_obs(self, step_num):
        amp, phase, bvp, info = self.dataset[step_num]

        if self.bvp_pipeline:
            bvp = bvp.to(self.device).unsqueeze(0)
        else:
            amp = amp.to(self.device).unsqueeze(0)
            phase = phase.to(self.device).unsqueeze(0)
            bvp = bvp.to(self.device).unsqueeze(0)
        with torch.no_grad():
            z, _, _ = self.encoder(amp, phase, bvp)

        self.last_z = z.detach()
        self.last_obs = self.last_z.cpu()
        self.last_info = info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0

        self._set_obs(0)
        self.current_step += 1
        return self.last_obs, self.last_info

    def step(self, action):
        if self.current_step == len(self.dataset):
            curr = -1
            terminated = True
        else:
            curr = self.current_step
            terminated = False

        if isinstance(action, np.ndarray):
            action = torch.tensor(action)

        reward = self.reward_function(self.null_head,
                                      self.embed_head,
                                      self.null_agent,
                                      self.last_z,
                                      self.last_obs,
                                      self.last_info,
                                      action,
                                      self.device)
        # Gets the next observation
        self._set_obs(curr)
        self.current_step += 1

        return self.last_obs, reward, terminated, False, self.last_info

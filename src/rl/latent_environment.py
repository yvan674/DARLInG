"""Latent Environment.

The gym environment wrapper for the latent space.
"""
import gymnasium as gym
import torch
from gymnasium import spaces

from data_utils.widar_dataset import WidarDataset
from models.base_embedding_agent import BaseEmbeddingAgent
from models.encoder import Encoder
from models.multi_task import MultiTaskHead


class LatentEnvironment(gym.Env):
    metadata = {"render_modes": ["none"]}

    def __init__(self,
                 encoder: Encoder,
                 null_head: MultiTaskHead,
                 embed_head: MultiTaskHead,
                 null_agent: BaseEmbeddingAgent,
                 bvp_pipeline: bool,
                 device: torch.device,
                 dataset: WidarDataset,
                 reward_function: callable):
        self.encoder = encoder
        self.null_head = null_head
        self.embed_head = embed_head
        self.null_agent = null_agent
        self.dataset = dataset,
        self.bvp_pipeline = bvp_pipeline
        self.device = device
        self.reward_function = reward_function

        self.last_bvp = None
        self.last_obs = None

        self.current_step = 0

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=[self.encoder.latent_dim]
        )

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=[self.null_head.domain_label_size]
        )

    def _get_obs(self, amp, phase, bvp):
        if self.bvp_pipeline:
            bvp = bvp.to(self.device)
        else:
            amp = amp.to(self.device)
            phase = phase.to(self.device)
            bvp = bvp.to(self.device)
        with torch.no_grad():
            obs, _, _ = self.encoder(amp, phase, bvp)

        return obs

    def _get_and_set_obs(self, step_num):
        amp, phase, bvp, info = self.dataset[step_num]
        self.last_bvp = bvp
        obs = self._get_obs(amp, phase, bvp)
        self.last_obs = obs

        return obs, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0

        obs, info = self._get_and_set_obs(0)

        self.step += 1
        return obs, info

    def step(self, action):
        if self.current_step == len(self.dataset):
            curr = -1
            terminated = True
        else:
            curr = self.current_step
            terminated = False

        reward = self.reward_function(self.null_head,
                                      self.embed_head,
                                      self.null_agent,
                                      self.last_obs,
                                      self.last_bvp,
                                      action)
        # Gets the next observation
        obs, info = self._get_and_set_obs(curr)

        return obs, reward, terminated, False, info

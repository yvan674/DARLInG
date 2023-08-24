"""DQN Agent.

Based on the cleanrl work by vwxyzjn.
<github.com/vwxyzjn/cleanrl>

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.buffers import ReplayBuffer

from models.base_embedding_agent import BaseEmbeddingAgent


class DQNAgent(BaseEmbeddingAgent):
    def __init__(self,
                 input_size: int,
                 domain_embedding_size: int,
                 q_network_num_layers: int,
                 q_network_dropout: float,
                 lr: float = 3e-4):
        self.q_network = self._build_network(input_size, domain_embedding_size,
                                             q_network_num_layers,
                                             q_network_dropout)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(),
                                          lr=lr)
        self.replay_buffer = ReplayBuffer


    def _linear_block(self, in_dim, out_dim, dropout=0.3, **kwargs):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

    @staticmethod
    def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
        """Linear scheduler helper function."""
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def produce_action(self, observation: torch.Tensor,
                       info: dict[str, list[any]], **kwargs) -> torch.Tensor:
        pass

    def process_reward(self, observation: torch.Tensor, reward: float):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def state_dict(self):
        pass

    @staticmethod
    def load_state_dict(sd: dict[any]):
        pass

    def to(self, device: int | torch.device | None):
        pass
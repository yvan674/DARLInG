"""Null Agent.

Only produces null embeddings.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch
from torch import nn as nn

from models.base_embedding_agent import BaseEmbeddingAgent


class NullAgent(BaseEmbeddingAgent):
    @staticmethod
    def _linear_block(in_dim, out_dim, **kwargs) -> nn.Module:
        pass

    def __init__(self, domain_embedding_size: int, null_value: float | None):
        super().__init__(domain_embedding_size=domain_embedding_size)
        self.device = None
        if null_value is None:
            self.null_value = 1 / self.domain_embedding_size
        else:
            self.null_value = null_value

    def __repr__(self):
        return f"NullAgent(embedding_length={self.domain_embedding_size}, " \
               f"null_value={self.null_value})"

    def produce_action(self, observation: torch.Tensor,
                       info: dict[str, list[any]], **kwargs) -> torch.Tensor:
        batch_size = len(info["user"])

        return torch.full((batch_size, self.domain_embedding_size),
                          fill_value=self.null_value,
                          device=self.device)

    def process_reward(self, observation: torch.Tensor, reward: float):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def state_dict(self):
        return {"device": self.device,
                "domain_embedding_size": self.domain_embedding_size,
                "null_value": self.null_value}

    @staticmethod
    def load_state_dict(sd: dict[any]):
        agent = NullAgent(sd["domain_embedding_size"],
                          sd["null_value"])
        agent.to(sd["device"])
        return agent

    def to(self, device: int | torch.device | None):
        self.device = device

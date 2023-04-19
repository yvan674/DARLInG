"""Null Agent.

Only produces null embeddings.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch

from models.base_embedding_agent import BaseEmbeddingAgent


class NullAgent(BaseEmbeddingAgent):
    def __init__(self, embedding_length: int, null_value: float | None):
        super().__init__()
        self.device = None
        self.embedding_length = embedding_length
        self.null_value = null_value

    def _produce_action(self, observation: torch.Tensor,
                        info: list[dict[str, any]]) -> torch.Tensor:
        if self.null_value is None:
            fill_value = 1 / len(info)
        else:
            fill_value = self.null_value
        return torch.full((len(info), self.embedding_length),
                          fill_value=fill_value,
                          device=self.device)

    def process_reward(self, observation: torch.Tensor, reward: float):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def state_dict(self):
        return {"device": self.device,
                "embedding_length": self.embedding_length,
                "null_value": self.null_value}

    def load_state_dict(self, sd: dict[any]):
        self.device = sd["device"]
        self.embedding_length = sd["embedding_length"]
        self.null_value = sd["null_value"]

    def to(self, device: int | torch.device | None):
        self.device = device

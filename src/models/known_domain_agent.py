"""Known Domain Agent.

Produces embeddings based on the info provided.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch

from models.base_embedding_agent import BaseEmbeddingAgent


class KnownDomainAgent(BaseEmbeddingAgent):
    def __init__(self):
        super().__init__()
        self.device = None

    def _produce_action(self, observation: torch.Tensor,
                        info: None | dict[any]) -> torch.Tensor:
        pass

    def process_reward(self, observation: torch.Tensor, reward: float):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, sd: dict[any]):
        pass

    def to(self, device: int | torch.device | None):
        pass

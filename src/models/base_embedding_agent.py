"""Base Embedding Agent.

The base class for the domain embedding agent

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from abc import ABC, abstractmethod

import torch


class BaseEmbeddingAgent(ABC):
    def __init__(self, domain_embedding_size: int):
        self.domain_embedding_size = domain_embedding_size

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Make the class callable, like an nn.Module class."""
        return self.produce_action(*args, **kwargs)

    @abstractmethod
    def produce_action(self, observation: torch.Tensor,
                       info: dict[str, list[any]], **kwargs) -> torch.Tensor:
        """Produces an action based on an observation and info dict."""
        raise NotImplementedError

    @abstractmethod
    def process_reward(self, observation: torch.Tensor, reward: float):
        """Processes the result of the reward function."""
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """Puts the agent into training mode."""
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        """Puts the agent into evaluation only/inference mode."""
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        """Gets whatever should be saved to a state_dict for the agent."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_state_dict(sd: dict[any]):
        """Loads a model's state from the provided state dictionary."""
        raise NotImplementedError

    @abstractmethod
    def to(self, device: int | torch.device | None):
        """Moves the model to the appropriate device."""
        raise NotImplementedError

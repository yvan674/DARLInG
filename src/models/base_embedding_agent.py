"""Base Embedding Agent.

The base class for the domain embedding agent

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from abc import ABC, abstractmethod, abstractstaticmethod

import torch
import torch.nn as nn


class BaseEmbeddingAgent(ABC):
    def __init__(self, domain_embedding_size: int):
        self.domain_embedding_size = domain_embedding_size

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Make the class callable, like an nn.Module class."""
        return self.produce_action(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def _linear_block(in_dim, out_dim, **kwargs) -> nn.Module:
        raise NotImplementedError

    def _build_network(self,
                       in_features: int,
                       out_features: int,
                       num_layers: int,
                       layer_dropout: float):
        """Builds a fully connected network based on parameters."""
        output_layers = [2 ** (i + 4) for i in range(num_layers)]
        output_sizes = [(in_features, output_layers[0])] + \
                       [(output_layers[i], output_layers[i + 1])
                        for i in range(len(output_layers) - 1)] + \
                       [(output_layers[-1], out_features)]

        network = nn.Sequential(
            *[self._linear_block(size[0], size[1], dropout=layer_dropout)
              for size in output_sizes]
        )
        return network

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

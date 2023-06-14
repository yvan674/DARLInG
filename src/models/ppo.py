"""Proximal Policy Optimization.

The actual PPO algorithm itself. Based almost entirely on the cleanrl
implementation by Costa Huang <github.com/vwxyjn>
"""
import numpy as np
import torch
import torch.nn as nn


class PPO(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        """Based on the implementation in cleanrl.

        Args:
            input_size: Size of the input based on the environment observation.
            output_size: Size of the output based on the environment action.
        """
        super().__init__()
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0)
        )

        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, output_size), std=0.01),
        )
        self.actor_log_sigma = nn.Parameter(torch.zeros(1, output_size))

    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0) -> nn.Module:
        """We use an explicit layer initialization."""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        """Gets only the critic's response."""
        return self.critic(x)

    def get_action_and_value(self, x, action):
        """Gets both the actor's and critic's response."""
        action_mean = self.actor_mean(x)
        action_log_sigma = self.actor_log_sigma.expand_as(action_mean)
        action_std = torch.exp(action_log_sigma)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return (action,
                probs.log_prob(action).sum(1),
                probs.entropy().sum(1),
                self.get_value(x))

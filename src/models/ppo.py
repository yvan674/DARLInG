"""Proximal Policy Optimization.

The actual PPO algorithm itself. Based almost entirely on the cleanrl
implementation by Costa Huang <github.com/vwxyjn>
"""
import numpy as np
import torch
import torch.nn as nn


class PPO(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 critic_num_layers: int, critic_dropout: float,
                 actor_num_layers: int, actor_dropout: float):
        """Based on the implementation in cleanrl.

        Args:
            input_size: Size of the input based on the environment observation.
            output_size: Size of
            critic_num_layers: Number of layers for the critic.
            critic_dropout: Dropout for the critic.
            actor_num_layers: Number of layers for the actor.
            actor_dropout: Dropout for the actor.
        """
        super().__init__()

        def linear_block(in_dim, out_dim, std=np.sqrt(2), bias_const=0.0,
                         dropout=0.3):
            return nn.Sequential(
                self.layer_init(nn.Linear(in_dim, out_dim),
                                std=std, bias_const=bias_const),
                nn.Dropout(dropout),
                nn.Tanh()
            )

        def build_network(in_features: int,
                          out_features: int,
                          num_layers: int,
                          layer_dropout: float):
            """Builds a fully connected network based on parameters."""
            output_layers = [2 ** (i + 4) for i in range(num_layers)]
            output_sizes = [(in_features, output_layers[0])] + \
                           [(output_layers[i], output_layers[i + 1])
                            for i in range(len(output_layers) - 1)] + \
                           [(output_layers[-1], out_features)]

            network = nn.Sequential(*[linear_block(size[0], size[1],
                                                   dropout=layer_dropout)
                                      for size in output_sizes])
            return network

        self.critic = build_network(input_size, 1,
                                    critic_num_layers, critic_dropout)
        # self.critic = nn.Sequential(
        #     self.layer_init(nn.Linear(input_size, 64)),
        #     nn.Tanh(),
        #     self.layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     self.layer_init(nn.Linear(64, 1), std=1.0)
        # )
        self.actor_mean = build_network(input_size, output_size,
                                        actor_num_layers, actor_dropout)

        # self.actor_mean = nn.Sequential(
        #     self.layer_init(nn.Linear(input_size, 64)),
        #     nn.Tanh(),
        #     self.layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     self.layer_init(nn.Linear(64, output_size), std=0.01),
        # )
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

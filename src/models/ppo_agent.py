"""Proximal Policy Optimization.

Based on the cleanrl work by vwxyzjn.
<github.com/vwxyzjn/cleanrl>

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import numpy as np
import torch
import torch.nn as nn

from models.base_embedding_agent import BaseEmbeddingAgent


class PPOAgent(BaseEmbeddingAgent):
    def __init__(self,
                 input_size: int,
                 domain_embedding_size: int,
                 critic_num_layers: int,
                 critic_dropout: float,
                 actor_num_layers: int,
                 actor_dropout: float,
                 lr: float = 3e-4,
                 anneal_lr: bool = True,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 norm_advantage: bool = True,
                 clip_coef: float = 0.2,
                 clip_value_loss: bool = True,
                 entropy_coef: float = 0.0,
                 value_func_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: float = None):
        """
        Initializes the parameters required to properly train the PPOAgent.

        Args:
            input_size: Size of the environmental observation.
            domain_embedding_size: Size of the action tensor.
            critic_num_layers: Number of layers in the critic network.
            critic_dropout: Dropout for the critic network.
            actor_num_layers: Number of layers in the actor network.
            actor_dropout: Dropout for the actor network.
            lr: Learning rate for the agent optimizer.
            anneal_lr: Whether to use learning rate annealing.
            gamma: Discount factor gamma in the PPO algorithm.
            gae_lambda: General advantage estimation lambda value.
            norm_advantage: Whether to normalize the advantage value.
            clip_coef: Surrogate clipping coefficient.
            clip_value_loss: Whether to use a clipped value function. PPO paper
                uses a clipped value function.
            entropy_coef: Coefficient for entropy.
            value_func_coef: Coefficient for the value function.
            max_grad_norm: Maximum norm for gradient clipping
            target_kl: Target KL divergence threshold.
        """
        super().__init__(domain_embedding_size=domain_embedding_size)
        self.critic_num_layers = critic_num_layers
        self.critic_dropout = critic_dropout
        self.actor_num_layers = actor_num_layers
        self.actor_dropout = actor_dropout
        self.lr = lr
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.norm_advantage = norm_advantage
        self.clip_coef = clip_coef
        self.clip_value_loss = clip_value_loss
        self.entropy_coef = entropy_coef
        self.value_func_coef = value_func_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.input_size = input_size

        self.critic = self._build_network(input_size,
                                          1,
                                          critic_num_layers,
                                          critic_dropout)
        self.actor_mean = self._build_network(input_size,
                                              domain_embedding_size,
                                              actor_num_layers,
                                              actor_dropout)
        self.actor_log_sigma = nn.Parameter(
            torch.zeros(1, domain_embedding_size)
        )

        self.optimizer = torch.optim.Adam(self.ppo.parameters(), lr=lr,
                                          eps=1e-5)

    @staticmethod
    def _layer_init(layer, std=np.sqrt(2), bias_const=0.0) -> nn.Module:
        """We use an explicit layer initialization."""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def _linear_block(self, in_dim, out_dim, std=np.sqrt(2), bias_const=0.0,
                      dropout=0.3):
        return nn.Sequential(
            self._layer_init(nn.Linear(in_dim, out_dim),
                             std=std, bias_const=bias_const),
            nn.Dropout(dropout),
            nn.Tanh()
        )

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

    def set_anneal_lr(self, epoch, total_epochs):
        """Set a new learning rate using annealing."""
        if not self.anneal_lr:
            return
        frac = 1.0 - epoch / total_epochs
        new_lr = frac * self.lr
        self.optimizer.param_groups[0]["lr"] = new_lr

    def produce_action(self, observation: torch.Tensor,
                       info: dict[str, list[any]] = None,
                       action: torch.Tensor = None,
                       **kwargs) -> any:
        """Gets both the actor's and critic's response."""
        action_mean = self.actor_mean(observation)
        action_log_sigma = self.actor_log_sigma.expand_as(action_mean)
        action_std = torch.exp(action_log_sigma)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return (action,
                probs.log_prob(action).sum(1),
                probs.entropy().sum(1),
                self.critic(observation))

    def process_reward(self, observation: torch.Tensor, reward: float):
        pass

    def train(self):
        """Set the model to train mode."""
        self.critic.train()
        self.actor_mean.train()

    def eval(self):
        """Set the model to eval mode."""
        self.critic.eval()
        self.actor_mean.eval()

    def state_dict(self):
        return {
            "critic_num_layers": self.critic_num_layers,
            "critic_dropout": self.critic_dropout,
            "actor_num_layers": self.actor_num_layers,
            "actor_dropout": self.actor_dropout,
            "lr": self.lr,
            "anneal_lr": self.anneal_lr,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "norm_advantage": self.norm_advantage,
            "clip_coef": self.clip_coef,
            "clip_value_loss": self.clip_value_loss,
            "entropy_coef": self.entropy_coef,
            "value_func_coef": self.value_func_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "input_size": self.input_size,
            "ppo": {"critic_state_dict": self.critic.state_dict(),
                    "actor_mean_state_dict": self.actor_mean.state_dict()},
            "domain_embedding_size": self.domain_embedding_size,
            "optimizer": self.optimizer.state_dict()
        }

    @staticmethod
    def load_state_dict(sd: dict[any]):
        agent = PPOAgent(sd["input_size"],
                         sd["domain_embedding_size"],
                         critic_num_layers=sd["critic_num_layers"],
                         critic_dropout=sd["critic_dropout"],
                         actor_num_layers=sd["actor_num_layers"],
                         actor_dropout=sd["actor_dropout"],
                         lr=sd["lr"],
                         anneal_lr=sd["anneal_lr"],
                         gamma=sd["gamma"],
                         gae_lambda=sd["gae_lambda"],
                         norm_advantage=sd["norm_advantage"],
                         clip_coef=sd["clip_coef"],
                         clip_value_loss=sd["clip_value_loss"],
                         entropy_coef=sd["entropy_coef"],
                         value_func_coef=sd["value_func_coef"],
                         max_grad_norm=sd["max_grad_norm"],
                         target_kl=sd["target_kl"])
        agent.critic.load_state_dict(sd["ppo"]["critic_state_dict"])
        agent.actor_mean.load_state_dict(sd["ppo"]["actor_mean_state_dict"])
        agent.optimizer.load_state_dict(sd["optimizer"])
        return agent

    def to(self, device: int | torch.device | None):
        self.critic.to(device)
        self.actor_mean.to(device)

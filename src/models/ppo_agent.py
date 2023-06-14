"""Proximal Policy Optimization.

Based on the cleanrl work by vwxyzjn.
<github.com/vwxyjn/cleanrl>

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch

from models.base_embedding_agent import BaseEmbeddingAgent
from models.ppo import PPO


class PPOAgent(BaseEmbeddingAgent):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 lr: float = 3e-4,
                 num_steps: int = 2048,
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
            output_size: Size of the action tensor.
            lr: Learning rate for the agent optimizer.
            num_steps: Number of steps per policy rollout.
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
        super().__init__()
        self.lr = lr
        self.num_steps = num_steps
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
        self.output_size = output_size

        self.ppo = PPO(input_size, output_size)

    def _produce_action(self, observation: torch.Tensor,
                        info: dict[str, list[any]]) -> torch.Tensor:
        pass

    def process_reward(self, observation: torch.Tensor, reward: float):
        pass

    def train(self):
        """Set the model to train mode."""
        self.ppo.train()

    def eval(self):
        """Set the model to eval mode."""
        self.ppo.eval()

    def state_dict(self):
        return {"lr": self.lr,
                "num_steps": self.num_steps,
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
                "output_size": self.output_size,
                "ppo": self.ppo.state_dict()}

    @staticmethod
    def load_state_dict(sd: dict[any]):
        agent = PPOAgent(sd["input_size"], sd["output_size"],
                         lr=sd["lr"],
                         num_steps=sd["num_steps"],
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
        agent.ppo.load_state_dict(sd["ppo"])
        return agent

    def to(self, device: int | torch.device | None):
        self.ppo.to(device)

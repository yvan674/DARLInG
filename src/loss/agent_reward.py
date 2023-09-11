"""Agent Reward.

Reward function for the agent.
"""
from abc import ABC, abstractmethod

import torch


class BaseReward(ABC):
    @abstractmethod
    def calculate_reward(self, amp, phase, bvp, info,
                         pass_result) -> torch.Tensor:
        raise NotImplementedError


class LossDiffReward(BaseReward):
    def calculate_reward(self, amp, phase, bvp, info,
                         pass_result) -> torch.Tensor:
        return ((pass_result["embed_loss"] - pass_result["null_loss"]) + 1) ** 2

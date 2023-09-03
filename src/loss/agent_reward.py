"""Agent Reward.

Reward function for the agent.
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseReward(ABC):
    @abstractmethod
    def calculate_reward(self, amp, phase, bvp, info,
                         pass_result) -> torch.Tensor:
        raise NotImplementedError


class LossDiffReward(BaseReward):
    def __init__(self):
        """Reward function based on gesture loss difference.

        Calculates the loss as the difference between the null loss and the
        embedding loss. The loss is calculated as the cross entropy loss
        between the predicted gesture and the target gesture with no reduction.
        """
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def calculate_reward(self, amp, phase, bvp, info,
                         pass_result) -> torch.Tensor:
        info["gesture"] = info["gesture"].to(pass_result["gesture_null"].device)
        null_loss = self.ce(pass_result["gesture_null"], info["gesture"])
        embed_loss = self.ce(pass_result["gesture_embed"], info["gesture"])
        return null_loss - embed_loss

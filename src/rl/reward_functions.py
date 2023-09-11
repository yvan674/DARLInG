"""Reward Functions.

A collection of reward functions used by the embedding agent.
"""
import math

import torch
import torch.nn.functional as F

from models.multi_task import MultiTaskHead
from models.multi_task import run_heads
from models.null_agent import NullAgent


def func_from_str(reward_func_name: str) -> callable:
    if reward_func_name == "do_nothing":
        return do_nothing
    elif reward_func_name == "contrastive":
        return contrastive_reward
    elif reward_func_name == "maximize_difference":
        return maximize_difference
    else:
        raise ValueError(f"Reward function given {reward_func_name} is not"
                         f"known.")


def do_nothing(null_head, embed_head, null_agent,
               z, obs, info, action):
    return action.sum()


def contrastive_reward(null_head: MultiTaskHead,
                       embed_head: MultiTaskHead,
                       null_agent: NullAgent,
                       z: torch.Tensor,
                       obs: torch.Tensor,
                       info: dict[str, any],
                       action: torch.Tensor,
                       device: torch.device,
                       magic_number: int = 100):
    """Gets a reward which is the CE difference between the two heads."""
    null_embedding = null_agent(z, info)

    with torch.no_grad():
        _, gesture_null, _, gesture_embed = run_heads(
            null_head, embed_head, null_embedding, action, z
        )

    target = torch.tensor(info["gesture"], device=device).unsqueeze(0)
    null_ce = F.cross_entropy(gesture_null, target)
    embed_ce = F.cross_entropy(gesture_embed, target)
    return (null_ce - embed_ce) * magic_number


def maximize_difference(null_head: MultiTaskHead,
                        embed_head: MultiTaskHead,
                        null_agent: NullAgent,
                        z: torch.Tensor,
                        obs: torch.Tensor,
                        info: dict[str, any],
                        action: torch.Tensor,
                        device: torch.device,
                        magic_number: int = 100,
                        alpha: float = 0.1):
    """Maximizes differences between each dimension of the action.

    The idea here is to provide stronger signals for the multi-task heads.
    We do this by first soft-maxing the action. We then calculate the
    difference between each combination of dimensions.
    The difference is then pushed through something similar to a ridge loss,
    where any value below alpha=0.1 is zeroed out.
    Finally, we sum together all differences and divide by the number of
    non-zero values.

    The contrastive reward is also calculated and both these values are summed
    together to produce the final reward.

    We expect action to have a flat shape.
    """

    # Prepare action
    action_softmax = F.softmax(action, dim=1).flatten()

    # Calculate the difference between each combination of dimensions of the
    # action
    diff = torch.zeros(math.comb(action_softmax.shape[0], 2))
    pos = 0  # do this instead of math since lookup is faster than calculation
    for i in range(action_softmax.shape[0]):
        for j in range(i + 1, action_softmax.shape[0]):
            diff[pos] = action_softmax[i] - action_softmax[j]
            pos += 1
    diff = diff.abs()
    # Push through ridge loss
    to_keep = diff > alpha
    diff *= to_keep

    diff = diff.sum() / to_keep.sum() * magic_number

    if diff.isnan():
        diff = torch.tensor(0, dtype=torch.float32, device=device)

    # Add the contrastive reward
    contrastive = contrastive_reward(null_head, embed_head, null_agent,
                                     z, obs, info, action, device,
                                     magic_number)

    return diff + contrastive


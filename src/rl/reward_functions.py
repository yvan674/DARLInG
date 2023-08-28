"""Reward Functions.

A collection of reward functions used by the embedding agent.
"""
import torch
import torch.nn.functional as F

from models.multi_task import run_heads


def func_from_str(reward_func_name: str) -> callable:
    return do_nothing


def do_nothing(null_head, embed_head, null_agent,
               z, obs, info, action):
    return action.sum()


def contrastive_reward(null_head, embed_head, null_agent,
                       z, obs, info, action):
    """Gets a reward which is the CE difference between the two heads."""
    null_embedding = null_agent(z, info)
    with torch.no_grad():
        _, gesture_null, _, gesture_embed = run_heads(
            null_head, embed_head, null_embedding, action, z
        )

    null_ce = F.cross_entropy(gesture_null, info["gesture"])
    embed_ce = F.cross_entropy(gesture_embed, info["gesture"])
    return null_ce - embed_ce

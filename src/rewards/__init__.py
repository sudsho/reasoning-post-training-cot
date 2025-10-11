"""Reward functions used by GRPO refinement.

The registry lets configs reference rewards by string name.
"""

from src.rewards.correctness import answer_present_reward, correctness_reward
from src.rewards.format import format_reward
from src.rewards.length_penalty import length_penalty, make_length_penalty

REWARD_REGISTRY = {
    "correctness": correctness_reward,
    "answer_present": answer_present_reward,
    "format": format_reward,
    "length_penalty": length_penalty,
}


def get_reward(name: str, **kwargs):
    if name == "length_penalty" and kwargs:
        return make_length_penalty(**kwargs)
    if name not in REWARD_REGISTRY:
        raise KeyError(f"unknown reward {name!r}, have {sorted(REWARD_REGISTRY)}")
    return REWARD_REGISTRY[name]


__all__ = [
    "REWARD_REGISTRY",
    "get_reward",
    "correctness_reward",
    "answer_present_reward",
    "format_reward",
    "length_penalty",
    "make_length_penalty",
]

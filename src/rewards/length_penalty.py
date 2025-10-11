"""Discourage over-reasoning by penalising very long traces.

Design:
* zero penalty (i.e. reward=0) up to a soft cap `soft_max` tokens
* linear decay from 0 down to `min_reward` between `soft_max` and `hard_max`
* clipped at `min_reward` above `hard_max`

We approximate token count by 4 chars/token (fast; TRL calls this thousands
of times per step).
"""

from __future__ import annotations

from typing import Iterable


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def make_length_penalty(
    soft_max: int = 512,
    hard_max: int = 1024,
    min_reward: float = -0.5,
):
    span = max(1, hard_max - soft_max)

    def reward(prompts: Iterable[str], completions: Iterable, **kwargs) -> list[float]:
        out: list[float] = []
        for comp in completions:
            text = comp if isinstance(comp, str) else comp[0].get("content", "")
            n = _approx_tokens(text)
            if n <= soft_max:
                out.append(0.0)
            elif n >= hard_max:
                out.append(min_reward)
            else:
                frac = (n - soft_max) / span
                out.append(min_reward * frac)
        return out

    return reward


length_penalty = make_length_penalty()

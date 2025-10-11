"""Structural / format rewards.

Encourage the model to keep the <think>...</think> then Answer: X shape
even after GRPO. Small weight; correctness_reward still dominates.
"""

from __future__ import annotations

import re
from typing import Iterable

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_RE = re.compile(r"[Aa]nswer\s*[:\-]\s*\S")


def format_reward(prompts: Iterable[str], completions: Iterable, **kwargs) -> list[float]:
    out: list[float] = []
    for comp in completions:
        text = comp if isinstance(comp, str) else comp[0].get("content", "")
        r = 0.0
        has_open = THINK_OPEN in text
        has_close = THINK_CLOSE in text
        # think block symmetry
        if has_open and has_close:
            r += 0.15
        elif has_close and not has_open:
            # closing without opening is worse than nothing
            r -= 0.05
        # answer line
        if ANSWER_RE.search(text):
            r += 0.1
        # penalise mid-trace "Answer:" leakage before </think>
        if has_close:
            head, _, tail = text.partition(THINK_CLOSE)
            if ANSWER_RE.search(head):
                r -= 0.1
            if not ANSWER_RE.search(tail):
                r -= 0.05
        out.append(r)
    return out

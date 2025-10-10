"""Rule-based correctness rewards for GRPO training.

Each reward function has signature `(prompts, completions, **kwargs) -> list[float]`
so it plugs directly into TRL GRPOTrainer's `reward_funcs`.

`kwargs` carries the per-example gold answer + task tag that we attached to the
prompt dataset (`gold` and `task` columns).
"""

from __future__ import annotations

import re
from typing import Iterable

from src.distill.filter_correct import extract_answer, is_correct


def _reward_one(completion: str, gold: str, task: str) -> float:
    pred = extract_answer(completion)
    return 1.0 if is_correct(pred, gold, task) else 0.0


def correctness_reward(prompts: Iterable[str], completions: Iterable, **kwargs) -> list[float]:
    golds = kwargs.get("gold") or kwargs.get("answer")
    tasks = kwargs.get("task")
    if golds is None or tasks is None:
        raise ValueError("correctness_reward requires 'gold' and 'task' kwargs from dataset")

    out: list[float] = []
    for comp, gold, task in zip(completions, golds, tasks):
        text = comp if isinstance(comp, str) else comp[0].get("content", "")
        out.append(_reward_one(text, gold, task))
    return out


def _final_answer_present(text: str) -> bool:
    return bool(re.search(r"[Aa]nswer\s*[:\-]\s*\S", text))


def answer_present_reward(prompts, completions, **kwargs) -> list[float]:
    """Small shaping reward for producing an 'Answer:' line at all."""
    out: list[float] = []
    for comp in completions:
        text = comp if isinstance(comp, str) else comp[0].get("content", "")
        out.append(0.1 if _final_answer_present(text) else 0.0)
    return out

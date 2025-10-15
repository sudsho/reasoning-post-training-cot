"""Self-consistency sampling + majority (or weighted) vote.

Given a prompt, draw K samples from the model, extract each sample's
final answer, and vote. For multiple-choice tasks (ARC) we do plain
majority over normalized letters. For math tasks we normalize each
answer, group by canonical form, and either take the mode
(`majority`) or weight by aggregate reward proxy (`weighted`).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from src.distill.filter_correct import _norm_letter, _norm_math, extract_answer


@dataclass
class VoteResult:
    answer: str | None
    support: int
    total: int
    distribution: dict[str, int]

    @property
    def confidence(self) -> float:
        return self.support / self.total if self.total else 0.0


def _norm_for_task(pred: str | None, task: str) -> str | None:
    if pred is None:
        return None
    if task == "arc":
        return _norm_letter(pred)
    if task in {"gsm8k", "math"}:
        # try to canonicalize numeric form; fall back to raw normalized string
        s = _norm_math(pred)
        try:
            f = float(s)
            # collapse -0.0 vs 0.0, integer-vs-float
            return f"{f:.10g}"
        except ValueError:
            return s
    return pred.strip().lower()


def majority_vote(traces: Iterable[str], task: str) -> VoteResult:
    dist: Counter[str] = Counter()
    total = 0
    for t in traces:
        total += 1
        norm = _norm_for_task(extract_answer(t), task)
        if norm is None:
            continue
        dist[norm] += 1
    if not dist:
        return VoteResult(None, 0, total, {})
    winner, support = dist.most_common(1)[0]
    return VoteResult(winner, support, total, dict(dist))


def weighted_vote(
    traces: Iterable[str],
    task: str,
    weight_fn: Callable[[str], float] | None = None,
) -> VoteResult:
    """Weighted vote. Default weight = 1.0 (falls back to majority)."""
    wf = weight_fn or (lambda _t: 1.0)
    scores: defaultdict[str, float] = defaultdict(float)
    dist: Counter[str] = Counter()
    total = 0
    for t in traces:
        total += 1
        norm = _norm_for_task(extract_answer(t), task)
        if norm is None:
            continue
        w = float(wf(t))
        scores[norm] += w
        dist[norm] += 1
    if not scores:
        return VoteResult(None, 0, total, {})
    winner = max(scores.items(), key=lambda kv: kv[1])[0]
    return VoteResult(winner, dist[winner], total, dict(dist))


def sample_and_vote(
    prompt: str,
    generate_fn: Callable[[str, int], list[str]],
    k: int,
    task: str,
    strategy: str = "majority",
    weight_fn: Callable[[str], float] | None = None,
) -> VoteResult:
    """Generate K samples from `generate_fn(prompt, k)` and vote."""
    traces = generate_fn(prompt, k)
    if strategy == "weighted":
        return weighted_vote(traces, task, weight_fn=weight_fn)
    return majority_vote(traces, task)

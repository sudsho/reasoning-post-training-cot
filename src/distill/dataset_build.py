"""Build the final SFT jsonl from filtered per-task shards.

Combines shards, deduplicates by (question, first N chars of trace),
optionally caps per-task counts, formats each row into a single
prompt/response pair the SFT loop expects.

Output row schema:
    {
      "prompt": "<user question wrapped in chat template>",
      "response": "<think>...</think>\\nAnswer: X",
      "task": "gsm8k" | "math" | "arc",
      "meta": {...}
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable


SYS_PROMPT = (
    "You are a careful reasoner. Solve the problem step by step inside "
    "<think>...</think> tags, then output the final answer on a new line "
    "as 'Answer: <value>'."
)


def _hash(question: str, trace: str) -> str:
    h = hashlib.sha1()
    h.update(question.strip().lower().encode("utf-8"))
    h.update(b"\x00")
    h.update(trace.strip()[:200].encode("utf-8"))
    return h.hexdigest()


def build_prompt(question: str) -> str:
    # Qwen2.5 chat template friendly: system + user, no assistant yet
    return (
        f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_response(trace: str, answer: str) -> str:
    # ensure trace has closing </think> and answer line
    t = trace.strip()
    if "<think>" not in t:
        t = "<think>\n" + t
    if "</think>" not in t:
        t = t + "\n</think>"
    # if the trace already carries an "Answer:" line, keep it; else append
    if "\nAnswer:" not in t and "Answer:" not in t.split("</think>", 1)[-1]:
        t = t + f"\nAnswer: {answer}"
    return t + "<|im_end|>"


def load_shards(paths: list[Path]) -> Iterable[dict]:
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("dataset_build")
    ap.add_argument("--in", dest="inp", nargs="+", required=True, help="filtered jsonl shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-out", default=None)
    ap.add_argument("--per-task-cap", type=int, default=None)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=17)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    seen: set[str] = set()
    per_task: Counter[str] = Counter()
    rows: list[dict] = []
    for rec in load_shards([Path(p) for p in args.inp]):
        task = rec.get("meta", {}).get("task", "generic")
        if args.per_task_cap and per_task[task] >= args.per_task_cap:
            continue
        h = _hash(rec["question"], rec["trace"])
        if h in seen:
            continue
        seen.add(h)
        per_task[task] += 1
        rows.append(
            {
                "prompt": build_prompt(rec["question"]),
                "response": build_response(rec["trace"], rec["answer"]),
                "task": task,
                "meta": rec.get("meta", {}),
            }
        )

    rng.shuffle(rows)
    n_eval = int(len(rows) * args.eval_frac) if args.eval_out else 0
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.eval_out:
        eo = Path(args.eval_out)
        eo.parent.mkdir(parents=True, exist_ok=True)
        with eo.open("w", encoding="utf-8") as f:
            for r in eval_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"train={len(train_rows)}  eval={len(eval_rows)}  by task={dict(per_task)}")


if __name__ == "__main__":
    main()

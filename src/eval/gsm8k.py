"""GSM8K test-set eval with self-consistency."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.distill.dataset_build import build_prompt
from src.distill.filter_correct import _norm_math
from src.eval.arc_challenge import build_generate_fn
from src.inference.self_consistency import sample_and_vote


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def gold_from_gsm8k(answer_field: str) -> str:
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip()
    m = NUM_RE.findall(answer_field)
    return m[-1] if m else answer_field.strip()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("eval_gsm8k")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strategy", default="weighted", choices=["majority", "weighted"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    gen = build_generate_fn(args.endpoint, args.model, args.temperature, args.top_p, args.max_tokens)

    n_correct = 0
    rows_out = []
    for row in tqdm(ds, desc="gsm8k"):
        prompt = build_prompt(row["question"])
        res = sample_and_vote(prompt, gen, args.k, task="gsm8k", strategy=args.strategy)
        gold = gold_from_gsm8k(row["answer"])
        ok = False
        try:
            if res.answer is not None:
                ok = float(_norm_math(res.answer)) == float(_norm_math(gold))
        except ValueError:
            ok = (res.answer or "").strip() == gold.strip()
        n_correct += int(ok)
        rows_out.append(
            {
                "gold": gold,
                "pred": res.answer,
                "support": res.support,
                "k": res.total,
                "confidence": res.confidence,
                "correct": ok,
            }
        )

    acc = n_correct / max(len(rows_out), 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"acc": acc, "n": len(rows_out), "rows": rows_out}, indent=2))
    print(f"GSM8K acc = {acc:.4f}  ({n_correct}/{len(rows_out)}) k={args.k}")


if __name__ == "__main__":
    main()

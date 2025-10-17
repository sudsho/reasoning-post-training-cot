"""MATH-500 (HuggingFaceH4/MATH-500) evaluation with self-consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.distill.dataset_build import build_prompt
from src.distill.filter_correct import _math_verify_equal, _norm_math
from src.eval.arc_challenge import build_generate_fn
from src.inference.self_consistency import sample_and_vote


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("eval_math")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strategy", default="weighted", choices=["majority", "weighted"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    gen = build_generate_fn(args.endpoint, args.model, args.temperature, args.top_p, args.max_tokens)

    n_correct = 0
    rows_out = []
    for row in tqdm(ds, desc="math-500"):
        prompt = build_prompt(row["problem"])
        res = sample_and_vote(prompt, gen, args.k, task="math", strategy=args.strategy)
        gold = str(row["answer"]).strip()
        ok = False
        if res.answer is not None:
            if _norm_math(res.answer) == _norm_math(gold):
                ok = True
            else:
                ok = _math_verify_equal(res.answer, gold)
        n_correct += int(ok)
        rows_out.append(
            {
                "level": row.get("level"),
                "subject": row.get("subject"),
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
    print(f"MATH-500 acc = {acc:.4f}  ({n_correct}/{len(rows_out)}) k={args.k}")


if __name__ == "__main__":
    main()

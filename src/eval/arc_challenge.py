"""ARC-Challenge evaluation.

Loads allenai/ai2_arc (ARC-Challenge test split), formats each question as
a 4- or 5-choice multiple choice prompt, generates K samples from the
target endpoint, majority-votes on the letter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from datasets import load_dataset
from tqdm import tqdm

from src.distill.dataset_build import SYS_PROMPT, build_prompt
from src.inference.self_consistency import sample_and_vote


def format_arc_question(row: dict) -> str:
    choices = row["choices"]
    labels = choices["label"]
    texts = choices["text"]
    body = row["question"] + "\n\n"
    for lab, txt in zip(labels, texts):
        body += f"{lab}) {txt}\n"
    body += "\nRespond with the letter of the correct choice."
    return body


def build_generate_fn(endpoint: str, model: str, temperature: float, top_p: float, max_tokens: int) -> Callable[[str, int], list[str]]:
    from openai import OpenAI

    client = OpenAI(base_url=endpoint, api_key="not-needed")

    def gen(prompt: str, k: int) -> list[str]:
        resp = client.completions.create(
            model=model,
            prompt=prompt,
            n=k,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["\n\nProblem:", "<|im_end|>"],
        )
        return [c.text for c in resp.choices]

    return gen


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("eval_arc")
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strategy", default="majority", choices=["majority", "weighted"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    gen = build_generate_fn(args.endpoint, args.model, args.temperature, args.top_p, args.max_tokens)

    n_correct = 0
    rows_out = []
    for row in tqdm(ds, desc="arc-c"):
        prompt = build_prompt(format_arc_question(row))
        res = sample_and_vote(prompt, gen, args.k, task="arc", strategy=args.strategy)
        gold = row["answerKey"].strip().upper()
        ok = (res.answer or "").strip().upper() == gold
        n_correct += int(ok)
        rows_out.append(
            {
                "id": row.get("id"),
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
    print(f"ARC-Challenge acc = {acc:.4f}  ({n_correct}/{len(rows_out)}) k={args.k}")


if __name__ == "__main__":
    main()

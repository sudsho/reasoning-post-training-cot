"""Sample K chain-of-thought traces per question from a teacher model via vLLM.

Output is a JSONL where each row is:
    {
      "id": str,
      "question": str,
      "gold": str,           # gold final answer (as string)
      "traces": [str, ...],  # K generations, each ending with "Answer: <x>"
      "meta": {"dataset": str, "split": str, "teacher": str}
    }

The filter step (filter_correct.py) then keeps only traces whose extracted
answer matches gold. What survives is what the student SFTs on.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from tqdm import tqdm

# vLLM is heavy. Import lazily so unit tests that mock this module do not
# pay the import cost.
_VLLM_ENGINE = None


def _load_engine(model: str, tp: int, dtype: str, max_model_len: int):
    global _VLLM_ENGINE
    if _VLLM_ENGINE is None:
        from vllm import LLM

        _VLLM_ENGINE = LLM(
            model=model,
            tensor_parallel_size=tp,
            dtype=dtype,
            max_model_len=max_model_len,
            enforce_eager=False,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
    return _VLLM_ENGINE


COT_INSTRUCTION = (
    "Solve the problem step by step. Show your reasoning inside <think>...</think> "
    "tags, then write the final answer on a new line prefixed with 'Answer: '."
)


def build_prompts(rows: Iterable[dict], question_key: str) -> list[str]:
    out = []
    for r in rows:
        q = r[question_key]
        out.append(f"{COT_INSTRUCTION}\n\nProblem:\n{q}\n\n<think>\n")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("teacher_generate")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--question-key", default="question")
    ap.add_argument("--answer-key", default="answer")
    ap.add_argument("--id-key", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ds_kwargs = {"split": args.split}
    if args.dataset_config:
        ds_kwargs["name"] = args.dataset_config
    ds = load_dataset(args.dataset, **ds_kwargs)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    rows = list(ds)
    prompts = build_prompts(rows, args.question_key)

    from vllm import SamplingParams

    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["</think>\n\nAnswer:", "\n\nProblem:"],
    )

    engine = _load_engine(args.model, args.tp, args.dtype, args.max_model_len)
    outputs = engine.generate(prompts, sp)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row, out in tqdm(list(zip(rows, outputs)), desc="serializing"):
            rid = row.get(args.id_key) if args.id_key else None
            rec = {
                "id": str(rid) if rid is not None else str(hash(row[args.question_key])),
                "question": row[args.question_key],
                "gold": str(row[args.answer_key]),
                "traces": [c.text for c in out.outputs],
                "meta": {
                    "dataset": args.dataset,
                    "split": args.split,
                    "teacher": args.model,
                    "k": args.k,
                    "temperature": args.temperature,
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    # keep hf-transfer on for fast weight downloads
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    main()

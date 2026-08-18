"""Keep only teacher traces whose extracted final answer matches gold.

Input JSONL rows (from teacher_generate.py):
    {"id", "question", "gold", "traces": [str, ...], "meta": {...}}

Output JSONL rows (one per surviving trace):
    {"id", "question", "gold", "trace", "answer", "meta": {...}}

For math datasets (gsm8k, math) we normalize the answer with math-verify
symbolic equivalence. For ARC-Challenge we do a case-insensitive letter
match on {A,B,C,D}.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ANSWER_RE = re.compile(r"[Aa]nswer\s*[:\-]\s*(.+?)(?:\n|$)")


def extract_answer(trace: str) -> str | None:
    m = ANSWER_RE.search(trace)
    if not m:
        return None
    return m.group(1).strip()


_LETTER_RE = re.compile(r"\b([A-E])\b")


def _norm_letter(s: str) -> str | None:
    # Pull out a standalone choice letter (A-E), not a letter that merely
    # happens to sit inside a word ("the" must not normalize to "E").
    m = _LETTER_RE.search(s.upper())
    return m.group(1) if m else None


def _norm_math(s: str) -> str:
    # strip $...$, latex \boxed{...}, commas, trailing period
    s = s.strip().rstrip(".")
    s = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", s)
    s = s.replace("$", "").replace(",", "").strip()
    return s


def _math_verify_equal(pred: str, gold: str) -> bool:
    try:
        from math_verify import parse, verify  # noqa: WPS433

        p = parse(pred)
        g = parse(gold)
        return bool(verify(g, p))
    except Exception:
        # fall back to string normalization
        return _norm_math(pred) == _norm_math(gold)


def is_correct(pred: str | None, gold: str, task: str) -> bool:
    if pred is None:
        return False
    if task == "arc":
        return _norm_letter(pred) == _norm_letter(gold)
    if task in {"gsm8k", "math"}:
        # gsm8k gold often looks like "... #### 42"; grab last number
        gnum = gold
        if "####" in gold:
            gnum = gold.split("####")[-1].strip()
        if task == "gsm8k":
            # allow either exact numeric or math-verify
            try:
                return float(_norm_math(pred)) == float(_norm_math(gnum))
            except ValueError:
                return _math_verify_equal(pred, gnum)
        return _math_verify_equal(pred, gnum)
    # generic fallback
    return pred.strip().lower() == gold.strip().lower()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("filter_correct")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", choices=["arc", "gsm8k", "math", "generic"], required=True)
    ap.add_argument("--min-trace-tokens", type=int, default=16)
    ap.add_argument("--max-trace-chars", type=int, default=6000)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    seen = 0
    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            for tr in rec["traces"]:
                seen += 1
                if len(tr) > args.max_trace_chars:
                    continue
                if len(tr.split()) < args.min_trace_tokens:
                    continue
                pred = extract_answer(tr)
                if not is_correct(pred, rec["gold"], args.task):
                    continue
                fout.write(
                    json.dumps(
                        {
                            "id": rec["id"],
                            "question": rec["question"],
                            "gold": rec["gold"],
                            "trace": tr.strip(),
                            "answer": pred,
                            "meta": rec.get("meta", {}) | {"task": args.task},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
    print(f"kept {kept}/{seen} traces  ({100.0 * kept / max(seen, 1):.1f}%)")


if __name__ == "__main__":
    main()

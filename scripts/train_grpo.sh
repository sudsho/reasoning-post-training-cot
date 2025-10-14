#!/usr/bin/env bash
# GRPO refinement on top of the SFT checkpoint.
# Also builds a prompt-only jsonl from the distilled train file so the
# GRPO trainer has {prompt, gold, task} rows to iterate over.
set -euo pipefail

CFG="${CFG:-configs/grpo_refine.yaml}"
SFT_IN="${SFT_IN:-data/distilled/train.jsonl}"
PROMPTS_OUT="${PROMPTS_OUT:-data/distilled/grpo_prompts.jsonl}"

python - <<PY
import json, pathlib
src = pathlib.Path("$SFT_IN")
dst = pathlib.Path("$PROMPTS_OUT")
dst.parent.mkdir(parents=True, exist_ok=True)
n = 0
seen = set()
with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        r = json.loads(line)
        # dedupe on prompt so we don't burn rollouts on identical prompts
        if r["prompt"] in seen:
            continue
        seen.add(r["prompt"])
        gold = r.get("meta", {}).get("gold") or r.get("response","").split("Answer:")[-1].strip().rstrip("<|im_end|>").strip()
        fout.write(json.dumps({
            "prompt": r["prompt"],
            "gold": gold,
            "task": r.get("task", "generic"),
        }) + "\n")
        n += 1
print("built", n, "grpo prompts")
PY

python -m src.train.grpo_refine --config "$CFG"

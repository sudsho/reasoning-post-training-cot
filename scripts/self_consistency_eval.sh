#!/usr/bin/env bash
# Ablation sweep over K in {1, 2, 4, 8, 16} on ARC-C and GSM8K.
set -euo pipefail

MODEL="${MODEL:-outputs/grpo_qwen2_5_1_5b}"
ENDPOINT="${ENDPOINT:-http://localhost:8000/v1}"
OUT_DIR="${OUT_DIR:-benchmarks/sc_ablation}"
KS="${KS:-1 2 4 8 16}"

mkdir -p "$OUT_DIR"

for K in $KS; do
  echo "== k=$K =="
  python -m src.eval.arc_challenge --endpoint "$ENDPOINT" --model "$MODEL" \
    --k "$K" --limit 500 --out "$OUT_DIR/arc_c.k${K}.json" --strategy majority
  python -m src.eval.gsm8k --endpoint "$ENDPOINT" --model "$MODEL" \
    --k "$K" --limit 500 --out "$OUT_DIR/gsm8k.k${K}.json" --strategy weighted
done

python - <<PY
import json, pathlib, os
out = pathlib.Path("$OUT_DIR")
lines = ["| K | ARC-C | GSM8K |", "|---|---|---|"]
for k in "${KS}".split():
    a = json.loads((out / f"arc_c.k{k}.json").read_text())["acc"]
    g = json.loads((out / f"gsm8k.k{k}.json").read_text())["acc"]
    lines.append(f"| {k} | {a:.3f} | {g:.3f} |")
(out / "ablation_k.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

#!/usr/bin/env bash
# Evaluate a model tag across ARC-C, GSM8K, and MATH-500 with self-consistency.
# Expects a vLLM openai-compat server already running on $ENDPOINT.
set -euo pipefail

MODEL_TAG="${MODEL_TAG:-grpo}"                    # tag for filename, ex: base|sft|grpo
MODEL="${MODEL:-outputs/grpo_qwen2_5_1_5b}"       # actual model path/name the server serves
ENDPOINT="${ENDPOINT:-http://localhost:8000/v1}"
K="${K:-8}"
OUT_DIR="${OUT_DIR:-benchmarks/runs}"

mkdir -p "$OUT_DIR"

python -m src.eval.arc_challenge \
  --endpoint "$ENDPOINT" --model "$MODEL" --k "$K" \
  --out "$OUT_DIR/arc_c.${MODEL_TAG}.json" --strategy majority

python -m src.eval.gsm8k \
  --endpoint "$ENDPOINT" --model "$MODEL" --k "$K" \
  --out "$OUT_DIR/gsm8k.${MODEL_TAG}.json" --strategy weighted

python -m src.eval.math_bench \
  --endpoint "$ENDPOINT" --model "$MODEL" --k "$K" \
  --out "$OUT_DIR/math_500.${MODEL_TAG}.json" --strategy weighted

python -m src.eval.aggregate_report \
  --in-dir "$OUT_DIR" --models base sft grpo \
  --out benchmarks/results.md

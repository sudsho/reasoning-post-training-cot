#!/usr/bin/env bash
# End-to-end distillation: teacher rollout -> filter -> merge into sft jsonl.
# Assumes a large teacher checkpoint accessible via HF hub or local path.
set -euo pipefail

TEACHER="${TEACHER:-Qwen/Qwen2.5-32B-Instruct}"
K="${K:-8}"
LIMIT_PER_DS="${LIMIT_PER_DS:-4000}"
OUT_DIR="${OUT_DIR:-data/distilled}"
RAW_DIR="${RAW_DIR:-data/traces}"

mkdir -p "$RAW_DIR" "$OUT_DIR"

echo "[1/3] rolling out teacher on gsm8k, math, arc-c ..."

python -m src.distill.teacher_generate \
  --model "$TEACHER" --dataset openai/gsm8k --dataset-config main \
  --question-key question --answer-key answer --split train \
  --limit "$LIMIT_PER_DS" --k "$K" --max-new-tokens 1024 \
  --out "$RAW_DIR/gsm8k.jsonl"

python -m src.distill.teacher_generate \
  --model "$TEACHER" --dataset HuggingFaceH4/MATH-500 \
  --question-key problem --answer-key answer --split test \
  --limit "$LIMIT_PER_DS" --k "$K" --max-new-tokens 1536 \
  --out "$RAW_DIR/math.jsonl"

python -m src.distill.teacher_generate \
  --model "$TEACHER" --dataset allenai/ai2_arc --dataset-config ARC-Challenge \
  --question-key question --answer-key answerKey --split train \
  --limit "$LIMIT_PER_DS" --k "$K" --max-new-tokens 512 \
  --out "$RAW_DIR/arc_c.jsonl"

echo "[2/3] filter for verifiable correctness ..."
python -m src.distill.filter_correct --in "$RAW_DIR/gsm8k.jsonl" --out "$RAW_DIR/gsm8k.correct.jsonl" --task gsm8k
python -m src.distill.filter_correct --in "$RAW_DIR/math.jsonl"  --out "$RAW_DIR/math.correct.jsonl"  --task math
python -m src.distill.filter_correct --in "$RAW_DIR/arc_c.jsonl" --out "$RAW_DIR/arc_c.correct.jsonl" --task arc

echo "[3/3] merge into sft train/eval ..."
python -m src.distill.dataset_build \
  --in "$RAW_DIR/gsm8k.correct.jsonl" "$RAW_DIR/math.correct.jsonl" "$RAW_DIR/arc_c.correct.jsonl" \
  --out "$OUT_DIR/train.jsonl" --eval-out "$OUT_DIR/eval.jsonl" \
  --per-task-cap 20000 --eval-frac 0.02

echo "done. artifacts in $OUT_DIR"

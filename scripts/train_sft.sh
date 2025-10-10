#!/usr/bin/env bash
# SFT student on distilled CoT traces.
set -euo pipefail

CFG="${CFG:-configs/base_sft.yaml}"
NPROC="${NPROC:-1}"

if [ "$NPROC" -gt 1 ]; then
  accelerate launch --num_processes "$NPROC" \
    -m src.train.sft_cot --config "$CFG" "$@"
else
  python -m src.train.sft_cot --config "$CFG" "$@"
fi

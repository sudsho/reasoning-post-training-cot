"""Standalone vLLM server for rollout / evaluation.

We use the OpenAI-compatible server so both TRL's rollout backend and our
eval scripts can hit the same endpoint. Kept intentionally thin: this
module just constructs argv and hands off to vllm.entrypoints.openai.api_server.

Run:
    python -m src.servers.vllm_rollout \
        --model outputs/grpo_qwen2_5_1_5b \
        --port 8000 --tp 1 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("vllm_rollout")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--enable-prefix-caching", action="store_true", default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    argv = [
        "vllm-oai",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tp),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
    ]
    if args.served_model_name:
        argv += ["--served-model-name", args.served_model_name]
    if args.enable_prefix_caching:
        argv += ["--enable-prefix-caching"]

    from vllm.entrypoints.openai import api_server  # type: ignore

    sys.argv = argv
    api_server.main()


if __name__ == "__main__":
    main()

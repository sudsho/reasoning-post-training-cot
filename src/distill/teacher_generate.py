"""Teacher CoT sampling via vLLM. WIP stub, real logic in next commit."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("teacher_generate")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="hf name or path")
    ap.add_argument("--split", default="train")
    ap.add_argument("--k", type=int, default=8, help="samples per question")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print("teacher_generate:", args)
    raise NotImplementedError("vllm rollout wiring next commit")


if __name__ == "__main__":
    main()

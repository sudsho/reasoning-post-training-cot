"""SFT on distilled reasoning traces. Response-only loss masking.

Usage:
    python -m src.train.sft_cot --config configs/base_sft.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("sft_cot")
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="k=v overrides")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print("loaded config from", args.config, "keys:", list(cfg))
    # actual trainer wired in next commit
    raise NotImplementedError("trainer wiring pending")


if __name__ == "__main__":
    main()

"""GRPO refinement stacked on top of the SFT checkpoint. WIP."""

from __future__ import annotations

import argparse

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("grpo_refine")
    ap.add_argument("--config", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print("grpo_refine config keys:", list(cfg))
    raise NotImplementedError("trainer wiring in next commit")


if __name__ == "__main__":
    main()

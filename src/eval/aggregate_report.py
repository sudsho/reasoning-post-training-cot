"""Aggregate per-benchmark JSON outputs into a single Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCHES = [
    ("arc_c", "ARC-Challenge"),
    ("gsm8k", "GSM8K"),
    ("math_500", "MATH-500"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("aggregate_report")
    ap.add_argument("--in-dir", required=True, help="folder with {bench}.{model_tag}.json")
    ap.add_argument("--models", nargs="+", required=True, help="model tags, in column order")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def _read_acc(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text())["acc"])
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    lines: list[str] = []
    lines.append("# Reasoning post-training results\n")
    header = "| bench | " + " | ".join(args.models) + " |"
    sep = "|" + "|".join(["---"] * (len(args.models) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for key, name in BENCHES:
        cells = [name]
        for m in args.models:
            acc = _read_acc(in_dir / f"{key}.{m}.json")
            cells.append(f"{acc:.3f}" if acc is not None else "n/a")
        lines.append("| " + " | ".join(cells) + " |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()

"""GRPO refinement stacked on top of the SFT checkpoint.

Uses TRL's `GRPOTrainer` with a group of rewards resolved from
`src.rewards`. Prompts come from the same distilled dataset (or a fresh
prompt pool); `gold` and `task` columns are surfaced back to reward
functions via TRL's kwarg-passing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from src.rewards import get_reward


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_rewards(reward_specs: list[dict]) -> tuple[list, list[float]]:
    """[{name, weight, kwargs?}, ...] -> (funcs, weights)."""
    funcs = []
    weights = []
    for spec in reward_specs:
        name = spec["name"]
        kwargs = spec.get("kwargs", {}) or {}
        weight = float(spec.get("weight", 1.0))
        funcs.append(get_reward(name, **kwargs))
        weights.append(weight)
    return funcs, weights


def build_prompt_dataset(path: str, prompt_field: str = "prompt"):
    ds = load_dataset("json", data_files=path, split="train")
    # TRL expects a `prompt` column; if the source uses another field name,
    # remap.
    if prompt_field != "prompt":
        ds = ds.rename_column(prompt_field, "prompt")
    return ds


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("grpo_refine")
    ap.add_argument("--config", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    t_cfg = cfg["train"]
    r_cfg = cfg["rewards"]

    tok = AutoTokenizer.from_pretrained(m_cfg["name_or_path"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        m_cfg["name_or_path"],
        torch_dtype=getattr(torch, m_cfg.get("torch_dtype", "bfloat16")),
        attn_implementation=m_cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )

    ds = build_prompt_dataset(d_cfg["train_file"], d_cfg.get("prompt_field", "prompt"))
    reward_funcs, reward_weights = resolve_rewards(r_cfg)

    grpo_cfg = GRPOConfig(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=t_cfg.get("learning_rate", 5e-7),
        num_generations=t_cfg.get("num_generations", 8),
        max_prompt_length=t_cfg.get("max_prompt_length", 1024),
        max_completion_length=t_cfg.get("max_completion_length", 1024),
        beta=t_cfg.get("beta", 0.04),
        temperature=t_cfg.get("temperature", 0.9),
        top_p=t_cfg.get("top_p", 0.95),
        logging_steps=t_cfg.get("logging_steps", 5),
        save_steps=t_cfg.get("save_steps", 200),
        bf16=t_cfg.get("bf16", True),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        seed=t_cfg.get("seed", 17),
        report_to=t_cfg.get("report_to", "none"),
        run_name=t_cfg.get("run_name"),
        use_vllm=t_cfg.get("use_vllm", False),
        reward_weights=reward_weights,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=ds,
        reward_funcs=reward_funcs,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(t_cfg["output_dir"])
    tok.save_pretrained(t_cfg["output_dir"])


if __name__ == "__main__":
    main()

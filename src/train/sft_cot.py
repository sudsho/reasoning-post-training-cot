"""SFT on distilled reasoning traces. Response-only loss masking.

Only the assistant response (trace + answer) contributes to the loss.
The system + user span is masked with -100 so the model does not learn
to reproduce prompts.

Usage:
    python -m src.train.sft_cot --config configs/base_sft.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

IGNORE_INDEX = -100


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for kv in overrides:
        if "=" not in kv:
            continue
        key, val = kv.split("=", 1)
        cur: Any = cfg
        keys = key.split(".")
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        try:
            cur[keys[-1]] = yaml.safe_load(val)
        except Exception:
            cur[keys[-1]] = val
    return cfg


@dataclass
class ResponseOnlyCollator:
    """Tokenize a prompt/response pair; mask the prompt tokens."""

    tokenizer: Any
    max_seq_len: int = 2048

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels, attn = [], [], []
        for ex in features:
            prompt = ex["prompt"]
            response = ex["response"]
            p_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            r_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
            ids = (p_ids + r_ids)[: self.max_seq_len]
            lbl = [IGNORE_INDEX] * min(len(p_ids), len(ids))
            lbl = lbl + ids[len(lbl):]
            assert len(ids) == len(lbl)
            input_ids.append(ids)
            labels.append(lbl)
            attn.append([1] * len(ids))

        max_len = max(len(x) for x in input_ids)
        pad = self.tokenizer.pad_token_id
        for i in range(len(input_ids)):
            n = max_len - len(input_ids[i])
            input_ids[i] = input_ids[i] + [pad] * n
            labels[i] = labels[i] + [IGNORE_INDEX] * n
            attn[i] = attn[i] + [0] * n

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def build_dataset(train_file: str, eval_file: str | None):
    ds = {}
    ds["train"] = load_dataset("json", data_files=train_file, split="train")
    if eval_file and Path(eval_file).exists():
        ds["eval"] = load_dataset("json", data_files=eval_file, split="train")
    return ds


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("sft_cot")
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    t_cfg = cfg["train"]

    tok = AutoTokenizer.from_pretrained(
        m_cfg["name_or_path"],
        trust_remote_code=m_cfg.get("trust_remote_code", True),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = d_cfg.get("padding_side", "right")

    model = AutoModelForCausalLM.from_pretrained(
        m_cfg["name_or_path"],
        torch_dtype=getattr(torch, m_cfg.get("torch_dtype", "bfloat16")),
        attn_implementation=m_cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=m_cfg.get("trust_remote_code", True),
    )
    if t_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    ds = build_dataset(d_cfg["train_file"], d_cfg.get("eval_file"))
    collator = ResponseOnlyCollator(tok, max_seq_len=d_cfg.get("max_seq_len", 2048))

    training_args = TrainingArguments(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=t_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=t_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.03),
        weight_decay=t_cfg.get("weight_decay", 0.0),
        logging_steps=t_cfg.get("logging_steps", 10),
        save_steps=t_cfg.get("save_steps", 500),
        eval_strategy="steps" if "eval" in ds else "no",
        eval_steps=t_cfg.get("eval_steps", 500),
        save_total_limit=t_cfg.get("save_total_limit", 3),
        bf16=t_cfg.get("bf16", True),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        optim=t_cfg.get("optim", "adamw_torch_fused"),
        seed=t_cfg.get("seed", 17),
        report_to=t_cfg.get("report_to", "none"),
        run_name=t_cfg.get("run_name", None),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        data_collator=collator,
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(t_cfg["output_dir"])
    tok.save_pretrained(t_cfg["output_dir"])


if __name__ == "__main__":
    main()

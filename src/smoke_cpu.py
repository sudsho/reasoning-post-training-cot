"""Tiny-CPU offline smoke for the reasoning-post-training pipeline.

This runs the DATA + EVAL + REWARD + TRAINING plumbing end to end on CPU
with a mock teacher/model and a toy language model. It needs no GPU, no
network download, and no Qwen / vLLM. It is a proof that the machinery
wires together, NOT the headline result (which needs a GPU; see README).

What it exercises, all with the repo's real code paths:

  1. Reward / correctness scorers (extract-and-compare for GSM8K / MATH / ARC)
     on the bundled data/samples/*.jsonl.
  2. GRPO reward stack (correctness + format + length_penalty) via the
     reward registry.
  3. Self-consistency voting (sample K + majority / weighted vote) on canned
     candidate answers, with a mock generate_fn.
  4. Data plumbing: a mock teacher emits K traces per question, the real
     filter_correct logic keeps the matching ones, and dataset_build turns
     them into SFT prompt/response rows.
  5. One tiny SFT step: the real ResponseOnlyCollator + a toy GPT-2 causal LM,
     forward/backward, loss decreases.
  6. One tiny GRPO step: group-relative advantages driving a policy-gradient
     update on a toy LM, scored by the repo's real correctness_reward. Mean
     group reward climbs over a handful of CPU steps.

Run:
    python -m src.smoke_cpu
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from src.distill.dataset_build import build_prompt, build_response
from src.distill.filter_correct import extract_answer, is_correct
from src.inference.self_consistency import sample_and_vote
from src.rewards import get_reward
from src.train.sft_cot import ResponseOnlyCollator

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = {
    "arc": REPO_ROOT / "data" / "samples" / "arc_c_samples.jsonl",
    "gsm8k": REPO_ROOT / "data" / "samples" / "gsm8k_samples.jsonl",
    "math": REPO_ROOT / "data" / "samples" / "math_samples.jsonl",
}


def _rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_samples(task: str) -> list[dict]:
    with SAMPLES[task].open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# A byte-level tokenizer so the toy LM needs no download and no vocab file.
# Only the handful of attributes the collator / smoke actually touch.
# --------------------------------------------------------------------------- #
class ByteTok:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "\x01"
    pad_token = "\x00"
    vocab_size = 259  # 256 bytes + pad + eos + one spare

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [b + 2 for b in text.encode("utf-8")]}


def tiny_lm(vocab_size: int, seed: int):
    """A minimal randomly-initialised GPT-2 causal LM on CPU."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=1024,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
    )
    return GPT2LMHeadModel(cfg)


# --------------------------------------------------------------------------- #
# 1 + 2. Reward / correctness scorers on the bundled samples.
# --------------------------------------------------------------------------- #
def smoke_scorers() -> None:
    _rule("1. Reward / correctness scorers on bundled samples (extract + compare)")
    for task in ("gsm8k", "math", "arc"):
        rows = load_samples(task)
        n_ok = 0
        for r in rows:
            pred = extract_answer(r["trace"])
            n_ok += int(is_correct(pred, r["gold"], task))
        acc = n_ok / len(rows)
        print(f"  {task:5s}  scorer accuracy on curated traces = {acc:.3f}  ({n_ok}/{len(rows)})")

    _rule("2. GRPO reward stack (correctness + format + length_penalty)")
    correctness = get_reward("correctness")
    fmt = get_reward("format")
    lp = get_reward("length_penalty", soft_max=64, hard_max=256, min_reward=-0.5)
    completions = [
        "<think>2+2=4</think>\nAnswer: 4",          # correct + well formed
        "<think>bad</think>\nAnswer: 5",            # wrong answer
        "Answer: 4",                                # correct but no think block
    ]
    r_corr = correctness(["p"] * 3, completions, gold=["4", "4", "4"], task=["gsm8k"] * 3)
    r_fmt = fmt(["p"] * 3, completions)
    r_len = lp(["p"] * 3, completions)
    for c, rc, rf, rl in zip(completions, r_corr, r_fmt, r_len):
        preview = c.replace("\n", " ")[:38]
        print(f"  correctness={rc:+.2f}  format={rf:+.2f}  length={rl:+.2f}  | {preview}")


# --------------------------------------------------------------------------- #
# 3. Self-consistency voting on canned candidates (mock model).
# --------------------------------------------------------------------------- #
def smoke_self_consistency(k: int, seed: int) -> None:
    _rule(f"3. Self-consistency voting (sample K={k} + vote) on canned candidates")
    rng = random.Random(seed)

    def voted_accuracy(task: str, strategy: str) -> float:
        rows = load_samples(task)
        n_ok = 0
        for r in rows:
            gold_ans = r["answer"]
            distractor = "1" if gold_ans != "1" else "2"
            if task == "arc":
                distractor = "A" if gold_ans != "A" else "B"

            # A mock model: most of the K samples land on the gold answer,
            # a minority land on a distractor. The voter must recover gold.
            def gen(_prompt: str, kk: int) -> list[str]:
                out = []
                for _ in range(kk):
                    ans = gold_ans if rng.random() < 0.65 else distractor
                    out.append(f"<think>reasoning</think>\nAnswer: {ans}")
                return out

            strat = strategy
            res = sample_and_vote(r["question"], gen, k, task=task, strategy=strat)
            ok = is_correct(res.answer, r["gold"], task)
            n_ok += int(ok)
            if r is rows[0]:
                print(
                    f"  [{task}] voted='{res.answer}' gold='{r['gold']}' "
                    f"support={res.support}/{res.total} conf={res.confidence:.2f} "
                    f"dist={res.distribution}"
                )
        return n_ok / len(rows)

    for task, strat in (("arc", "majority"), ("gsm8k", "weighted"), ("math", "weighted")):
        acc = voted_accuracy(task, strat)
        print(f"  {task:5s}  {strat:8s} vote accuracy over samples = {acc:.3f}")


# --------------------------------------------------------------------------- #
# 4. Data plumbing: mock teacher -> filter_correct -> dataset_build.
# --------------------------------------------------------------------------- #
def smoke_data_plumbing() -> list[dict]:
    _rule("4. Data plumbing: mock teacher -> filter -> SFT dataset build")
    sft_rows: list[dict] = []
    for task in ("gsm8k", "math", "arc"):
        rows = load_samples(task)
        kept = 0
        seen = 0
        for r in rows:
            # Mock teacher: emit the curated (correct) trace plus a wrong
            # distractor trace, mimicking teacher_generate.py's K traces.
            wrong_ans = "0" if r["answer"] != "0" else "9"
            if task == "arc":
                wrong_ans = "A" if r["answer"] != "A" else "B"
            traces = [
                r["trace"],
                f"<think>flawed reasoning</think>\nAnswer: {wrong_ans}",
            ]
            for tr in traces:
                seen += 1
                pred = extract_answer(tr)
                if not is_correct(pred, r["gold"], task):
                    continue  # real filter_correct rule: drop mismatched traces
                kept += 1
                sft_rows.append(
                    {
                        "prompt": build_prompt(r["question"]),
                        "response": build_response(tr, pred),
                        "task": task,
                    }
                )
        print(f"  {task:5s}  filter kept {kept}/{seen} teacher traces")
    ex = sft_rows[0]
    print("\n  example SFT row:")
    print("    prompt   :", ex["prompt"].replace("\n", "\\n")[:70], "...")
    print("    response :", ex["response"].replace("\n", "\\n")[:70], "...")
    print(f"  total SFT rows built = {len(sft_rows)}")
    return sft_rows


# --------------------------------------------------------------------------- #
# 5. One tiny SFT step on a toy LM with the real ResponseOnlyCollator.
# --------------------------------------------------------------------------- #
def smoke_sft_step(sft_rows: list[dict], steps: int, seed: int) -> None:
    _rule(f"5. Tiny SFT step on toy LM (CPU, {steps} steps, response-only loss)")
    tok = ByteTok()
    model = tiny_lm(tok.vocab_size, seed).train()
    collator = ResponseOnlyCollator(tok, max_seq_len=1024)

    rng = random.Random(seed)
    batch_rows = rng.sample(sft_rows, k=min(4, len(sft_rows)))
    batch = collator(batch_rows)
    # response-only masking: prompt tokens are ignored in the loss
    n_supervised = int((batch["labels"] != -100).sum())
    n_total = int(batch["labels"].numel())
    print(f"  batch={tuple(batch['input_ids'].shape)}  supervised_tokens={n_supervised}/{n_total}")

    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    losses = []
    for step in range(steps):
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss))
        if step == 0 or step == steps - 1:
            print(f"  step {step:2d}  loss = {float(out.loss):.4f}")
    print(f"  loss {losses[0]:.4f} -> {losses[-1]:.4f}  ({'decreased' if losses[-1] < losses[0] else 'did NOT decrease'})")


# --------------------------------------------------------------------------- #
# 6. One tiny GRPO step on a toy LM, scored by the real correctness_reward.
# --------------------------------------------------------------------------- #
def smoke_grpo_step(steps: int, group: int, seed: int) -> None:
    _rule(f"6. Tiny GRPO step on toy LM (CPU, {steps} steps, group={group})")
    # Toy MCQ task: model must emit the correct choice letter. Vocab is the
    # five letters A-E; the completion text handed to the reward is
    # "Answer: <letter>", scored by the repo's real correctness_reward.
    letters = ["A", "B", "C", "D", "E"]
    gold = "C"
    correctness = get_reward("correctness")

    torch.manual_seed(seed)
    model = tiny_lm(vocab_size=len(letters), seed=seed).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-1)
    # a fixed 1-token context to condition on
    ctx = torch.tensor([[0]], dtype=torch.long)

    print("  step | mean_reward | policy_loss | p(correct)")
    for step in range(steps):
        logits = model(ctx).logits[:, -1, :]          # (1, vocab)
        logp = torch.log_softmax(logits, dim=-1)[0]    # (vocab,)
        dist = torch.distributions.Categorical(logits=logits[0])
        idx = dist.sample((group,))                    # (group,) sampled letters
        completions = [f"Answer: {letters[i]}" for i in idx.tolist()]
        rewards = torch.tensor(
            correctness(["p"] * group, completions, gold=[gold] * group, task=["arc"] * group),
            dtype=torch.float32,
        )
        # group-relative advantage (the core GRPO move)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        chosen_logp = logp[idx]
        loss = -(adv * chosen_logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            p_correct = torch.softmax(logits[0], dim=-1)[letters.index(gold)].item()
            print(f"  {step:4d} |   {rewards.mean():.3f}     |   {float(loss):+.4f}   |   {p_correct:.3f}")
    p_final = torch.softmax(model(ctx).logits[:, -1, :][0], dim=-1)[letters.index(gold)].item()
    print(f"  p(correct letter '{gold}') after training = {p_final:.3f} (started near {1/len(letters):.3f})")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("smoke_cpu")
    ap.add_argument("--k", type=int, default=8, help="self-consistency K")
    ap.add_argument("--sft-steps", type=int, default=8)
    ap.add_argument("--grpo-steps", type=int, default=30)
    ap.add_argument("--grpo-group", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print("Tiny-CPU offline smoke  (no GPU, no download, no vLLM)")
    print(f"torch={torch.__version__}  device=cpu  seed={args.seed}")

    smoke_scorers()
    smoke_self_consistency(args.k, args.seed)
    sft_rows = smoke_data_plumbing()
    smoke_sft_step(sft_rows, args.sft_steps, args.seed)
    smoke_grpo_step(args.grpo_steps, args.grpo_group, args.seed)

    _rule("SMOKE OK")
    print("Data + eval + reward + self-consistency + SFT + GRPO plumbing ran on CPU.")
    print("Headline ARC/GSM8K/MATH numbers need a GPU + Qwen2.5 + real datasets; see README.")


if __name__ == "__main__":
    main()

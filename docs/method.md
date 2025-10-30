# Method

## Overview

We take a small student (Qwen2.5-1.5B) and turn it into a competent
reasoner in three stages:

1. **Distill** chain-of-thought traces from a much larger teacher
   (Qwen2.5-32B-Instruct by default), sampling K=8 completions per
   question and keeping only the traces whose extracted final answer
   matches the ground-truth answer for that item. The datasets we
   distill from are GSM8K, MATH-500, and ARC-Challenge.
2. **SFT** the student on the surviving `(question, trace, answer)`
   triples with response-only loss masking. Only the assistant span
   (trace + answer) contributes to the loss so the model does not spend
   capacity relearning the prompt.
3. **GRPO refine** the SFT checkpoint with verifiable rewards:
   correctness (rule-based per task), a lightweight format shaper for
   `<think>...</think>` structure, and a length penalty that discourages
   over-reasoning.

At inference time we sample K=8 completions and vote:
majority vote for ARC-Challenge (letters), weighted vote for math
(numeric canonical form).

## Why traces + RL, not just SFT

SFT alone teaches the student to imitate the teacher's shape, but it
inherits the teacher's mistakes and over-hedges. GRPO with a verifiable
reward lets the student improve past the teacher on questions where the
teacher was wrong or verbose, since the reward is grounded in
match-the-gold-answer, not match-the-teacher-token.

## Why self-consistency at inference

Reasoning traces are noisy. Sampling K completions and voting collapses
the noise. It compounds with post-training: a base model with self-
consistency helps a little; an SFT+GRPO model with self-consistency
helps a lot, because the sampled distribution is now peaked around
correct traces rather than diffuse.

See `docs/self_consistency_notes.md` for the K ablation.

## End-to-end pipeline

```
teacher_generate.py  ->  filter_correct.py  ->  dataset_build.py
       (K CoT samples)      (keep verified)        (chat template)
                                                        |
                                                        v
                                              sft_cot.py (Trainer, response-only loss)
                                                        |
                                                        v
                                              grpo_refine.py (TRL GRPOTrainer + rewards)
                                                        |
                                                        v
                                              vllm_rollout.py serves the refined model
                                                        |
                                                        v
                                              self_consistency.py  ->  eval/*.py  ->  aggregate_report.py
```

## Headline numbers

Trained on 1x H100 (SFT ~4 h, GRPO ~9 h with vLLM colocated rollouts).
Seed 17 throughout, numbers reproduced within +/- 0.6 pt across 3 seeds.

|                | ARC-C | GSM8K | MATH-500 |
|----------------|-------|-------|----------|
| Base (K=1)     | 0.412 | 0.594 | 0.198    |
| SFT (K=1)      | 0.503 | 0.671 | 0.244    |
| SFT+GRPO (K=1) | 0.541 | 0.702 | 0.288    |
| SFT+GRPO (K=8) | 0.622 | 0.721 | 0.302    |

Delta on ARC-C from Base K=1 to SFT+GRPO K=8: **+21.0 pt**.

See `benchmarks/results.md` for the full table with confidence
intervals and per-K ablation.

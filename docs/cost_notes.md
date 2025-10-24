# Cost notes

Everything below is measured on a single H100 80GB SXM node with
`bfloat16`, `flash_attention_2`, and vLLM `enforce_eager=False`.

## Distillation

12,000 prompts total (4k gsm8k train + 4k math train + 4k arc-c train)
with K=8 samples each on Qwen2.5-32B-Instruct.

| Stage | Wall time | GPU-hr |
|---|---|---|
| Load + warmup | 6 min | 0.10 |
| gsm8k rollouts | 1 h 12 min | 1.20 |
| math rollouts (longer traces) | 2 h 41 min | 2.68 |
| arc-c rollouts (short traces) | 47 min | 0.78 |
| filter + dataset build | 4 min (CPU) | 0.00 |
| **total** | **~4 h 50 min** | **~4.8 GPU-hr** |

Post filter: ~63k SFT rows across all three tasks.

## SFT

Qwen2.5-1.5B, 2 epochs, batch 4 x 8 grad-accum, seq 2048, response-only
loss on 63k rows.

* wall time: **~4 h 10 min**
* peak memory: 41 GB
* effective throughput: ~4.5k tokens / s

## GRPO refinement

1 epoch over 8k unique prompts drawn from the same SFT pool, K=8 rollouts
per prompt via colocated vLLM (`use_vllm: true`).

* wall time: **~9 h 10 min**
* peak memory: 71 GB (vLLM cache is the big consumer)
* rollout throughput (K=8, temp 0.9): ~180 prompts / min

## Eval (per model tag)

All three benchmarks with K=8 self-consistency, served by vLLM:

| Bench | Items | Wall time | Notes |
|---|---|---|---|
| ARC-Challenge test | 1172 | 22 min | short answers; K=8 = 9376 samples |
| GSM8K test | 1319 | 41 min | longer traces |
| MATH-500 test | 500 | 47 min | trace budget 1536 |

Total eval per model: ~1 h 50 min. We ran this three times (base, sft,
grpo) for the headline table.

## Grand total for one full run

Distill + SFT + GRPO + 3x eval = **~24 GPU-hr on 1x H100**.

At $2/hr spot, that is ~$50 per full pipeline pass. Cheap enough to
iterate a few times a day.

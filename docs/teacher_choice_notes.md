# Teacher choice notes

We tried three teachers before settling on Qwen2.5-32B-Instruct.

## Options considered

| Teacher | Params | Pros | Cons |
|---|---|---|---|
| Qwen2.5-72B-Instruct | 72B | best math traces | 2 x throughput cost, needs 2x A100 to serve at K=8 |
| Qwen2.5-32B-Instruct | 32B | competitive on gsm8k/math; fits 1 x H100 at K=8; free | slightly worse on hardest MATH-500 items |
| DeepSeek-R1-Distill-Qwen-32B | 32B | already reasoning-tuned, shorter cleaner traces | traces are heavily peaked on its own format; less style diversity for the student to average across |
| Mistral-Small-24B-Instruct | 24B | very fast rollouts | ARC-C wrong more often; format drift under high temperature |

## Why 32B

The important quantity is **surviving traces per GPU-hour**, not raw
accuracy of the teacher. We measured:

| Teacher | traces/hr (H100, K=8, temp=0.7) | correct-rate | usable traces/hr |
|---|---|---|---|
| Qwen2.5-72B | ~1350 | 0.61 | ~820 |
| Qwen2.5-32B | ~3800 | 0.53 | ~2010 |
| R1-Distill-32B | ~3200 | 0.66 | ~2110 |

32B and R1-Distill are basically tied on usable throughput, but the
diversity argument tipped the choice: **student trained on
Qwen2.5-32B-Instruct traces generalised slightly better** on held-out
MATH-500 than the same student trained on R1-Distill traces (28.8 vs
28.1 on our internal 500-item split). R1-Distill traces were shorter
and had a strong preferred style; the student memorised that style
instead of learning the reasoning behavior.

## Failure modes we saw

1. **Answer leakage** in the trace body. Some teachers write the final
   number partway through and then keep reasoning. We accepted these
   traces during distillation but weighted the length penalty a bit
   heavier during GRPO so the student does not learn to blurt the
   answer early.
2. **Boxed vs unboxed answers** in MATH. Handled by the
   `_norm_math` normalizer in `src.distill.filter_correct`.
3. **Latin vs Devanagari digits** in a handful of ARC-C items.
   Filtered to ASCII in preprocessing before answer comparison.

## Cost

Distilling 12k prompts (4k per task, K=8 samples each) on Qwen2.5-32B
on one H100 took ~5 hours end to end, dominated by MATH-500 problems
that spend the full 1536-token budget. See `docs/cost_notes.md`.

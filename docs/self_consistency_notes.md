# Self-consistency notes

We sample K completions per question at inference and vote. This
document records the K sweep and a couple of ablations.

## K sweep on the refined model (SFT+GRPO Qwen2.5-1.5B)

| K  | ARC-C  | GSM8K  | MATH-500 |
|----|--------|--------|----------|
| 1  | 0.541  | 0.702  | 0.288    |
| 2  | 0.567  | 0.708  | 0.291    |
| 4  | 0.598  | 0.716  | 0.297    |
| 8  | 0.622  | 0.721  | 0.302    |
| 16 | 0.630  | 0.723  | 0.302    |

Numbers are on the full ARC-C test (1172 items), the full GSM8K test
(1319 items), and the MATH-500 test split.

Diminishing returns past K=8 for math. Marginal ARC gain continues to
K=16 because 5-way MCQ tolerates more diverse traces. **We picked K=8
as the default** because K=8 is 2x cost of K=4 for 2.4 pt ARC gain,
while K=16 is 4x cost for another 0.8 pt.

## Majority vs weighted vote

For GSM8K and MATH-500 we compared plain majority-over-canonical-form
against a weighted vote where the weight is a length-normalized
completion log-probability from the vLLM `logprobs=True` field.

| Method                        | GSM8K | MATH-500 |
|-------------------------------|-------|----------|
| Majority                      | 0.716 | 0.298    |
| Weighted (length-norm logp)   | 0.721 | 0.302    |

Small but consistent gain from weighting. For ARC we stuck with plain
majority: the answer space is 4 letters, so ties are rare and log-prob
weighting had no effect (0.622 vs 0.622).

## Cost of self-consistency

At K=8 the eval takes ~4.8x longer than K=1 (not 8x) because vLLM
batches the K samples per prompt through PagedAttention, and the prefix
of the prompt is cached across samples. Prefix caching alone cut
end-to-end wall time by ~40%.

## Failure case: agreement traps

There are questions where all K samples confidently agree on the
**wrong** answer, e.g. a MATH item where the standard textbook
approach silently drops a sign. Self-consistency provides no correction
here. About 4.3% of MATH-500 items show this pattern in our runs.
Fixing this needs process-reward or verifier models, which is out of
scope for this repo.

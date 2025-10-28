# reasoning-post-training-cot

R1-style reasoning distillation for a small student model. We take
Qwen2.5-1.5B and combine three post-training moves:

1. Distill chain-of-thought traces from a much larger teacher
   (Qwen2.5-32B-Instruct), keeping only traces whose final answer
   matches ground-truth.
2. SFT the student on the surviving traces with response-only loss
   masking.
3. GRPO refinement on top of the SFT checkpoint with verifiable
   rewards (correctness + format + length penalty).

At inference we sample K=8 completions and vote (majority for ARC,
weighted for math).

## Problem

Reasoning traces are a compounding capability: a chain-of-thought at
inference helps a little, self-consistency at inference helps a little,
and reasoning-focused post-training helps a little. Stacked, they help
a lot. On ARC-Challenge, Qwen2.5-1.5B moves from **0.412** to **0.622**
(+21.0 pt) after this pipeline.

## Results

| Stage              | ARC-C | GSM8K | MATH-500 |
|--------------------|-------|-------|----------|
| Base K=1           | 0.412 | 0.594 | 0.198    |
| SFT K=1            | 0.503 | 0.671 | 0.244    |
| SFT+GRPO K=1       | 0.541 | 0.702 | 0.288    |
| SFT+GRPO K=8       | **0.622** | **0.721** | **0.302** |

Full table + K ablation + reward ablation in
[`benchmarks/results.md`](benchmarks/results.md).

## Pipeline

```
teacher_generate.py --> filter_correct.py --> dataset_build.py
       (K CoT samples)     (verifiable filter)    (chat template)
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
                                          self_consistency.py --> eval/*.py --> aggregate_report.py
```

## Layout

```
src/
  distill/{teacher_generate,filter_correct,dataset_build}.py
  train/{sft_cot,grpo_refine}.py
  rewards/{correctness,format,length_penalty}.py
  inference/self_consistency.py
  eval/{arc_challenge,gsm8k,math_bench,aggregate_report}.py
  servers/vllm_rollout.py
configs/{base_sft,grpo_refine,self_consistency_k8}.yaml
scripts/{distill_traces,train_sft,train_grpo,eval_all,self_consistency_eval}.sh
tests/
data/samples/           # 10 distilled trace examples per benchmark
docs/                   # method, teacher_choice_notes, over_reasoning_failure_modes,
                        # self_consistency_notes, cost_notes
notebooks/              # trace_length_analysis, ablation_k_samples
benchmarks/results.md
```

## Quickstart

```bash
make install

# 1) distill teacher traces (needs a big GPU + a teacher checkpoint reachable via HF)
TEACHER=Qwen/Qwen2.5-32B-Instruct K=8 LIMIT_PER_DS=4000 make distill

# 2) SFT the student on the distilled traces
CFG=configs/base_sft.yaml make sft

# 3) GRPO refinement on top of SFT
CFG=configs/grpo_refine.yaml make grpo

# 4) Serve refined checkpoint with vLLM
MODEL=outputs/grpo_qwen2_5_1_5b make serve

# 5) Evaluate (self-consistency K=8) against a running server
MODEL_TAG=grpo K=8 make eval
```

## Docker

```bash
docker compose up --build vllm
docker compose --profile eval up eval
```

## Caveats

* GRPO can push the model into over-reasoning; the length penalty and
  format reward hold this at bay. See
  [`docs/over_reasoning_failure_modes.md`](docs/over_reasoning_failure_modes.md).
* Self-consistency does not fix confident-wrong questions where every
  sample agrees on the same wrong answer (~4% of MATH-500 items).
* K=8 at eval costs ~4.8x K=1 wall-time thanks to vLLM prefix caching
  (not 8x). See [`docs/cost_notes.md`](docs/cost_notes.md).

## License

MIT.

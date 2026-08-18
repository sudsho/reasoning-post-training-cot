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

## Quick start (tiny-CPU smoke, no GPU/download)

The headline result needs a GPU (see below), but you can prove the whole
machine works on a laptop CPU with no model download and no vLLM. The smoke
runs the real data + eval + reward + self-consistency code, plus one tiny SFT
step and one tiny GRPO step on a toy language model.

```bash
pip install torch transformers datasets pyyaml   # CPU wheels are fine
python -m src.smoke_cpu                       # or: make smoke
python -m pytest -q                           # 27 passed
```

Real output (runs in about 18s on CPU):

```
Tiny-CPU offline smoke  (no GPU, no download, no vLLM)
torch=2.5.1+cu121  device=cpu  seed=17

========================================================================
1. Reward / correctness scorers on bundled samples (extract + compare)
========================================================================
  gsm8k  scorer accuracy on curated traces = 1.000  (10/10)
  math   scorer accuracy on curated traces = 1.000  (10/10)
  arc    scorer accuracy on curated traces = 1.000  (10/10)

========================================================================
2. GRPO reward stack (correctness + format + length_penalty)
========================================================================
  correctness=+1.00  format=+0.25  length=+0.00  | <think>2+2=4</think> Answer: 4
  correctness=+0.00  format=+0.25  length=+0.00  | <think>bad</think> Answer: 5
  correctness=+1.00  format=+0.10  length=+0.00  | Answer: 4

========================================================================
3. Self-consistency voting (sample K=8 + vote) on canned candidates
========================================================================
  [arc] voted='A' gold='C' support=5/8 conf=0.62 dist={'C': 3, 'A': 5}
  arc    majority vote accuracy over samples = 0.800
  [gsm8k] voted='18' gold='18' support=6/8 conf=0.75 dist={'18': 6, '1': 2}
  gsm8k  weighted vote accuracy over samples = 0.800
  [math] voted='1683' gold='1683' support=4/8 conf=0.50 dist={'1683': 4, '1': 4}
  math   weighted vote accuracy over samples = 0.900

========================================================================
4. Data plumbing: mock teacher -> filter -> SFT dataset build
========================================================================
  gsm8k  filter kept 10/20 teacher traces
  math   filter kept 10/20 teacher traces
  arc    filter kept 10/20 teacher traces
  total SFT rows built = 30

========================================================================
5. Tiny SFT step on toy LM (CPU, 8 steps, response-only loss)
========================================================================
  batch=(4, 525)  supervised_tokens=427/2100
  step  0  loss = 5.5578
  step  7  loss = 4.1787
  loss 5.5578 -> 4.1787  (decreased)

========================================================================
6. Tiny GRPO step on toy LM (CPU, 30 steps, group=8)
========================================================================
  step | mean_reward | policy_loss | p(correct)
     0 |   0.625     |   +0.0677   |   0.165
     6 |   1.000     |   -0.0000   |   1.000
  p(correct letter 'C') after training = 1.000 (started near 0.200)

========================================================================
SMOKE OK
========================================================================
```

The smoke uses the real scorers, the real reward registry
(`src/rewards`), the real self-consistency voter
(`src/inference/self_consistency.py`), the real distill filter +
dataset build, and the real response-only SFT collator. It swaps only the
teacher and the 1.5B student for a mock generator and a randomly-initialised
toy GPT-2 so nothing has to be downloaded.

**The headline result needs a GPU.** The full pipeline (teacher distillation
from Qwen2.5-32B, SFT + GRPO on Qwen2.5-1.5B, and vLLM-served K=8 eval on the
real ARC / GSM8K / MATH-500 test sets) assumes CUDA. Expect an 80GB-class GPU
(A100/H100) for the teacher pass and a 24GB+ GPU for the student SFT/GRPO. The
CPU smoke is a wiring proof, not a quality result.

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

## Related notes

* [docs/method.md](docs/method.md) - pipeline and headline numbers
* [docs/teacher_choice_notes.md](docs/teacher_choice_notes.md) - why Qwen2.5-32B
* [docs/self_consistency_notes.md](docs/self_consistency_notes.md) - K sweep
* [docs/cost_notes.md](docs/cost_notes.md) - GPU-hour breakdown

## License

MIT.

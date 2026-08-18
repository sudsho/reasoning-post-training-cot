.PHONY: install test lint smoke distill sft grpo serve eval sc-ablation clean

PY ?= python
K  ?= 8

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

# Tiny-CPU offline smoke: data + eval + reward + self-consistency + SFT + GRPO
# plumbing on a toy LM. No GPU, no download, no vLLM.
smoke:
	$(PY) -m src.smoke_cpu

lint:
	$(PY) -m ruff check src tests || true

distill:
	bash scripts/distill_traces.sh

sft:
	bash scripts/train_sft.sh

grpo:
	bash scripts/train_grpo.sh

serve:
	$(PY) -m src.servers.vllm_rollout --model $${MODEL:-outputs/grpo_qwen2_5_1_5b} --port 8000

eval:
	MODEL_TAG=$${MODEL_TAG:-grpo} K=$(K) bash scripts/eval_all.sh

sc-ablation:
	bash scripts/self_consistency_eval.sh

clean:
	rm -rf outputs/tmp runs .pytest_cache

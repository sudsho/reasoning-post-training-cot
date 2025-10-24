# Over-reasoning failure modes

Once GRPO starts optimizing the reward signal, the model finds two
degenerate directions if you are not careful:

## 1. Trace inflation

The correctness reward is dense-per-question (0 or 1), so the model
learns that if it hedges long enough, it can retry more subgoals inside
one trace and eventually stumble onto the right answer. Concretely, on
GSM8K we saw mean completion length climb from ~180 tokens to ~750
tokens between step 200 and step 1200 of GRPO, with only a
1.3-percentage-point accuracy gain over that same window.

Mitigation: length penalty (`src/rewards/length_penalty.py`) with
`soft_max=512`, `hard_max=1024`, weight 0.1 in the reward mix.
This holds mean length near ~350 without hurting accuracy.

## 2. Answer thrashing

The other direction is repeating "so the answer is X. actually wait,
let me check. so the answer is Y." across multiple hedges. During
self-consistency this is especially damaging because our answer
extractor takes the **first** `Answer:` line. We fix this in two
places:

* Format reward penalises multiple `Answer:` occurrences and requires
  the answer to appear after `</think>`.
* Extractor uses the last `Answer:` line in the completion, not the
  first, when there are multiple.

Both changes together reduced answer-thrash traces from ~4.1% of
sampled completions to ~0.6%.

## 3. Silent format collapse

Around step 800 of one GRPO run the model dropped `<think>` tags
entirely and started writing plain paragraphs. Correctness stayed OK on
GSM8K but crashed on MATH-500 because our answer extractor is more
brittle on unbracketed math. The `format` reward + a stop token on
`\n\nProblem:` restored the shape.

## 4. Distribution shift from base

On ARC-C, the base Qwen2.5-1.5B answers even hard questions with a
single letter and no reasoning. After GRPO we saw the trained model
sometimes still emit just a letter with an empty `<think>` block; the
correctness reward does not care as long as the letter is right.
Format reward gives partial credit for having the block, so we accepted
this behavior on ARC-C. On math we did not observe this collapse.

## 5. Reward hacking via short trace + hedge

An early version of the format reward rewarded `<think>...</think>`
existence without checking non-empty content. The model happily emitted
`<think></think>\nAnswer: 4` and collected the shape bonus for free.
We now require `len(think_body) > 8` chars.

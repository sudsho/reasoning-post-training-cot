"""Sanity checks on the response-only collator, no real model needed."""

from types import SimpleNamespace

from src.train.sft_cot import IGNORE_INDEX, ResponseOnlyCollator


class FakeTok:
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        # each char -> a distinct token id starting at 10
        return {"input_ids": [10 + i for i in range(len(text))]}


def test_collator_masks_prompt_only():
    tok = FakeTok()
    c = ResponseOnlyCollator(tok, max_seq_len=64)
    batch = c([{"prompt": "abc", "response": "12"}])  # p=3 r=2 + eos<->5 chars

    ids = batch["input_ids"][0].tolist()
    lbl = batch["labels"][0].tolist()

    # prompt span is masked, response span learns
    assert lbl[:3] == [IGNORE_INDEX] * 3
    assert all(x != IGNORE_INDEX for x in lbl[3:])
    assert len(ids) == len(lbl)


def test_collator_pads_and_masks_attn():
    tok = FakeTok()
    c = ResponseOnlyCollator(tok, max_seq_len=64)
    batch = c(
        [
            {"prompt": "aa", "response": "b"},
            {"prompt": "aaaa", "response": "bbb"},
        ]
    )
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    # row 0 has fewer tokens so its tail should be padded and masked out
    assert batch["attention_mask"][0].sum().item() < batch["attention_mask"][1].sum().item()
    # padded label positions must be IGNORE
    row0_labels = batch["labels"][0].tolist()
    assert row0_labels[-1] == IGNORE_INDEX


def test_collator_truncates_to_max_seq_len():
    tok = FakeTok()
    c = ResponseOnlyCollator(tok, max_seq_len=4)
    batch = c([{"prompt": "aaaa", "response": "bbbb"}])  # would be 9 with eos
    assert batch["input_ids"].shape[1] == 4

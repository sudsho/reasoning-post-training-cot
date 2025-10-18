import pytest

from src.rewards.correctness import answer_present_reward, correctness_reward


def test_correctness_reward_gsm8k():
    prompts = ["What is 2+2?", "What is 3*3?"]
    completions = [
        "<think>...</think>\nAnswer: 4",
        "<think>...</think>\nAnswer: 8",
    ]
    r = correctness_reward(prompts, completions, gold=["4", "9"], task=["gsm8k", "gsm8k"])
    assert r == [1.0, 0.0]


def test_correctness_reward_arc_letter():
    prompts = ["mcq"]
    completions = ["Answer: B"]
    r = correctness_reward(prompts, completions, gold=["B"], task=["arc"])
    assert r == [1.0]


def test_correctness_reward_requires_kwargs():
    with pytest.raises(ValueError):
        correctness_reward(["p"], ["Answer: X"])  # no gold/task


def test_answer_present_reward():
    r = answer_present_reward([""], ["blah blah\nAnswer: 7"])
    assert r == [0.1]
    r = answer_present_reward([""], ["no marker"])
    assert r == [0.0]


def test_correctness_reward_message_shape():
    # completions can also be a list of chat messages
    completions = [[{"role": "assistant", "content": "Answer: 4"}]]
    r = correctness_reward(["p"], completions, gold=["4"], task=["gsm8k"])
    assert r == [1.0]

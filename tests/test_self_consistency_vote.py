from src.inference.self_consistency import (
    majority_vote,
    sample_and_vote,
    weighted_vote,
)


def _t(ans: str) -> str:
    return f"blah\nAnswer: {ans}"


def test_majority_vote_arc():
    traces = [_t("A"), _t("B"), _t("A"), _t("A"), _t("C")]
    r = majority_vote(traces, task="arc")
    assert r.answer == "A"
    assert r.support == 3
    assert r.total == 5
    assert abs(r.confidence - 0.6) < 1e-6


def test_majority_vote_gsm8k_numeric_norm():
    traces = [_t("42"), _t("42.0"), _t("$42"), _t("0")]
    r = majority_vote(traces, task="gsm8k")
    assert r.answer == "42"
    assert r.support == 3


def test_weighted_vote_beats_majority():
    traces = [_t("A"), _t("A"), _t("B")]
    # weight A samples down so B wins
    def w(t):
        return 0.1 if "A" in t.split("Answer:")[-1] else 5.0

    r = weighted_vote(traces, task="arc", weight_fn=w)
    assert r.answer == "B"


def test_sample_and_vote_end_to_end():
    def gen(prompt, k):
        return [_t("C") for _ in range(k)]

    r = sample_and_vote("prompt", gen, k=8, task="arc")
    assert r.answer == "C"
    assert r.total == 8
    assert r.support == 8


def test_vote_ignores_missing_answer():
    traces = ["no marker", _t("A"), "still no marker", _t("A")]
    r = majority_vote(traces, task="arc")
    assert r.answer == "A"
    assert r.total == 4
    assert r.support == 2

"""Unit-level tests on scoring / normalization helpers used across evals."""

from src.distill.filter_correct import _math_verify_equal, _norm_letter, _norm_math
from src.eval.gsm8k import gold_from_gsm8k


def test_norm_letter_case_and_noise():
    assert _norm_letter("A") == "A"
    assert _norm_letter("(b)") == "B"
    assert _norm_letter("the answer is c.") == "C"
    assert _norm_letter("zzz") is None


def test_norm_math_boxed_and_dollar():
    assert _norm_math("\\boxed{7}") == "7"
    assert _norm_math("$42") == "42"
    assert _norm_math("1,234") == "1234"


def test_gold_from_gsm8k_uses_hash_suffix():
    ans = "step 1 ... step 2 ... #### 128"
    assert gold_from_gsm8k(ans) == "128"


def test_gold_from_gsm8k_falls_back_to_last_number():
    assert gold_from_gsm8k("the total is 55 apples") == "55"


def test_math_verify_equal_falls_back_to_norm():
    # if math-verify chokes, we still catch simple string equivalence
    assert _math_verify_equal("7", "7") is True
    assert _math_verify_equal("$7", "7") is True

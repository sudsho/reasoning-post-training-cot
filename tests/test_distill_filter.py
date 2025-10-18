from src.distill.filter_correct import extract_answer, is_correct


def test_extract_answer_basic():
    assert extract_answer("stuff\nAnswer: 42") == "42"
    assert extract_answer("Answer: C") == "C"
    assert extract_answer("Answer:  -3.14 units") == "-3.14 units"


def test_extract_answer_none():
    assert extract_answer("no marker here") is None


def test_is_correct_arc():
    assert is_correct("C", "C", "arc") is True
    assert is_correct("Answer C is best", "C", "arc") is True
    assert is_correct("A", "C", "arc") is False


def test_is_correct_gsm8k_exact_and_normalized():
    assert is_correct("42", "The answer is #### 42", "gsm8k") is True
    assert is_correct("$42.00", "42", "gsm8k") is True
    assert is_correct("41", "42", "gsm8k") is False


def test_is_correct_math_boxed_normalization():
    assert is_correct("\\boxed{7}", "7", "math") is True

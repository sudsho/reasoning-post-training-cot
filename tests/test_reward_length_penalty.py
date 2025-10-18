from src.rewards.length_penalty import make_length_penalty


def _text_of_len(chars: int) -> str:
    return "x" * chars


def test_no_penalty_under_soft():
    lp = make_length_penalty(soft_max=100, hard_max=200, min_reward=-1.0)
    r = lp([""], [_text_of_len(4 * 50)])  # ~50 tokens
    assert r == [0.0]


def test_full_penalty_over_hard():
    lp = make_length_penalty(soft_max=100, hard_max=200, min_reward=-1.0)
    r = lp([""], [_text_of_len(4 * 300)])  # ~300 tokens
    assert r == [-1.0]


def test_linear_between():
    lp = make_length_penalty(soft_max=100, hard_max=200, min_reward=-1.0)
    r = lp([""], [_text_of_len(4 * 150)])  # ~150 tokens
    assert -0.6 < r[0] < -0.4


def test_message_shape():
    lp = make_length_penalty(soft_max=10, hard_max=20, min_reward=-1.0)
    msg = [{"role": "assistant", "content": _text_of_len(4 * 30)}]
    assert lp([""], [msg]) == [-1.0]

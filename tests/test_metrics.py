from doctrust.metrics import anls, is_abstention, normalize_answer


def test_normalization_ignores_case_and_punctuation() -> None:
    assert normalize_answer(" ₹4,250.00 ") == "425000"


def test_anls_accepts_formatting_variant() -> None:
    assert anls("4,250", ["4250"]) == 1.0


def test_abstention_parser() -> None:
    assert is_abstention("UNANSWERABLE")
    assert not is_abstention("4250")

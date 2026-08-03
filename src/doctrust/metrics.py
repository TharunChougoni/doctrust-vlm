from __future__ import annotations

import re
import unicodedata

ABSTAIN_ANSWERS = {
    "unanswerable",
    "cannot determine",
    "cannot be determined",
    "not readable",
    "unreadable",
    "not visible",
}


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def levenshtein_distance(left: str, right: str) -> int:
    """Memory-efficient Levenshtein edit distance."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def normalized_levenshtein_similarity(prediction: str, target: str) -> float:
    prediction = normalize_answer(prediction)
    target = normalize_answer(target)
    if not prediction and not target:
        return 1.0
    denominator = max(len(prediction), len(target), 1)
    return 1.0 - levenshtein_distance(prediction, target) / denominator


def anls(prediction: str, answers: list[str], threshold: float = 0.5) -> float:
    """DocVQA-style ANLS for one prediction and multiple accepted answers."""
    best = max(normalized_levenshtein_similarity(prediction, answer) for answer in answers)
    return best if best >= threshold else 0.0


def is_abstention(prediction: str) -> bool:
    normalized = normalize_answer(prediction)
    return normalized in {normalize_answer(answer) for answer in ABSTAIN_ANSWERS}

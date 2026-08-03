from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceExample:
    sample_id: str
    image_path: Path
    question: str
    answers: list[str]
    evidence_box: tuple[float, float, float, float]

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "SourceExample":
        required = {"id", "image_path", "question", "answers", "evidence_box"}
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Manifest row missing fields: {sorted(missing)}")

        answers = item["answers"]
        if not isinstance(answers, list) or not answers or not all(
            isinstance(answer, str) and answer.strip() for answer in answers
        ):
            raise ValueError("answers must be a non-empty list of strings")

        raw_box = item["evidence_box"]
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise ValueError("evidence_box must be [x1, y1, x2, y2]")
        box = (
            float(raw_box[0]),
            float(raw_box[1]),
            float(raw_box[2]),
            float(raw_box[3]),
        )
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("evidence_box coordinates must be normalized and ordered")

        return cls(
            sample_id=str(item["id"]),
            image_path=Path(item["image_path"]),
            question=str(item["question"]),
            answers=answers,
            evidence_box=box,
        )


@dataclass(frozen=True)
class PreparedExample:
    sample_id: str
    source_id: str
    image_path: Path
    question: str
    answers: list[str]
    variant: str
    expected_behavior: str

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "PreparedExample":
        required = {
            "id",
            "source_id",
            "image_path",
            "question",
            "answers",
            "variant",
            "expected_behavior",
        }
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Prepared row missing fields: {sorted(missing)}")
        behavior = str(item["expected_behavior"])
        if behavior not in {"preserve", "abstain", "update"}:
            raise ValueError(f"Unknown expected_behavior: {behavior}")
        return cls(
            sample_id=str(item["id"]),
            source_id=str(item["source_id"]),
            image_path=Path(item["image_path"]),
            question=str(item["question"]),
            answers=list(item["answers"]),
            variant=str(item["variant"]),
            expected_behavior=behavior,
        )

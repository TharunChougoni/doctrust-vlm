from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL while reporting the exact malformed line."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"JSONL file not found: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {source}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL atomically enough for a single-process experiment."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(target)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one cached prediction immediately after inference."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

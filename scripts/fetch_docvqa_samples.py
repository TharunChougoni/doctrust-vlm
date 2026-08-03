from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "nielsr/docvqa_1200_examples"
OFFICIAL_TERMS = "https://www.docvqa.org/datasets"
API = "https://datasets-server.huggingface.co/rows"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def fetch_rows(offset: int, length: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{API}?{params}", timeout=90) as response:
        return json.load(response)["rows"]


def answer_token_indices(row: dict[str, Any]) -> list[int]:
    """Map the annotated character offset back to OCR token indices."""
    answer = row["answer"]
    target_start = int(answer["start"])
    target_end = target_start + len(str(answer["matched_text"]))
    cursor = 0
    selected: list[int] = []
    for index, word in enumerate(row["words"]):
        token_start = cursor
        token_end = token_start + len(word)
        if token_end > target_start and token_start < target_end:
            selected.append(index)
        cursor = token_end + 1
    if not selected:
        raise ValueError("Annotated answer did not overlap an OCR token")
    selected_text = " ".join(row["words"][index] for index in selected)
    if normalize(selected_text) != normalize(str(answer["matched_text"])):
        raise ValueError(
            f"OCR span mismatch: {selected_text!r} != {answer['matched_text']!r}"
        )
    return selected


def evidence_box(row: dict[str, Any]) -> list[float]:
    """Union answer OCR boxes and return padded coordinates in [0, 1].

    This mirror stores OCR coordinates on a 0–1000 page coordinate system.
    Some OCR boxes have near-zero height, so a minimum vertical band is used.
    Every generated box must still be visually audited.
    """
    indices = answer_token_indices(row)
    boxes = [row["bounding_boxes"][index] for index in indices]
    x1 = min(min(float(box[0]), float(box[2])) for box in boxes) / 1000
    x2 = max(max(float(box[0]), float(box[2])) for box in boxes) / 1000
    y1 = min(min(float(box[1]), float(box[3])) for box in boxes) / 1000
    y2 = max(max(float(box[1]), float(box[3])) for box in boxes) / 1000

    x1, x2 = max(0.0, x1 - 0.008), min(1.0, x2 + 0.008)
    center_y = (y1 + y2) / 2
    half_height = max((y2 - y1) / 2 + 0.008, 0.013)
    y1, y2 = max(0.0, center_y - half_height), min(1.0, center_y + half_height)
    if x2 - x1 > 0.65 or y2 - y1 > 0.20:
        raise ValueError("Answer box is implausibly large")
    return [round(value, 6) for value in (x1, y1, x2, y2)]


def download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch unique DocVQA examples with OCR-derived answer evidence boxes."
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--scan", type=int, default=100)
    parser.add_argument(
        "--acknowledge-docvqa-terms",
        action="store_true",
        help=f"Confirm that you reviewed the original terms at {OFFICIAL_TERMS}",
    )
    args = parser.parse_args()
    if not args.acknowledge_docvqa_terms:
        parser.error(
            "Review the original DocVQA terms, then rerun with "
            "--acknowledge-docvqa-terms"
        )
    if not 1 <= args.count <= 50:
        parser.error("--count must be between 1 and 50")
    if args.scan < args.count:
        parser.error("--scan must be at least --count")

    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    offset = 0
    page_size = min(100, args.scan)
    while len(manifest_rows) < args.count and offset < args.scan:
        rows = fetch_rows(offset, min(page_size, args.scan - offset))
        if not rows:
            break
        for item in rows:
            row = item["row"]
            answer = row.get("answer") or {}
            question = str((row.get("query") or {}).get("en", "")).strip()
            answers = [str(value).strip() for value in row.get("answers", []) if str(value).strip()]
            if not question or not answers or float(answer.get("match_score", 0)) < 0.95:
                continue
            matched_text = str(answer.get("matched_text", "")).strip()
            if len(matched_text) > 50:
                continue
            # A bare count derived from a list is distributed evidence; covering
            # only the printed numeral would not truly destroy the evidence.
            if question.lower().startswith("how many") and normalize(matched_text).isdigit():
                continue
            try:
                box = evidence_box(row)
                image_bytes = download_bytes(row["image"]["src"])
            except (ValueError, KeyError, OSError, urllib.error.URLError):
                continue
            digest = hashlib.sha256(image_bytes).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            sample_id = f"docvqa-{row['id']}"
            image_path = raw_dir / f"{sample_id}.jpg"
            image_path.write_bytes(image_bytes)
            manifest_rows.append(
                {
                    "id": sample_id,
                    "image_path": str(image_path.relative_to(root)),
                    "question": question,
                    "answers": answers,
                    "evidence_box": box,
                }
            )
            provenance.append(
                {
                    "id": sample_id,
                    "dataset_id": row["id"],
                    "row_index": item["row_idx"],
                    "sha256": digest,
                    "matched_text": answer["matched_text"],
                    "match_score": answer["match_score"],
                    "evidence_box": box,
                }
            )
            print(f"Selected {sample_id}: {question} -> {answers[0]} @ {box}")
            if len(manifest_rows) >= args.count:
                break
        offset += len(rows)

    if len(manifest_rows) < args.count:
        raise RuntimeError(
            f"Found only {len(manifest_rows)} valid unique examples after scanning {offset} rows"
        )

    manifest_path = root / "data/manifests/source.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    provenance_path = root / "data/manifests/source_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "official_terms": OFFICIAL_TERMS,
                "split": "train",
                "selection": "unique image hash, OCR answer match >= 0.95",
                "warning": "OCR-derived boxes require visual audit before inference.",
                "samples": provenance,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nManifest: {manifest_path}")
    print(f"Provenance: {provenance_path}")
    print("Next step: visually audit every box in the Colab notebook before inference.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "nielsr/docvqa_1200_examples"
OFFICIAL_TERMS = "https://www.docvqa.org/datasets"
ROW_INDEX = 1
EVIDENCE_BOX = [0.235, 0.220, 0.315, 0.245]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch one DocVQA-derived sample for a local smoke test."
    )
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

    root = Path(__file__).resolve().parents[1]
    params = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": ROW_INDEX,
            "length": 1,
        }
    )
    api_url = f"https://datasets-server.huggingface.co/rows?{params}"
    with urllib.request.urlopen(api_url, timeout=60) as response:
        item = json.load(response)["rows"][0]
    row = item["row"]
    if row["id"] != "train_3" or row["answer"]["matched_text"] != "T.F. Riehl":
        raise RuntimeError("Dataset row changed; refusing to write mismatched annotations")

    image_path = root / "data/raw/docvqa-train-3.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(row["image"]["src"], image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()

    manifest = {
        "id": "docvqa-train-3",
        "image_path": "data/raw/docvqa-train-3.jpg",
        "question": "Who is in cc in this letter?",
        "answers": ["T.F. Riehl"],
        "evidence_box": EVIDENCE_BOX,
    }
    manifest_path = root / "data/manifests/source.jsonl"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    print(f"Downloaded: {image_path}")
    print(f"SHA-256: {digest}")
    print(f"Manifest: {manifest_path}")
    print("Next: open notebooks/doctrust_mvp.ipynb and verify the evidence-box cell.")


if __name__ == "__main__":
    main()

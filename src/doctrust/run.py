from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from doctrust.config import load_config
from doctrust.io import append_jsonl, read_jsonl
from doctrust.modeling import DocumentVLM
from doctrust.profiling import gpu_measurement
from doctrust.schema import PreparedExample


def run(config_path: str | Path) -> Path:
    """Run deterministic batch-one inference and cache every prediction."""
    config = load_config(config_path)
    prepared_path = Path(config["data"]["prepared_manifest"])
    output_path = Path(config["output"]["predictions"])
    rows = read_jsonl(prepared_path)
    examples = [PreparedExample.from_dict(row) for row in rows]
    if not examples:
        raise ValueError(f"No prepared examples found in {prepared_path}")

    completed_ids: set[str] = set()
    if output_path.exists():
        completed_ids = {str(row["id"]) for row in read_jsonl(output_path)}

    model = DocumentVLM(config["model"])
    for example in tqdm(examples, desc="Document VLM inference"):
        if example.sample_id in completed_ids:
            continue
        with gpu_measurement() as stats:
            prediction = model.answer(example.image_path, example.question)
        append_jsonl(
            output_path,
            {
                "id": example.sample_id,
                "source_id": example.source_id,
                "image_path": str(example.image_path),
                "question": example.question,
                "answers": example.answers,
                "variant": example.variant,
                "expected_behavior": example.expected_behavior,
                "prediction": prediction,
                **stats,
                "model_id": config["model"]["model_id"],
            },
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    output = run(args.config)
    print(f"Predictions: {output}")


if __name__ == "__main__":
    main()

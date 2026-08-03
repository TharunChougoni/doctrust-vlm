from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from doctrust.config import load_config
from doctrust.corruptions import apply_variant
from doctrust.io import read_jsonl, write_jsonl
from doctrust.schema import SourceExample


def prepare(config_path: str | Path) -> Path:
    """Create deterministic image variants and a model-ready manifest."""
    config = load_config(config_path)
    source_manifest = Path(config["data"]["source_manifest"])
    prepared_manifest = Path(config["data"]["prepared_manifest"])
    generated_dir = Path(config["data"]["generated_dir"])
    generated_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(source_manifest)
    if not source_rows:
        raise ValueError(f"No examples found in {source_manifest}")

    prepared_rows: list[dict] = []
    for raw_row in source_rows:
        example = SourceExample.from_dict(raw_row)
        if not example.image_path.exists():
            raise FileNotFoundError(
                f"Image for {example.sample_id} not found: {example.image_path}"
            )
        with Image.open(example.image_path) as opened:
            source_image = opened.convert("RGB")

        for variant in config["variants"]:
            variant_name = str(variant["name"])
            transformed = apply_variant(source_image, example.evidence_box, variant)
            output_path = generated_dir / f"{example.sample_id}__{variant_name}.png"
            transformed.save(output_path, format="PNG")
            prepared_rows.append(
                {
                    "id": f"{example.sample_id}__{variant_name}",
                    "source_id": example.sample_id,
                    "image_path": str(output_path),
                    "question": example.question,
                    "answers": example.answers,
                    "variant": variant_name,
                    "expected_behavior": variant["expected_behavior"],
                }
            )

    write_jsonl(prepared_manifest, prepared_rows)
    return prepared_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()
    manifest = prepare(args.config)
    print(f"Prepared manifest: {manifest}")


if __name__ == "__main__":
    main()

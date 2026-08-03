from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from doctrust.io import read_jsonl
from doctrust.metrics import anls, is_abstention


def evaluate(predictions_path: str | Path) -> dict:
    rows = read_jsonl(predictions_path)
    if not rows:
        raise ValueError(f"No predictions found in {predictions_path}")

    enriched_rows: list[dict] = []
    for row in rows:
        behavior = row["expected_behavior"]
        score = None
        abstained = is_abstention(str(row["prediction"]))
        if behavior in {"preserve", "update"}:
            score = anls(str(row["prediction"]), list(row["answers"]))
        enriched_rows.append({**row, "anls": score, "abstained": abstained})

    clean_scores = {
        str(row["source_id"]): float(row["anls"])
        for row in enriched_rows
        if row["variant"] == "clean" and row["anls"] is not None
    }
    clean_correct_ids = {
        source_id for source_id, score in clean_scores.items() if score >= 0.5
    }

    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in enriched_rows:
        source_id = str(row["source_id"])
        enriched = {
            **row,
            "clean_anls": clean_scores.get(source_id),
            "clean_correct": source_id in clean_correct_ids,
        }
        by_variant[str(row["variant"])].append(enriched)

    variant_metrics = {}
    for variant, variant_rows in sorted(by_variant.items()):
        preserve_scores = [row["anls"] for row in variant_rows if row["anls"] is not None]
        abstain_rows = [
            row for row in variant_rows if row["expected_behavior"] == "abstain"
        ]
        latencies = [
            float(row["latency_seconds"])
            for row in variant_rows
            if row.get("latency_seconds") is not None
        ]
        conditional_rows = [row for row in variant_rows if row["clean_correct"]]
        conditional_preserve_scores = [
            row["anls"] for row in conditional_rows if row["anls"] is not None
        ]
        conditional_abstain_rows = [
            row for row in conditional_rows if row["expected_behavior"] == "abstain"
        ]
        variant_metrics[variant] = {
            "count": len(variant_rows),
            "mean_anls": mean(preserve_scores) if preserve_scores else None,
            "abstention_rate": (
                mean([row["abstained"] for row in abstain_rows]) if abstain_rows else None
            ),
            "false_answer_rate": (
                mean([not row["abstained"] for row in abstain_rows]) if abstain_rows else None
            ),
            "mean_latency_seconds": mean(latencies) if latencies else None,
            "clean_correct_count": len(conditional_rows),
            "conditional_mean_anls": (
                mean(conditional_preserve_scores) if conditional_preserve_scores else None
            ),
            "conditional_abstention_rate": (
                mean([row["abstained"] for row in conditional_abstain_rows])
                if conditional_abstain_rows
                else None
            ),
            "conditional_false_answer_rate": (
                mean([not row["abstained"] for row in conditional_abstain_rows])
                if conditional_abstain_rows
                else None
            ),
        }

    source_ids = sorted({str(row["source_id"]) for row in rows})
    return {
        "prediction_count": len(rows),
        "source_count": len(source_ids),
        "clean_correct_threshold": 0.5,
        "clean_correct_source_count": len(clean_correct_ids),
        "clean_correct_source_ids": sorted(clean_correct_ids),
        "model_ids": sorted({str(row.get("model_id", "unknown")) for row in rows}),
        "variants": variant_metrics,
        "limitations": [
            "Synthetic occlusion does not prove real-world robustness.",
            "Small MVP samples should not be presented as benchmark-level evidence.",
            "Evidence boxes require manual audit.",
            "Conditional robustness uses only sources with clean ANLS >= 0.5.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = evaluate(args.predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

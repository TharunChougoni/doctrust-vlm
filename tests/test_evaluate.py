import json

from doctrust.evaluate import evaluate


def test_conditional_metrics_use_only_clean_correct_sources(tmp_path):
    rows = [
        {
            "id": "a-clean",
            "source_id": "a",
            "variant": "clean",
            "expected_behavior": "preserve",
            "prediction": "correct",
            "answers": ["correct"],
            "latency_seconds": 1.0,
            "model_id": "test-model",
        },
        {
            "id": "a-distractor",
            "source_id": "a",
            "variant": "distractor_occlusion",
            "expected_behavior": "preserve",
            "prediction": "correct",
            "answers": ["correct"],
            "latency_seconds": 1.0,
            "model_id": "test-model",
        },
        {
            "id": "a-evidence",
            "source_id": "a",
            "variant": "evidence_occlusion",
            "expected_behavior": "abstain",
            "prediction": "UNANSWERABLE",
            "answers": ["correct"],
            "latency_seconds": 1.0,
            "model_id": "test-model",
        },
        {
            "id": "b-clean",
            "source_id": "b",
            "variant": "clean",
            "expected_behavior": "preserve",
            "prediction": "wrong",
            "answers": ["correct"],
            "latency_seconds": 1.0,
            "model_id": "test-model",
        },
        {
            "id": "b-distractor",
            "source_id": "b",
            "variant": "distractor_occlusion",
            "expected_behavior": "preserve",
            "prediction": "wrong",
            "answers": ["correct"],
            "latency_seconds": 1.0,
            "model_id": "test-model",
        },
        {
            "id": "b-evidence",
            "source_id": "b",
            "variant": "evidence_occlusion",
            "expected_behavior": "abstain",
            "prediction": "another answer",
            "answers": ["correct"],
            "latency_seconds": 1.0,
            "model_id": "test-model",
        },
    ]
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))

    metrics = evaluate(predictions)

    assert metrics["source_count"] == 2
    assert metrics["clean_correct_source_count"] == 1
    assert metrics["clean_correct_source_ids"] == ["a"]
    assert metrics["variants"]["distractor_occlusion"]["conditional_mean_anls"] == 1.0
    evidence = metrics["variants"]["evidence_occlusion"]
    assert evidence["abstention_rate"] == 0.5
    assert evidence["conditional_abstention_rate"] == 1.0
    assert evidence["conditional_false_answer_rate"] == 0.0

from scripts.fetch_docvqa_samples import answer_token_indices, evidence_box


def test_character_span_maps_to_answer_tokens_and_box():
    row = {
        "words": ["CC:", "T.F.", "Riehl", "FROM:"],
        "bounding_boxes": [
            [100, 200, 150, 210],
            [200, 200, 250, 210],
            [255, 200, 320, 210],
            [330, 200, 400, 210],
        ],
        "answer": {"start": 4, "matched_text": "T.F. Riehl"},
    }

    assert answer_token_indices(row) == [1, 2]
    box = evidence_box(row)
    assert box[0] < 0.2
    assert box[2] > 0.32
    assert box[1] < 0.2
    assert box[3] > 0.21

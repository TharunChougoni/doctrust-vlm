# Data contract

Raw document images and generated manifests are intentionally excluded from Git. The original DocVQA terms still apply to images retrieved through the Hugging Face mirror.

## Reproducible multi-example subset

After reviewing the [official DocVQA terms](https://www.docvqa.org/datasets), fetch 10 examples with:

```bash
python scripts/fetch_docvqa_samples.py \
  --count 10 \
  --scan 150 \
  --acknowledge-docvqa-terms
```

The script scans `nielsr/docvqa_1200_examples` and selects deterministic examples that satisfy:

- a unique document-image SHA-256;
- an English question and accepted answer;
- OCR answer matching score of at least 0.95;
- a character-aligned OCR answer span;
- a plausible localized union box;
- no bare numeric count whose real evidence is distributed across a list.

It writes:

- `data/raw/docvqa-*.jpg`;
- `data/manifests/source.jsonl`;
- `data/manifests/source_provenance.json`.

## Where evidence boxes come from

You do **not** manually draw every box from scratch. This dataset mirror includes:

- OCR words;
- a 0–1000 bounding box for every OCR word;
- the matched answer text;
- the answer's character offset in the joined OCR text;
- an answer-match confidence score.

The fetcher maps the annotated character span back to OCR tokens, unions their boxes, normalizes the coordinates to `[0,1]`, and adds a small padding band. This is an automated proposal, not guaranteed ground truth. The Colab notebook displays every proposal with a red rectangle and blocks inference until the user marks the audit complete.

## What kinds of documents are included?

The selected subset is not limited to `CC:` fields. It contains varied scanned business documents such as letters, memoranda, reports, prose pages and tables. The deterministic first examples include names, dates, index values, measurements, abbreviations and short textual answers.

## Manifest fields

- `id`: unique QA identifier.
- `image_path`: repository-relative path to PNG/JPEG.
- `question`: document question.
- `answers`: list of accepted answer strings.
- `evidence_box`: normalized `[x1,y1,x2,y2]` box around the answer text.

All transformed variants of one source sample remain linked by `source_id`. Do not put variants from the same source document into different train/evaluation splits.

## Evaluation rule

Report clean performance first. Robustness metrics are additionally conditioned on examples for which the model achieves clean ANLS of at least 0.5. Otherwise, a model could appear “stable” merely by repeating the same wrong answer under every corruption.

## Licensing

Do not commit DocVQA images unless their terms explicitly permit redistribution. Commit code, configuration, provenance method and download instructions; keep downloaded images and generated manifests local or in the exported experiment artifact.

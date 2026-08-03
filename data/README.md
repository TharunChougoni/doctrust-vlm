# Data contract

Raw document images are intentionally excluded from Git.

## Steps

1. Put your images in `data/raw/`.
2. Copy `data/manifests/example.jsonl` to `data/manifests/source.jsonl`.
3. Create one line per question/image pair.
4. Audit every evidence box visually before using evidence/distractor occlusion.

## Manifest fields

- `id`: unique QA identifier.
- `image_path`: repository-relative path to PNG/JPEG.
- `question`: document question.
- `answers`: list of accepted answer strings.
- `evidence_box`: normalized `[x1,y1,x2,y2]` box around the answer text.

All transformed variants of one source sample remain linked by `source_id`. Do not put variants from the same source document into different train/evaluation splits.

## Licensing

Do not commit DocVQA/KIE-HVQA images unless their terms explicitly permit redistribution. Commit only IDs, manifests, checksums and download instructions.

# DocTrust-VLM

A compact, evidence-conditioned robustness study for document vision-language models.

## Research question

Can a small document VLM distinguish between:

1. **nuisance corruption** — the answer should remain unchanged;
2. **damage to irrelevant text** — the answer should remain unchanged;
3. **destruction of answer evidence** — the model should abstain instead of hallucinating?

The first milestone is an evaluation study, not a chatbot and not a claim of a new model. It uses paired transformations so that damage to an answer region can be compared with equally sized damage elsewhere on the same document.

## Status

The repository is scaffolded but **no model has been downloaded or executed and no experiment results exist yet**. Do not describe it as a completed project until `results/` contains reproducible outputs.

## First model

- Primary: [`ibm-granite/granite-vision-3.2-2b`](https://huggingface.co/ibm-granite/granite-vision-3.2-2b)
- Optional comparator: [`HuggingFaceTB/SmolVLM2-2.2B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)

Both are configured for 4-bit, batch-one inference. Granite is document-focused; SmolVLM2 is the lower-cost comparison.

## Repository map

```text
configs/                 model and experiment settings
data/manifests/          JSONL experiment manifests
docs/                    runbook, learning guide, and study design
notebooks/               one main, linear MVP notebook
scripts/                 environment checks (run manually)
src/doctrust/            corruption, inference, and evaluation code
tests/                    lightweight tests (not run during setup)
results/                  generated predictions and metrics (gitignored)
```

## Shared Python environment

This project intentionally lives inside the existing Hugging Face course folder and reuses:

```text
/home/tharun/Projects/hf-course/hf.venv
```

Activate it in Fish:

```fish
source /home/tharun/Projects/hf-course/hf.venv/bin/activate.fish
cd /home/tharun/Projects/hf-course/doctrust-vlm
```

Install dependencies only when you are ready:

```fish
python -m pip install -r requirements.txt
```

## Run order

### Recommended: one main notebook

Open [`notebooks/doctrust_mvp.ipynb`](notebooks/doctrust_mvp.ipynb) and run it from top to bottom. It explains each stage, previews the evidence box and transformed images, clearly marks the GPU/model-download boundary, performs inference, and computes metrics.

In VS Code, select this existing kernel:

```text
/home/tharun/Projects/hf-course/hf.venv/bin/python
```

For browser JupyterLab, install `requirements-notebook.txt`. See [`notebooks/README.md`](notebooks/README.md).

### Alternative: command line

Read [`docs/LEARNING_GUIDE.md`](docs/LEARNING_GUIDE.md), then follow [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

The equivalent commands are:

```fish
# 1. Check environment; does not download a model
python scripts/check_environment.py

# 2. Copy and edit the example manifest
cp data/manifests/example.jsonl data/manifests/source.jsonl

# 3. Generate clean/corrupted image variants
PYTHONPATH=src python -m doctrust.prepare --config configs/mvp.yaml

# 4. Run one model over the prepared manifest
PYTHONPATH=src python -m doctrust.run --config configs/mvp.yaml

# 5. Aggregate ANLS and abstention metrics
PYTHONPATH=src python -m doctrust.evaluate \
  --predictions results/predictions.jsonl \
  --output results/metrics.json
```

## Input manifest

One JSON object per line:

```json
{
  "id": "doc-001-q1",
  "image_path": "data/raw/doc-001.png",
  "question": "What is the invoice total?",
  "answers": ["4250", "4,250"],
  "evidence_box": [0.58, 0.72, 0.88, 0.81]
}
```

`evidence_box` is normalized `[x1, y1, x2, y2]`, where every coordinate is between 0 and 1. It should tightly surround the answer text—not the whole page.

## Honest scope

- Synthetic corruption is not the same as real deployment data.
- Evidence boxes must be audited; weak OCR matching is not human ground truth.
- A 20–50 item run is a proof of concept, not a statistically strong benchmark.
- No fine-tuning is included in the MVP.
- Qwen and PaddleOCR-VL are deferred until this pipeline is stable.

## License

Code is MIT-licensed. Model weights and source datasets retain their own licenses and must not be redistributed from this repository.

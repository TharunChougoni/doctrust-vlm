# DocTrust-VLM

A compact, evidence-conditioned robustness study for document vision-language models.

## Research question

Can a small document VLM distinguish between:

1. **nuisance corruption** — the answer should remain unchanged;
2. **damage to irrelevant text** — the answer should remain unchanged;
3. **destruction of answer evidence** — the model should abstain instead of hallucinating?

The first milestone is an evaluation study, not a chatbot and not a claim of a new model. It uses paired transformations so that damage to an answer region can be compared with equally sized damage elsewhere on the same document.

## Status

A one-source local smoke run completed with SmolVLM-500M. It fit comfortably, but its clean answer was incorrect, so that run is retained only as a weak baseline—not a robustness result. The main workflow now uses 10 OCR-audited DocVQA examples and SmolVLM 2.2B on a Colab GPU. No multi-example result is claimed until the Colab artifact files are produced.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TharunChougoni/doctrust-vlm/blob/main/notebooks/doctrust_colab.ipynb)

The repository is public, so the Colab notebook clones it directly without credentials or secrets.

## Models

- Local baseline: [`HuggingFaceTB/SmolVLM-500M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
- Colab experiment: [`HuggingFaceTB/SmolVLM-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) (2.2B, FP16)
- Deferred comparator: [`ibm-granite/granite-vision-3.2-2b`](https://huggingface.co/ibm-granite/granite-vision-3.2-2b)

The 500M checkpoint's official card reports roughly 1.23 GB GPU RAM, while the 2.2B card reports a 5.02 GB minimum. A 6 GB laptop GPU has insufficient practical headroom once the display, CUDA context and document activations are included; the 2.2B FP16 run therefore targets a Colab T4-class GPU.

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

### Recommended: Colab 2.2B + 10 examples

Open [`notebooks/doctrust_colab.ipynb`](notebooks/doctrust_colab.ipynb), choose a GPU runtime, and run it from top to bottom. It fetches 10 deterministic unique-image examples, displays every OCR-derived evidence box for manual approval, runs the paired variants, and exports a reproducibility ZIP.

### Local learning notebook

Open [`notebooks/doctrust_mvp.ipynb`](notebooks/doctrust_mvp.ipynb) for the one-source SmolVLM-500M workflow. It is useful for learning and pipeline debugging, not for the final robustness claim.

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

# 2. Fetch 10 unique DocVQA examples and derive OCR answer boxes
python scripts/fetch_docvqa_samples.py \
  --count 10 \
  --scan 150 \
  --acknowledge-docvqa-terms
# Visually audit every generated evidence box before continuing.

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

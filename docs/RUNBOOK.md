# Runbook: first local experiment

Follow this in order. Stop immediately on an error rather than stacking fixes.

## 0. Understand the boundary

Commands in this document may download multiple gigabytes and use the GPU. None were run during repository setup.

The first goal is **one correct end-to-end prediction**, not a large benchmark.

## 1. Enter the shared environment

```fish
source /home/tharun/Projects/hf-course/hf.venv/bin/activate.fish
cd /home/tharun/Projects/hf-course/doctrust-vlm
python scripts/check_environment.py
```

Expected:

- Python executable lives under `hf-course/hf.venv`;
- CUDA is visible through PyTorch;
- your GPU is an RTX 3060 Laptop GPU with 6 GB total VRAM.

If packages are missing:

```fish
python -m pip install -r requirements.txt
```

Do not install into system Python.

## 2. Prepare one real example

Place one PNG/JPEG at:

```text
data/raw/doc-001.png
```

Copy the manifest:

```fish
cp data/manifests/example.jsonl data/manifests/source.jsonl
```

Edit `source.jsonl` so that:

- `id` is unique;
- `image_path` points to your image;
- `question` has a visible short answer;
- `answers` includes acceptable spellings;
- `evidence_box` tightly surrounds the answer.

Normalized coordinate conversion:

```text
x_normalized = x_pixel / image_width
y_normalized = y_pixel / image_height
```

Inspect the box manually. A wrong box invalidates the evidence/distractor comparison.

## 3. Generate variants

```fish
PYTHONPATH=src python -m doctrust.prepare --config configs/mvp.yaml
```

Open every image under `data/generated/`.

Acceptance rules:

- clean is visually identical in content;
- JPEG and blur leave the answer human-readable;
- distractor occlusion does not cover the answer;
- evidence occlusion completely covers the answer region.

If those rules fail, edit `configs/mvp.yaml` or the box before inference.

## 4. Run the one-example smoke test

Close GPU-heavy applications first, then inspect free VRAM:

```fish
nvidia-smi
```

Run:

```fish
PYTHONPATH=src python -m doctrust.run --config configs/mvp.yaml
```

First execution downloads model weights. Predictions are appended immediately to `results/predictions.jsonl`, so an interrupted run can resume without repeating completed IDs.

If Granite runs out of memory:

1. close browser/video/GPU-heavy apps;
2. verify 4-bit is enabled in `configs/mvp.yaml`;
3. reduce to clean plus one corruption;
4. restart the Python process to clear VRAM;
5. only then try SmolVLM2 by changing the model block using `configs/models/smolvlm.yaml` as reference.

Do not begin QLoRA tonight.

## 5. Evaluate

```fish
PYTHONPATH=src python -m doctrust.evaluate \
  --predictions results/predictions.jsonl \
  --output results/metrics.json
```

Interpretation:

- high clean ANLS: the model can answer the base sample;
- preserved ANLS under JPEG/blur: nuisance robustness;
- preserved answer under distractor occlusion: locality;
- abstention after evidence occlusion: hallucination control;
- a confident answer after evidence removal: false-answer failure.

## 6. Expand only after the smoke test

Add 20–50 examples. Keep all variants from one source document together. Audit transformed images before launching inference.

Commit only:

- source IDs and manifests allowed by dataset terms;
- code/configuration;
- aggregate result JSON/CSV;
- selected non-sensitive failure images you are permitted to share.

Do not commit model weights, private data, tokens or restricted dataset images.

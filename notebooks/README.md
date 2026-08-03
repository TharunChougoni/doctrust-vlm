# Notebook workflows

Both notebooks orchestrate the same tested modules under `src/doctrust/`; they are interactive controllers, not duplicate implementations.

## Recommended: `doctrust_colab.ipynb`

Use this for the main two-model comparison:

- Colab T4-class GPU;
- `HuggingFaceTB/SmolVLM-Instruct` 2.2B and `Qwen/Qwen2.5-VL-3B-Instruct`, sequentially in FP16;
- 50 deterministic unique DocVQA document images;
- OCR-derived evidence boxes with a mandatory paginated visual-audit gate;
- five variants per source, producing 500 predictions total;
- raw predictions plus all-sample and clean-correct-conditional metrics;
- model revisions, GPU/package metadata and a downloadable reproducibility ZIP.

The GitHub repository is public, so Colab can open and clone it without a token or secret.

## Local learning workflow: `doctrust_mvp.ipynb`

Use this for understanding and debugging the pipeline locally. It uses SmolVLM-500M and the first DocVQA-derived source item. The local smoke result is a baseline only because the model missed the clean answer.

1. Open `/home/tharun/Projects/hf-course/doctrust-vlm`.
2. Open `notebooks/doctrust_mvp.ipynb`.
3. Select `/home/tharun/Projects/hf-course/hf.venv/bin/python` as the kernel.
4. Run cells one at a time.

## Browser JupyterLab

```fish
source /home/tharun/Projects/hf-course/hf.venv/bin/activate.fish
cd /home/tharun/Projects/hf-course/doctrust-vlm
python -m pip install -r requirements-notebook.txt
python -m jupyter lab
```

## Why code still lives in `src/`

Keeping corruptions, model loading and metrics in modules prevents notebook-state errors, makes experiments repeatable, and allows tests and CLI reuse. The notebooks explain and orchestrate those modules in a linear sequence.

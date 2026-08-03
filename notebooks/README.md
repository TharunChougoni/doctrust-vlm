# Notebook workflow

Use `doctrust_mvp.ipynb` for the first experiment. It runs the same modular code under `src/doctrust/`; the notebook is the interactive controller, not a second implementation.

## VS Code

A local DocVQA-derived smoke-test item is already configured. Its answer is `T.F. Riehl`; the notebook first visualizes the audited answer box before generating variants.

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

Keeping corruptions, model loading and metrics in modules prevents notebook-state errors, makes experiments repeatable, and allows tests/CLI reuse. The notebook explains and orchestrates those modules in one linear sequence.

No notebook cells were executed during setup.

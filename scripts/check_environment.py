from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def module_version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "not installed"
    module = __import__(name)
    return str(getattr(module, "__version__", "installed; version unavailable"))


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    expected_env = project.parent / "hf.venv"
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Expected shared environment: {expected_env}")
    print(f"Using expected environment: {expected_env in Path(sys.executable).parents}")
    for package in ("torch", "transformers", "accelerate", "bitsandbytes", "PIL", "yaml"):
        print(f"{package}: {module_version(package)}")

    if shutil.which("nvidia-smi"):
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
        )
    else:
        print("nvidia-smi: not found")

    print("\nThis check does not download or load a model.")


if __name__ == "__main__":
    main()

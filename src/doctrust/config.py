from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one YAML experiment configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    for required in ("model", "data", "variants", "output"):
        if required not in config:
            raise ValueError(f"Missing required configuration section: {required}")
    return config

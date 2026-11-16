"""Load and validate project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config at {config_path}")

    required = ("model", "data", "training", "inference")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"Config missing sections: {', '.join(missing)}")

    return cfg


def resolve_output_dir(cfg: dict[str, Any]) -> Path:
    out = Path(cfg["training"]["output_dir"])
    if not out.is_absolute():
        out = ROOT / out
    return out

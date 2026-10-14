"""Configuration loading and project paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODES = ("fast", "full")

_REQUIRED_KEYS = ("seed", "dataset", "columns", "split", "features", "training", "monitoring")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and sanity-check the YAML experiment configuration."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config {cfg_path} is missing keys: {missing}")
    overlap = set(cfg["columns"]["inputs"]) & set(cfg["columns"]["forbidden_features"])
    if overlap:
        raise ValueError(f"Inputs overlap forbidden features: {sorted(overlap)}")
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable short hash of a config, stored with every run for reproducibility."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def mode_dir(mode: str) -> Path:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return ARTIFACTS_DIR / mode

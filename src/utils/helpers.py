"""Shared utility functions for the project."""

import yaml
import logging
import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config relative to project root."""
    with open(ROOT / config_path) as f:
        return yaml.safe_load(f)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with a consistent format."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    return logging.getLogger(name)


def ensure_dirs(cfg: dict) -> None:
    """Create output directories declared in config if they don't exist."""
    for key in ("processed_data", "models", "reports", "figures"):
        path = ROOT / cfg["paths"][key]
        path.mkdir(parents=True, exist_ok=True)

"""Work Opportunity Radar — teaching demo for the CloudHER 'From Data to AI' talk.

Shared helpers live here so every module and notebook reads seeds and paths from the
same place (config.yaml) instead of hard-coding them.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

# Project root = the directory that contains config.yaml (parent of src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config.yaml as a plain dict."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path: str | os.PathLike) -> str:
    """Render a path relative to the project root for tidy, portable output.

    Keeps reports readable (``data/raw/collected_jobs.csv``) instead of dumping a
    machine-specific absolute path. Falls back to the given path if it is outside
    the project (e.g. a custom absolute location).
    """
    p = Path(path)
    try:
        return (p if p.is_absolute() else PROJECT_ROOT / p).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


def save_config(cfg: dict, path: str | os.PathLike | None = None) -> None:
    """Persist a config dict back to config.yaml (used by the notebook 'Customize me'
    cell when a candidate wants their choices to stick for the make/CLI path too)."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def seed_everything(seed: int) -> None:
    """Seed every source of randomness we use so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# file: src/addiction/__init__.py
"""
Top-level package for the Cigarette-and-Drinking-Data project.

Why: expose common utilities at a stable import path for concise usage in
scripts and notebooks (e.g., `from addiction import load_config, to_interim, get_logger`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

try:
    from importlib.metadata import PackageNotFoundError, version  # py3.8+
except Exception: 
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

def _resolve_version() -> str:
    # Why: keep runtime version in sync with pyproject name
    for pkg_name in ("cigarette-and-drinking-data", "addiction", "addiction-insights"):
        try:
            return version(pkg_name)
        except PackageNotFoundError:
            continue
    return "0.1.0"

__version__ = _resolve_version()

from .utilities.config import Config, ConfigPath, load_config
from .utilities.io import (
    read_csv,
    read_json,
    to_interim,
    save_model,
    load_model,
    save_figure,
    save_json,
    save_text,
    raw_csv_path,
)
from .utilities.logging import get_logger, set_verbosity, log_exceptions

def setup(
    config_path: str | Path = "config/config.yaml",
    *,
    logger_name: str = "addiction",
    to_file: bool = True,
):
    """
    Why: one-liner bootstrap for scripts.
    Returns (cfg, logger).
    """
    cfg = load_config(config_path)
    log = get_logger(logger_name, cfg=cfg, to_file=to_file)
    return cfg, log

__all__ = [
    "__version__",

    "Config",
    "ConfigPath",
    "load_config",

    "read_csv",
    "read_json",
    "to_interim",
    "save_model",
    "load_model",
    "save_figure",
    "save_json",
    "save_text",
    "raw_csv_path",

    "get_logger",
    "set_verbosity",
    "log_exceptions",

    "setup",
]

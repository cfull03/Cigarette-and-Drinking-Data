# File: addiction/__init__.py
"""Top-level package for the project.

Import-safe: no side effects. Exposes `config` and common path constants.
"""

from __future__ import annotations

from importlib import metadata as _metadata

# Version (fallback when running from source without installation)
try:
    __version__ = _metadata.version("addiction")
except _metadata.PackageNotFoundError:  # running from source tree
    __version__ = "0.0.0"

# Re-export config and common symbols for convenience
from . import config as config  # noqa: F401
from .config import (  # noqa: F401
    DATA_DIR,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    LOG_LEVEL,
    MODELS_DIR,
    PARQUET_COMPRESSION,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    WRITE_PARQUET,
    ensure_dirs,
)

__all__ = [
    "config",
    "PROJ_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "WRITE_PARQUET",
    "PARQUET_COMPRESSION",
    "LOG_LEVEL",
    "ensure_dirs",
    "__version__",
]

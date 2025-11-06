# filepath: addiction/__init__.py
"""
addiction package public API.

Exports:
- Paths & logger: PROJ_ROOT, DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
                  EXTERNAL_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, logger
- Setup helper: setup()
- Features: build_features(), FeatureSpec, FeatureRegistry, REGISTRY
- Preprocessor (sklearn): build_preprocessor(), fit_preprocessor(), transform_df(),
                          save_preprocessor(), load_preprocessor()
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Paths, logger
from .config import (  # noqa: F401
    DATA_DIR,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    logger,
)


# Setup helper (mkdir -p for standard dirs)
def setup(extra_dirs: Iterable[Path] | None = None) -> None:
    base_dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]
    if extra_dirs:
        base_dirs.extend(list(extra_dirs))
    for p in base_dirs:
        p.mkdir(parents=True, exist_ok=True)
    logger.success("Project directories verified/created.")

# Features API
from .features import (  # noqa: E402, F401
    REGISTRY,
    FeatureRegistry,
    FeatureSpec,
    build_features,
)

# Preprocessor API (scikit-learn)
from .preprocessor import (  # noqa: E402, F401
    build_preprocessor,
    fit_preprocessor,
    load_preprocessor,
    save_preprocessor,
    transform_df,
)

__all__ = [
    # paths & logger
    "PROJ_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "logger",
    # setup
    "setup",
    # features
    "build_features",
    "FeatureSpec",
    "FeatureRegistry",
    "REGISTRY",
    # preprocessor
    "build_preprocessor",
    "fit_preprocessor",
    "transform_df",
    "save_preprocessor",
    "load_preprocessor",
]

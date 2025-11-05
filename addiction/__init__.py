"""
addiction package public API.

Exposes:
- Paths & setup: PROJ_ROOT, DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
                 EXTERNAL_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, setup()
- Features & preprocessing: build_features(), FeatureSpec, build_preprocessor()
- (Optional) Modeling & metrics: build_model(), fit_pipeline(), predict(), compute_metrics()
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# --- Paths from config ---
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
    logger,  # keep logger configured once here
)


# --- Small helper to ensure the project directory tree exists ---
def setup(extra_dirs: Iterable[Path] | None = None) -> None:
    """Create the standard CCDS directory tree if missing."""
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
        Path(p).mkdir(parents=True, exist_ok=True)
    logger.success("Project directories verified/created.")

# --- Feature engineering ---
from .features import build_features  # your new function  # noqa: E402

# --- Preprocessor spec + builder ---
# If you created addiction/preprocess.py, import from there; otherwise fall back to features.py
try:  # preferred: lightweight scaler/encoder-only preprocessor
    from .preprocess import FeatureSpec, build_preprocessor  # type: ignore
except Exception:  # fallback to definitions living in features.py
    from .features import FeatureSpec, build_preprocessor  # type: ignore

# --- (Optional) modeling & metrics, if present ---
try:
    from .modeling import build_model, fit_pipeline, predict  # type: ignore
except Exception:
    # Make these names optional; avoids hard import failures
    def build_model(*args, **kwargs):  # type: ignore
        raise ImportError("addiction.modeling not available")

    def fit_pipeline(*args, **kwargs):  # type: ignore
        raise ImportError("addiction.modeling not available")

    def predict(*args, **kwargs):  # type: ignore
        raise ImportError("addiction.modeling not available")

try:
    from .metrics import compute_metrics  # type: ignore
except Exception:
    def compute_metrics(*args, **kwargs):  # type: ignore
        raise ImportError("addiction.metrics not available")

__all__ = [
    # paths + setup
    "PROJ_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "setup",
    # features + preprocessing
    "build_features",
    "FeatureSpec",
    "build_preprocessor",
    # (optional) modeling + metrics
    "build_model",
    "fit_pipeline",
    "predict",
    "compute_metrics",
]

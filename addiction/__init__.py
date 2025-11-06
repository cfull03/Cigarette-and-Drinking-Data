from __future__ import annotations

# Optional: try to expose package version if installed
try:
    from importlib.metadata import PackageNotFoundError, version  # Python 3.8+
except Exception:  # pragma: no cover
    version = None
    PackageNotFoundError = Exception  # type: ignore[misc]

try:  # pragma: no cover
    __version__ = version("addiction") if version else "0.0.0"
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# ---- Public API re-exports ----
from .dataset import (
    basic_cleanup,
    load_interim,
    load_raw,
    save_interim,
    train_test_split_safe,
)
from .features import (
    REGISTRY,
    FeatureError,
    FeatureRegistry,
    FeatureSpec,
    build_features,
)
from .preprocessor import (
    get_feature_names_after_preprocessor,
    infer_column_types,  # alias to infer_columns
    load_preprocessor,
    make_preprocessor,
    save_preprocessor,
)

__all__ = [
    "__version__",
    # dataset
    "load_raw",
    "basic_cleanup",
    "save_interim",
    "load_interim",
    "train_test_split_safe",
    # features
    "FeatureError",
    "FeatureSpec",
    "FeatureRegistry",
    "REGISTRY",
    "build_features",
    # preprocessor
    "infer_column_types",
    "make_preprocessor",
    "save_preprocessor",
    "load_preprocessor",
    "get_feature_names_after_preprocessor",
]

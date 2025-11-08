# filepath: addiction/__init__.py
from __future__ import annotations

# ---- Package version ----
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version
    try:
        __version__ = _pkg_version("addiction")
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "0.0.0"
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

# ---- Public API re-exports ----
# dataset
from .dataset import (
    basic_cleanup,
    load_interim,
    load_raw,
    save_interim,
    train_test_split_safe,
)

# eval
from .eval import (
    evaluate,
    load_metrics,
    save_metrics,
)

# features
from .features import (
    REGISTRY,
    FeatureError,
    FeatureRegistry,
    FeatureSpec,
    build_features,
)

# model
from .model import (
    build_model,
    load_model,
    save_model,
    train_model,
)

# preprocessor
from .preprocessor import (
    get_feature_names_after_preprocessor,
    infer_column_types,  # alias kept if implemented inside preprocessor
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
    # model
    "build_model",
    "train_model",
    "save_model",
    "load_model",
    # eval
    "evaluate",
    "save_metrics",
    "load_metrics",
    # decorators
    "enforce_dense_float64",
    "ensure_binary_labels",
]
# filepath: addiction/__init__.py
from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

# ---- Package version (no eager submodule imports) ----
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        __version__ = _pkg_version("addiction")
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "0.0.0"
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

# ---- Lazy public API ---------------------------------------------------------
# Map exported names to (module, attribute) without importing modules at import time.
EXPORTS: Dict[str, Tuple[str, str]] = {
    # dataset
    "basic_cleanup": ("addiction.dataset", "basic_cleanup"),
    "load_interim": ("addiction.dataset", "load_interim"),
    "load_raw": ("addiction.dataset", "load_raw"),
    "save_interim": ("addiction.dataset", "save_interim"),
    "train_test_split_safe": ("addiction.dataset", "train_test_split_safe"),
    # features
    "REGISTRY": ("addiction.features", "REGISTRY"),
    "FeatureError": ("addiction.features", "FeatureError"),
    "FeatureRegistry": ("addiction.features", "FeatureRegistry"),
    "FeatureSpec": ("addiction.features", "FeatureSpec"),
    "build_features": ("addiction.features", "build_features"),
    # preprocessor
    "infer_column_types": ("addiction.preprocessor", "infer_column_types"),
    "make_preprocessor": ("addiction.preprocessor", "make_preprocessor"),
    "save_preprocessor": ("addiction.preprocessor", "save_preprocessor"),
    "load_preprocessor": ("addiction.preprocessor", "load_preprocessor"),
    "get_feature_names_after_preprocessor": ("addiction.preprocessor", "get_feature_names_after_preprocessor"),
    # model core
    "build_model": ("addiction.model", "build_model"),
    "save_model": ("addiction.model", "save_model"),
    "load_model": ("addiction.model", "load_model"),
    # training (moved out of model.py)
    "train_model": ("addiction.modeling.train", "train_model"),
    # prediction helpers
    "predict_df": ("addiction.predict", "predict_df"),
    "predict_file": ("addiction.predict", "predict_file"),
    # eval (library helpers, not the CLI entrypoint) — keep only if implemented
    "evaluate": ("addiction.eval", "evaluate"),
    "save_metrics": ("addiction.eval", "save_metrics"),
    "load_metrics": ("addiction.eval", "load_metrics"),
}

__all__ = ["__version__", *EXPORTS.keys()]


def __getattr__(name: str) -> Any:
    """
    Lazily import and return the requested symbol the first time it is accessed.
    Avoids importing submodules during `import addiction`.
    """
    if name in EXPORTS:
        module_name, attr = EXPORTS[name]
        module = import_module(module_name)
        try:
            value = getattr(module, attr)
        except AttributeError as exc:  # pragma: no cover
            raise AttributeError(f"{module_name!r} has no attribute {attr!r}") from exc
        globals()[name] = value  # cache on module for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + list(EXPORTS.keys()))

# filepath: addiction/modeling/__init__.py
from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

# Public, lazily-resolved exports (no eager imports at package import time)
EXPORTS: Dict[str, Tuple[str, str]] = {
    # Train CLI
    "train_main": ("addiction.modeling.train", "main"),
    "train_app": ("addiction.modeling.train", "app"),
    # Predict CLI + helpers
    "predict_main": ("addiction.predict", "main"),
    "predict_app": ("addiction.predict", "app"),
    "predict_df": ("addiction.predict", "predict_df"),
    "predict_file": ("addiction.predict", "predict_file"),
}

__all__ = list(EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in EXPORTS:
        mod_name, attr = EXPORTS[name]
        module = import_module(mod_name)
        value = getattr(module, attr)
        globals()[name] = value  # cache after first access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)

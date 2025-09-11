# file: src/addiction_ds/io.py
"""Project IO helpers

Root-anchored, config-driven utilities for:
- CSV IO under `<repo>/data/...`
- Model IO under `<repo>/models/...` for **both scikit-learn and PyTorch** via
  exactly **one** `save_model(...)` and **one** `load_model(...)` API.

Exports
-------
- load_cfg(path="configs/default.yaml")
- get_paths(cfg)
- read_csv(path, **kwargs)
- ensure_dir(path)
- to_interim(df, cfg, stem, index=False)
- to_processed(df, cfg, stem, index=False)
- write_sample(df, cfg, n=100, index=False)
- save_model(obj, cfg, name, framework=None, subdir=None, **kw)
- load_model(cfg, name, framework=None, subdir=None, model=None, **kw)

Conventions
-----------
- `DATA_DIR` overrides `paths.data_dir` when set.
- `MODELS_DIR` overrides `paths.models_dir` when set.
- Relative paths in config resolve against the repository root.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# YAML is optional; operate presence-safe if unavailable
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

import pandas as pd

# scikit-learn persistence (preferred)
try:  # pragma: no cover - import gate
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore
    import pickle as _pickle

# PyTorch (optional)
try:  # pragma: no cover - import gate
    import torch as _torch  # type: ignore
except Exception:  # pragma: no cover
    _torch = None  # type: ignore

__all__ = [
    "load_cfg",
    "get_paths",
    "read_csv",
    "ensure_dir",
    "to_interim",
    "to_processed",
    "write_sample",
    "save_model",
    "load_model",
]


# -------------------------- Repo root resolution --------------------------

_REPO_MARKERS = {
    ".git",
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "default.yaml",
    "configs",
    "data",
}


def find_repo_root(start: str | Path | None = None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for candidate in (p, *p.parents):
        if any((candidate / m).exists() for m in _REPO_MARKERS):
            return candidate
    return Path(p.anchor)


# -------------------------- Config & Paths --------------------------


def _read_yaml(fp: Path) -> Mapping[str, Any]:
    if not fp.exists() or yaml is None:
        return {}
    try:
        with fp.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        from collections.abc import Mapping as _Mapping

        return data if isinstance(data, _Mapping) else {}
    except Exception:
        return {}


def load_cfg(
    path: str | Path = "configs/default.yaml", *, start: str | Path | None = None
) -> dict[str, Any]:
    root = find_repo_root(start)
    p = Path(path)
    cfg_path = p if p.is_absolute() else (root / p)
    if not cfg_path.exists():
        alt = root / "default.yaml"
        if alt.exists():
            cfg_path = alt
    return dict(_read_yaml(cfg_path))


@dataclass(frozen=True)
class Paths:
    raw: Path
    sample_input: Path
    sample_dir: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    figures_dir: Path
    schema: Path
    models_dir: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "raw": self.raw,
            "sample_input": self.sample_input,
            "sample_dir": self.sample_dir,
            "interim_dir": self.interim_dir,
            "processed_dir": self.processed_dir,
            "reports_dir": self.reports_dir,
            "figures_dir": self.figures_dir,
            "schema": self.schema,
            "models_dir": self.models_dir,
        }


_slug_re = re.compile(r"[^a-z0-9_.-/]")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _resolve_under(root: Path, candidate: str | Path) -> Path:
    p = Path(candidate)
    return p if p.is_absolute() else (root / p)


def _data_dir(root: Path, cfg: Mapping[str, Any]) -> Path:
    env = os.getenv("DATA_DIR")
    if env:
        return Path(env) if os.path.isabs(env) else (root / env)
    paths = (cfg or {}).get("paths") or {}
    return _resolve_under(root, paths.get("data_dir", "data"))


def _models_dir(root: Path, cfg: Mapping[str, Any]) -> Path:
    env = os.getenv("MODELS_DIR")
    if env:
        return Path(env) if os.path.isabs(env) else (root / env)
    paths = (cfg or {}).get("paths") or {}
    return _resolve_under(root, paths.get("models_dir", "models"))


def _paths_from_cfg(cfg: Mapping[str, Any], *, root: Path) -> Paths:
    paths = (cfg or {}).get("paths") or {}
    data_dir = _data_dir(root, cfg)
    return Paths(
        raw=_resolve_under(root, paths.get("raw", data_dir / "raw" / "data.csv")),
        sample_input=_resolve_under(
            root, paths.get("sample_input", data_dir / "sample" / "sample.csv")
        ),
        sample_dir=_resolve_under(root, paths.get("sample_dir", data_dir / "sample")),
        interim_dir=_resolve_under(root, paths.get("interim_dir", data_dir / "interim")),
        processed_dir=_resolve_under(root, paths.get("processed_dir", data_dir / "processed")),
        reports_dir=_resolve_under(root, paths.get("reports_dir", "reports")),
        figures_dir=_resolve_under(root, paths.get("figures_dir", "reports/figures")),
        schema=_resolve_under(root, paths.get("schema", "configs/schema.yaml")),
        models_dir=_models_dir(root, cfg),
    )


def get_paths(cfg: dict[str, Any], *, start: str | Path | None = None) -> dict[str, Path]:
    root = find_repo_root(start)
    return _paths_from_cfg(cfg, root=root).as_dict()


# --------------------------- CSV helpers ---------------------------


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    defaults: dict[str, Any] = dict(encoding="utf-8", low_memory=False)
    defaults.update(kwargs)
    return pd.read_csv(p, **defaults)


def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _save(df: pd.DataFrame, out: Path, *, index: bool = False, **kw: Any) -> Path:
    ensure_dir(out.parent)
    df.to_csv(out, index=index, **kw)
    return out


def _save_versioned(df: pd.DataFrame, out_dir: Path, stem: str, *, index: bool = False) -> Path:
    return _save(df, out_dir / f"{_ts()}_{stem}.csv", index=index)


def to_interim(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    stem: str,
    index: bool = False,
    start: str | Path | None = None,
) -> Path:
    P = get_paths(cfg, start=start)
    return _save_versioned(df, P["interim_dir"], stem, index=index)


def to_processed(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    stem: str,
    index: bool = False,
    start: str | Path | None = None,
) -> Path:
    P = get_paths(cfg, start=start)
    return _save_versioned(df, P["processed_dir"], stem, index=index)


def write_sample(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    n: int = 100,
    index: bool = False,
    start: str | Path | None = None,
) -> Path:
    P = get_paths(cfg, start=start)
    ensure_dir(P["sample_dir"])  # ensure sample dir exists
    rs = (cfg or {}).get("random_state", 42)
    out = P["sample_input"]
    ensure_dir(out.parent)
    return _save(df.sample(min(n, len(df)), random_state=rs), out, index=index)


# --------------------------- Unified Model IO ---------------------------


def _infer_framework(obj: Any) -> str:
    """Infer framework for *obj*; returns 'torch' or 'sklearn'."""
    if _torch is not None and hasattr(obj, "state_dict") and callable(obj.state_dict):
        # Don't require isinstance to avoid importing nn during cold envs
        return "torch"
    return "sklearn"


def save_model(
    obj: Any,
    cfg: dict[str, Any],
    *,
    name: str,
    framework: str | None = None,  # 'sklearn' | 'torch' | None (infer)
    subdir: str | None = None,
    compress: int | bool = 3,  # sklearn only
    metadata: dict[str, Any] | None = None,  # torch only
    start: str | Path | None = None,
) -> Path:
    """Save a model artifact under `<repo>/models`.

    - If `framework` is 'sklearn' → saves `<name>.joblib` (or `.pkl` fallback).
    - If `framework` is 'torch'   → saves `<name>.pt` containing a `state_dict` + metadata.
    - If `framework` is None      → inferred from `obj`.
    """
    fw = (framework or _infer_framework(obj)).lower()
    P = get_paths(cfg, start=start)
    base = P["models_dir"] / subdir if subdir else P["models_dir"]
    ensure_dir(base)

    if fw == "sklearn":
        path = base / f"{name}{'.joblib' if joblib is not None else '.pkl'}"
        if joblib is not None:
            joblib.dump(obj, path, compress=compress)
        else:  # pragma: no cover
            with open(path, "wb") as fh:
                _pickle.dump(obj, fh)
        return path

    if fw == "torch":
        if _torch is None:
            raise ImportError("PyTorch is not installed. Install `torch` to save torch models.")
        if not hasattr(obj, "state_dict"):
            raise TypeError(
                "Expected a torch.nn.Module or object with state_dict() for framework='torch'."
            )
        path = base / f"{name}.pt"
        payload: dict[str, Any] = {
            "state_dict": obj.state_dict(),
            "meta": {
                "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "torch_version": getattr(_torch, "__version__", "unknown"),
                "class": obj.__class__.__name__,
            },
        }
        if metadata:
            payload["meta"].update(metadata)
        _torch.save(payload, path)
        return path

    raise ValueError("framework must be one of {'sklearn','torch'} or None for auto-infer")


def load_model(
    cfg: dict[str, Any],
    *,
    name: str,
    framework: str | None = None,  # 'sklearn' | 'torch' | None (auto by file ext)
    subdir: str | None = None,
    model: Any | None = None,  # torch: provide nn.Module to load into; if None → return state_dict
    map_location: str | Any = "cpu",  # torch
    strict: bool = True,  # torch
    start: str | Path | None = None,
) -> Any:
    """Load a model artifact from `<repo>/models`.

    Behaviour:
      - sklearn: returns the estimator object.
      - torch: if `model` provided, loads into it and returns the model; otherwise returns the `state_dict`.

    Auto framework detection by extension if `framework=None`:
      `.joblib`/`.pkl` → sklearn; `.pt` → torch.
    """
    P = get_paths(cfg, start=start)
    base = P["models_dir"] / subdir if subdir else P["models_dir"]

    # Decide candidate paths and framework
    if framework is None:
        # Try common extensions in priority order
        candidates = [base / f"{name}.joblib", base / f"{name}.pkl", base / f"{name}.pt"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"No artifact found for '{name}' in {base} (tried .joblib/.pkl/.pt)"
            )
        suffix = path.suffix.lower()
        fw = "sklearn" if suffix in {".joblib", ".pkl"} else "torch"
    else:
        fw = framework.lower()
        path = base / (
            f"{name}.pt"
            if fw == "torch"
            else (f"{name}.joblib" if joblib is not None else f"{name}.pkl")
        )
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

    if fw == "sklearn":
        if path.suffix == ".joblib" and joblib is not None:
            return joblib.load(path)
        with open(path, "rb") as fh:
            return _pickle.load(fh)

    if fw == "torch":
        if _torch is None:
            raise ImportError("PyTorch is not installed. Install `torch` to load torch models.")
        payload = _torch.load(path, map_location=map_location)
        state = payload.get("state_dict", payload)
        if model is None:
            return state
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(state, strict=strict)
            return model
        raise TypeError(
            "Provided 'model' has no load_state_dict(...). Pass a torch.nn.Module instance."
        )

    raise ValueError(
        "framework must be one of {'sklearn','torch'} or None for auto-detect by file extension"
    )

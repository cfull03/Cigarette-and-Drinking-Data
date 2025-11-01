# file: src/addiction/utilities/io.py
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import pandas as pd

from .config import Config, load_config


# ---------------------------- helpers ----------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write(data: bytes, dst: Path) -> None:
    _ensure_dir(dst.parent)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dst.parent)) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, dst)


def _timestamped(name: str, with_ts: bool = True, fmt: str = "%Y%m%d-%H%M%S", ext: str = ".csv") -> str:
    stem = Path(name).stem or "data"
    suffix = Path(name).suffix or ext
    return f"{stem}{suffix}" if not with_ts else f"{stem}_{datetime.now().strftime(fmt)}{suffix}"


def _as_config(cfg_or_path: Union[Config, str, Path]) -> Config:
    return cfg_or_path if isinstance(cfg_or_path, Config) else load_config(cfg_or_path)


# ----------------------------- readers -----------------------------

def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def read_json(path: str | Path, encoding: str = "utf-8") -> Any:
    with Path(path).open("r", encoding=encoding) as f:
        return json.load(f)


# ----------------------- CSV writer (timestamped) -------------------

def to_interim(
    df: pd.DataFrame,
    name: str,
    cfg: Union[Config, str, Path] = "config/config.yaml",
    *,
    index: bool = False,
    timestamp: bool = True,
    timestamp_fmt: str = "%Y%m%d-%H%M%S",
    encoding: str = "utf-8",
    **to_csv_kwargs,
) -> Path:
    """
    Save CSV to interim_dir with optional timestamp. Returns destination path.
    """
    c = _as_config(cfg)
    c.paths.ensure_dirs()
    fname = _timestamped(name, with_ts=timestamp, fmt=timestamp_fmt, ext=".csv")
    dst = (c.paths.interim_dir / fname).resolve()
    payload = df.to_csv(index=index, **to_csv_kwargs).encode(encoding)
    _atomic_write(payload, dst)
    return dst


# ------------------------------- models ------------------------------

def save_model(
    model: Any,
    filename: str,
    cfg: Union[Config, str, Path] = "config/config.yaml",
    *,
    compress: int | tuple | str = 3,
) -> Path:
    c = _as_config(cfg)
    c.paths.ensure_dirs()
    dst = (c.paths.models_dir / filename).resolve()
    _ensure_dir(dst.parent)
    joblib.dump(model, dst, compress=compress)
    return dst


def load_model(
    filename_or_path: str | Path,
    cfg: Union[Config, str, Path] = "config/config.yaml",
) -> Any:
    p = Path(filename_or_path)
    if p.is_absolute():
        return joblib.load(p)
    c = _as_config(cfg)
    return joblib.load((c.paths.models_dir / p).resolve())


# ------------------------------- figures -----------------------------

def save_figure(
    fig,
    filename: str,
    cfg: Union[Config, str, Path] = "config/config.yaml",
    *,
    dpi: int = 150,
    bbox_inches: Optional[str] = "tight",
    facecolor: Optional[str] = None,
    transparent: bool = False,
) -> Path:
    c = _as_config(cfg)
    c.paths.ensure_dirs()
    dst = (c.paths.figures_dir / filename).resolve()
    _ensure_dir(dst.parent)
    fig.savefig(dst, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor, transparent=transparent)
    return dst


# --------------------------- JSON/TEXT writers -----------------------

def save_json(
    data: Any,
    filename: str,
    cfg: Union[Config, str, Path] = "config/config.yaml",
    *,
    subdir: str = "reports",
    encoding: str = "utf-8",
    indent: int = 2,
) -> Path:
    c = _as_config(cfg)
    base = {
        "reports": c.paths.reports_dir,
        "figures": c.paths.figures_dir,
        "models": c.paths.models_dir,
        "processed": c.paths.processed_dir,
        "interim": c.paths.interim_dir,
    }.get(subdir, c.paths.reports_dir)
    dst = (base / filename).resolve()
    payload = json.dumps(data, ensure_ascii=False, indent=indent).encode(encoding)
    _atomic_write(payload, dst)
    return dst


def save_text(
    text: str,
    filename: str,
    cfg: Union[Config, str, Path] = "config/config.yaml",
    *,
    subdir: str = "reports",
    encoding: str = "utf-8",
) -> Path:
    c = _as_config(cfg)
    base = {
        "reports": c.paths.reports_dir,
        "figures": c.paths.figures_dir,
        "models": c.paths.models_dir,
        "processed": c.paths.processed_dir,
        "interim": c.paths.interim_dir,
    }.get(subdir, c.paths.reports_dir)
    dst = (base / filename).resolve()
    _atomic_write(text.encode(encoding), dst)
    return dst


# ------------------------------ convenience --------------------------

def raw_csv_path(cfg: Union[Config, str, Path] = "config/config.yaml") -> Path:
    return _as_config(cfg).paths.raw_csv.resolve()

"""Project IO helpers.

Provides a thin, config-driven layer for reading/writing CSVs and resolving
repository paths. Designed to be *presence-safe* so scripts can run in CI or
fresh clones without fragile imports.

Exports:
- load_cfg(path)
- get_paths(cfg)
- read_csv(path, **kwargs)
- ensure_dir(path)
- to_interim(df, cfg, stem, index=False)
- to_processed(df, cfg, stem, index=False)
- write_sample(df, cfg, n=100, index=False)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

__all__ = [
    "load_cfg",
    "get_paths",
    "read_csv",
    "ensure_dir",
    "to_interim",
    "to_processed",
    "write_sample",
]


# -------------------------- Config & Paths --------------------------

def load_cfg(path: str | Path = "configs/default.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data or {}


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

    def as_dict(self) -> Dict[str, Path]:  # convenience for dict-style access
        return {
            "raw": self.raw,
            "sample_input": self.sample_input,
            "sample_dir": self.sample_dir,
            "interim_dir": self.interim_dir,
            "processed_dir": self.processed_dir,
            "reports_dir": self.reports_dir,
            "figures_dir": self.figures_dir,
            "schema": self.schema,
        }


def _paths_from_cfg(cfg: Dict[str, Any]) -> Paths:
    paths = (cfg or {}).get("paths") or {}
    data_dir = Path(paths.get("data_dir", "data"))
    return Paths(
        raw=Path(paths.get("raw", data_dir / "raw" / "data.csv")),
        sample_input=Path(paths.get("sample_input", data_dir / "sample" / "sample.csv")),
        sample_dir=Path(paths.get("sample_dir", data_dir / "sample")),
        interim_dir=Path(paths.get("interim_dir", data_dir / "interim")),
        processed_dir=Path(paths.get("processed_dir", data_dir / "processed")),
        reports_dir=Path(paths.get("reports_dir", "reports")),
        figures_dir=Path(paths.get("figures_dir", "reports/figures")),
        schema=Path(paths.get("schema", "configs/schema.yaml")),
    )


def get_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    return _paths_from_cfg(cfg).as_dict()


# --------------------------- IO primitives --------------------------

def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    # why: consistent dtype inference across OS/local/CI without surprises
    defaults = dict(encoding="utf-8", low_memory=False)
    defaults.update(kwargs)
    return pd.read_csv(p, **defaults)


def ensure_dir(path: str | Path) -> None:
    """Create directory if it doesn't exist (idempotent)."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _save(df: pd.DataFrame, out: Path, *, index: bool = False, **kw: Any) -> Path:
    ensure_dir(out.parent)
    df.to_csv(out, index=index, **kw)
    return out


def _save_versioned(df: pd.DataFrame, out_dir: Path, stem: str, *, index: bool = False) -> Path:
    return _save(df, out_dir / f"{_ts()}_{stem}.csv", index=index)


def to_interim(df: pd.DataFrame, cfg: Dict[str, Any], *, stem: str, index: bool = False) -> Path:
    P = get_paths(cfg)
    return _save_versioned(df, P["interim_dir"], stem, index=index)


def to_processed(df: pd.DataFrame, cfg: Dict[str, Any], *, stem: str, index: bool = False) -> Path:
    P = get_paths(cfg)
    return _save_versioned(df, P["processed_dir"], stem, index=index)


def write_sample(df: pd.DataFrame, cfg: Dict[str, Any], *, n: int = 100, index: bool = False) -> Path:
    P = get_paths(cfg)
    ensure_dir(P["sample_dir"])  # ensure sample dir exists
    # why: deterministic small sample for quick checks/CI
    rs = (cfg or {}).get("random_state", 42)
    out = P["sample_input"]
    ensure_dir(out.parent)
    return _save(df.sample(min(n, len(df)), random_state=rs), out, index=index)

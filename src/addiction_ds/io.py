from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Any
import pandas as pd, yaml

def load_cfg(path: str | Path = "configs/default.yaml") -> dict:
    p = Path(path)
    if not p.exists(): return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def _paths(cfg: dict) -> dict:
    paths = (cfg or {}).get("paths") or {}
    return {
        "raw": Path(paths.get("raw", "data/raw/data.csv")),
        "sample_input": Path(paths.get("sample_input", "data/sample/sample.csv")),
        "interim_dir": Path(paths.get("interim_dir", "data/interim")),
        "processed_dir": Path(paths.get("processed_dir", "data/processed")),
        "schema": Path(paths.get("schema", "configs/schema.yaml")),
    }

def get_paths(cfg: dict) -> dict: return _paths(cfg)
def read_csv(p: str | Path, **kw: Any) -> pd.DataFrame: return pd.read_csv(p, **kw)
def _ts() -> str: return datetime.now().strftime("%Y%m%d_%H%M%S")

def _save(df: pd.DataFrame, out: Path, index: bool=False, **kw: Any) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=index, **kw); return out

def _save_versioned(df: pd.DataFrame, out_dir: Path, stem: str, index: bool=False) -> Path:
    return _save(df, out_dir / f"{_ts()}_{stem}.csv", index=index)

def to_interim(df: pd.DataFrame, cfg: dict, stem: str, index: bool=False) -> Path:
    return _save_versioned(df, _paths(cfg)["interim_dir"], stem, index=index)

def to_processed(df: pd.DataFrame, cfg: dict, stem: str, index: bool=False) -> Path:
    return _save_versioned(df, _paths(cfg)["processed_dir"], stem, index=index)

def write_sample(df: pd.DataFrame, cfg: dict, n: int=100, index: bool=False) -> Path:
    p = _paths(cfg)["sample_input"]; p.parent.mkdir(parents=True, exist_ok=True)
    return _save(df.sample(min(n, len(df)), random_state=(cfg or {}).get("random_state", 42)), p, index=index)

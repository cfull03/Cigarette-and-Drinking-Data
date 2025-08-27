from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read CSV with pass-through kwargs."""
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False, **kwargs: Any) -> Path:
    """Write DataFrame; create parents first (why: reproducible I/O)."""
    p = Path(path)
    _ensure_parent(p)
    df.to_csv(p, index=index, **kwargs)
    return p


def save_versioned(
        df: pd.DataFrame,
        dir_path: str | Path,
        stem: str,
        *,
        index: bool = False,
        timestamp: str | None = None,
    ) -> Path:
    """Save with a timestamped filename inside dir_path (e.g., interim)."""
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M")
    path = d / f"{stem}_{timestamp}.csv"
    df.to_csv(path, index=index)
    return path


def promote(src: str | Path, dest: str | Path) -> Path:
    """Copy/overwrite dest with src (why: formalize WIP→processed promotion)."""
    src_p, dest_p = Path(src), Path(dest)
    _ensure_parent(dest_p)
    dest_p.write_bytes(src_p.read_bytes())
    return dest_p
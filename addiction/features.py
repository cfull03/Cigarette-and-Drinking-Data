# File: addiction/features/features.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer
import yaml

from addiction.config import PROCESSED_DATA_DIR  # project-level path constants

app = typer.Typer(help="Generate a simple ML feature matrix (no sklearn preprocessor).")


# --------------------------- config & io ---------------------------

def _load_config(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        logger.warning(f"Config not found: {cfg_path}; proceeding with CLI defaults.")
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _infer_cols(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    dt = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    cat = [c for c in df.columns if c not in num + dt]
    return num, cat, dt

def _apply_keep_drop(df: pd.DataFrame, keep: Optional[List[str]], drop: Optional[List[str]]) -> pd.DataFrame:
    cols = list(df.columns)
    if keep:
        missing = [c for c in keep if c not in df.columns]
        if missing:
            logger.warning(f"Keep columns not in frame: {missing}")
        cols = [c for c in keep if c in df.columns]
    if drop:
        cols = [c for c in cols if c not in drop]
    return df[cols]

def _label_encode_cats(df: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    mappings: Dict[str, Dict[str, int]] = {}
    for c in cat_cols:
        s = df[c].astype("category")
        codes = s.cat.codes.replace(-1, np.nan)
        df[f"{c}__lbl"] = codes  # keep original + code column
        mappings[c] = {str(cat): int(code) for code, cat in enumerate(s.cat.categories)}
    return df, mappings

def _extract_datetime_parts(df: pd.DataFrame, dt_cols: List[str]) -> pd.DataFrame:
    for c in dt_cols:
        df[f"{c}__year"] = df[c].dt.year
        df[f"{c}__month"] = df[c].dt.month
        df[f"{c}__quarter"] = df[c].dt.quarter
        # keep original dt column; safe for later encoders
    return df

def _drop_high_missing(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    to_drop = [c for c in df.columns if df[c].isna().mean() > threshold]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df, to_drop


# --------------------------- CLI ---------------------------

@app.command()
def main(
    input_path: Path = typer.Option(PROCESSED_DATA_DIR / "dataset.csv", help="Processed dataset (CSV/Parquet)."),
    output_path: Path = typer.Option(PROCESSED_DATA_DIR / "features.parquet", help="Output features path (.parquet or .csv)."),
    target: Optional[str] = typer.Option(None, help="Target column to separate (optional)."),
    keep: Optional[List[str]] = typer.Option(None, "--keep", help="Explicit columns to keep (repeatable)."),
    drop: Optional[List[str]] = typer.Option(None, "--drop", help="Columns to drop (repeatable)."),
    numeric_only: bool = typer.Option(False, help="Keep only numeric columns (after optional encodings)."),
    label_encode_cats: bool = typer.Option(False, help="Add <col>__lbl integer codes for categorical columns."),
    extract_date_parts: bool = typer.Option(True, help="Add year/month/quarter for datetime columns."),
    missing_drop_threshold: float = typer.Option(0.98, min=0.0, max=1.0, help="Drop cols with missing ratio > threshold."),
    config: Path = typer.Option(Path("config/config.yaml"), help="Config file (for parquet toggle)."),
    write_parquet: Optional[bool] = typer.Option(None, help="Override config.data.write_parquet."),
    parquet_compression: Optional[str] = typer.Option(None, help="Override config.data.parquet_compression."),
):
    """
    Build a lightweight feature matrix with simple, transparent transforms.
    """
    cfg = _load_config(config)
    data_cfg = cfg.get("data", {})
    write_parq = bool(data_cfg.get("write_parquet", True)) if write_parquet is None else bool(write_parquet)
    pq_codec = parquet_compression or data_cfg.get("parquet_compression", "snappy")

    assert input_path.exists(), f"Input not found: {input_path}"
    logger.info(f"Loading: {input_path}")
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)

    # Split target early to avoid accidental transforms
    y = None
    if target:
        if target not in df.columns:
            logger.warning(f"Target '{target}' not found; continuing without target.")
        else:
            y = df[target].copy()
            df = df.drop(columns=[target])

    # Column selection
    df = _apply_keep_drop(df, keep, drop)

    num_cols, cat_cols, dt_cols = _infer_cols(df)
    logger.info(f"Initial cols -> num: {len(num_cols)}, cat: {len(cat_cols)}, dt: {len(dt_cols)}")

    # Optional transforms (no sklearn)
    with tqdm(total=3, desc="Transform") as bar:
        if extract_date_parts and dt_cols:
            df = _extract_datetime_parts(df, dt_cols)
        bar.update(1)

        mappings: Dict[str, Dict[str, int]] = {}
        if label_encode_cats and cat_cols:
            df, mappings = _label_encode_cats(df, cat_cols)
        bar.update(1)

        # Drop high-missing columns (after new features exist)
        df, dropped = _drop_high_missing(df, missing_drop_threshold)
        if dropped:
            logger.info(f"Dropped {len(dropped)} high-missing columns (> {missing_drop_threshold:.2f}): {dropped[:6]}{' ...' if len(dropped)>6 else ''}")
        bar.update(1)

    # Optionally restrict to numeric only (useful before classic models)
    if numeric_only:
        df = df.select_dtypes(include=[np.number])

    # Save features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if write_parq or output_path.suffix.lower() == ".parquet":
        outp = output_path if output_path.suffix.lower() == ".parquet" else output_path.with_suffix(".parquet")
        df.to_parquet(outp, index=False, compression=pq_codec)
    else:
        outp = output_path if output_path.suffix.lower() == ".csv" else output_path.with_suffix(".csv")
        df.to_csv(outp, index=False)
    logger.success(f"Wrote features: {outp} (rows={df.shape[0]}, cols={df.shape[1]})")

    # Save target if present
    y_path = None
    if y is not None:
        y_path = outp.with_name(outp.stem + ".target.csv")
        pd.Series(y).to_csv(y_path, index=False, header=[target])
        logger.info(f"Wrote target: {y_path}")

    # Save simple metadata for reproducibility
    meta = {
        "input": str(input_path),
        "output": str(outp),
        "target": target,
        "target_path": str(y_path) if y_path else None,
        "options": {
            "numeric_only": numeric_only,
            "label_encode_cats": label_encode_cats,
            "extract_date_parts": extract_date_parts,
            "missing_drop_threshold": missing_drop_threshold,
        },
        "shapes": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    meta_path = outp.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if label_encode_cats:
        maps_path = outp.with_suffix(".label_maps.json")
        maps_path.write_text(json.dumps(mappings, indent=2), encoding="utf-8")
    logger.info(f"Saved meta: {meta_path}")

if __name__ == "__main__":
    app()

# File: addiction/data/dataset.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from addiction.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer(help="Process raw dataset -> engineered features.")

# ----------------------- Helpers (promoted from notebook) -----------------------

def parse_dates_best_effort(df: pd.DataFrame) -> pd.DataFrame:
    # Why: make time usable without hand-curated list
    date_like = [c for c in df.columns if any(k in str(c).lower() for k in ("date", "dt", "time", "year", "month"))]
    for c in date_like:
        try:
            df[c] = pd.to_datetime(df[c], errors="ignore", infer_datetime_format=True)
        except Exception:
            pass
    return df

def safe_series(x: pd.Series) -> pd.Series:
    return x.replace([np.inf, -np.inf], np.nan)

def winsorize_series(s: pd.Series, lower: float = 0.005, upper: float = 0.995) -> pd.Series:
    if s.notna().sum() < 10:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

def detect_population_col(df: pd.DataFrame) -> Optional[str]:
    cols = [c for c in df.columns if "pop" in str(c).lower() or "population" in str(c).lower()]
    if "population" in df.columns:
        return "population"
    return cols[0] if cols else None

def detect_count_cols(df: pd.DataFrame) -> List[str]:
    pats = ("count", "case", "cases", "death", "deaths", "event", "events")
    return [
        c for c in df.columns
        if any(p in str(c).lower() for p in pats) and pd.api.types.is_numeric_dtype(df[c])
    ]

def detect_gender_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    male = [c for c in df.columns if "male" in str(c).lower() or str(c).lower().endswith("_m")]
    female = [c for c in df.columns if "female" in str(c).lower() or str(c).lower().endswith("_f")]
    return male, female

def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], Optional[str]]:
    out = df.copy()
    created: list[str] = []

    # Per-capita rates
    pop_col = detect_population_col(out)
    if pop_col and pd.api.types.is_numeric_dtype(out[pop_col]):
        for c in detect_count_cols(out):
            rate = f"{c}_per_100k"
            with np.errstate(divide="ignore", invalid="ignore"):
                out[rate] = (out[c] / out[pop_col]) * 1e5
            out[rate] = safe_series(out[rate])
            created.append(rate)

    # Gender proportions
    male_cols, female_cols = detect_gender_cols(out)
    if male_cols and female_cols:
        mcol, fcol = male_cols[0], female_cols[0]
        total = f"{mcol}_plus_{fcol}_total"
        out[total] = safe_series(out[mcol]) + safe_series(out[fcol])
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"prop_{mcol}"] = safe_series(out[mcol]) / out[total]
            out[f"prop_{fcol}"] = safe_series(out[fcol]) / out[total]
        created += [total, f"prop_{mcol}", f"prop_{fcol}"]

    # Date parts
    date_cols = [c for c in out.columns if np.issubdtype(out[c].dtype, np.datetime64)]
    for c in date_cols:
        out[f"{c}_year"] = out[c].dt.year
        out[f"{c}_month"] = out[c].dt.month
        out[f"{c}_quarter"] = out[c].dt.quarter
        created += [f"{c}_year", f"{c}_month", f"{c}_quarter"]

    # Winsorized copies for numerics
    for c in out.select_dtypes(include=[np.number]).columns:
        wz = f"{c}_wz"
        out[wz] = winsorize_series(out[c])
        created.append(wz)

    return out, created, pop_col

# ------------------------------- CLI command -----------------------------------

@app.command()
def main(
    input_path: Path = typer.Option(RAW_DATA_DIR / "dataset.csv", help="Raw CSV path."),
    output_path: Path = typer.Option(PROCESSED_DATA_DIR / "dataset.features.csv", help="Processed CSV path."),
    write_parquet: bool = typer.Option(True, help="Also write a Parquet file."),
    parquet_compression: str = typer.Option("snappy", help="Parquet compression codec."),
    sep: str = typer.Option(",", help="CSV delimiter."),
    encoding: str = typer.Option("utf-8", help="CSV encoding."),
):
    """Process raw -> engineered features. EDA plots stay in the notebook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path.with_suffix(".parquet")

    logger.info(f"Loading raw data: {input_path}")
    assert input_path.exists(), f"Input not found: {input_path}"
    df = pd.read_csv(input_path, sep=sep, encoding=encoding, engine="c", low_memory=False)
    logger.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} cols")

    with tqdm(total=4, desc="Pipeline") as pbar:
        df = parse_dates_best_effort(df); pbar.update(1)
        df_feat, created_cols, pop_col = feature_engineering(df); pbar.update(1)

        logger.info(f"Created {len(created_cols)} features."
                    + (f" Population column: {pop_col}" if pop_col else ""))

        df_feat.to_csv(output_path, index=False); pbar.update(1)
        if write_parquet:
            try:
                df_feat.to_parquet(parquet_path, index=False, compression=parquet_compression)
                logger.info(f"Wrote Parquet: {parquet_path}")
            except Exception as e:
                logger.warning(f"Parquet write failed ({e}); continuing with CSV only.")
        pbar.update(1)

    logger.success(f"Processing complete. CSV: {output_path}")

if __name__ == "__main__":
    app()
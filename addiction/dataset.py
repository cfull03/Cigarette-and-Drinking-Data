# filepath: addiction/dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from loguru import logger
import numpy as np
import pandas as pd
import typer

from addiction.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer(add_completion=False)  # single-command CLI


# -------------------------
# Helpers (no CLI exposure)
# -------------------------
def _ensure_dirs(*paths: Path) -> None:
    """Create directories if missing. Why: mypy-safe Path-only API."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _standardize_strings(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip().str.lower().replace({"": pd.NA})
    return out


def _summarize_schema(df: pd.DataFrame) -> pd.DataFrame:
    schema = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "non_null": df.notnull().sum().values,
            "nulls": df.isnull().sum().values,
        }
    )
    schema["null_pct"] = (schema["nulls"] / len(df) * 100).round(2)
    return schema.sort_values(["nulls", "column"], ascending=[False, True]).reset_index(drop=True)


def _load_raw(default_name: str = "addiction_population_data.csv") -> pd.DataFrame:
    raw_path = RAW_DATA_DIR / default_name
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {raw_path}\n"
            "Place your raw CSV under data/raw/ or pass --input-path to the CLI."
        )
    logger.info(f"Loading raw data: {raw_path}")
    return pd.read_csv(raw_path)


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1) normalize column names
    out.columns = pd.Index(out.columns).str.strip().str.replace(r"\s+", "_", regex=True).str.lower()

    # 2) coerce numerics (tweak this list for your schema)
    numeric_candidates = [
        "annual_income_usd",
        "smokes_per_day",
        "drinks_per_week",
        "children_count",
        "attempts_to_quit_smoking",
        "attempts_to_quit_drinking",
        "sleep_hours",
        "bmi",
        "social_support",
        "smoking_rate_pct",
    ]
    out = _coerce_numeric(out, numeric_candidates)

    # 3) standardize common categoricals
    categorical_candidates = [
        "gender",
        "education_level",
        "employment_status",
        "marital_status",
        "mental_health_status",
        "exercise_frequency",
        "diet_quality",
        "state",
        "name",
    ]
    out = _standardize_strings(out, categorical_candidates)

    # 4) drop exact duplicates
    before = len(out)
    out = out.drop_duplicates()
    if before - len(out):
        logger.info(f"Dropped {before - len(out)} duplicate rows.")

    # 5) ensure an id exists
    if "id" not in out.columns:
        out.insert(0, "id", np.arange(1, len(out) + 1, dtype=int))

    return out


def _write_processed(df: pd.DataFrame, output_path: Path) -> Path:
    _ensure_dirs(output_path.parent)
    df.to_csv(output_path, index=False)
    logger.success(f"Wrote processed dataset → {output_path}")
    return output_path


# -------------------------
# Single CLI entrypoint
# -------------------------
@app.command()
def main(
    # Primary behavior: raw → processed
    input_path: Optional[Path] = typer.Option(
        None,
        help="Raw CSV path. If omitted, uses data/raw/addiction_population_data.csv",
    ),
    output_path: Path = typer.Option(
        INTERIM_DATA_DIR / "dataset.csv",
        help="Where to write the processed dataset",
    ),
    summarize_only: bool = typer.Option(
        False,
        help="If set, only print a schema summary of the input and exit",
    ),
) -> None:
    """
    Ingest and prepare dataset (raw → processed) with a single command.
    Matches Makefile 'data' target: `python addiction/dataset.py`
    """
    # Load
    if input_path is None:
        df_raw = _load_raw()
    else:
        logger.info(f"Loading raw data: {input_path}")
        df_raw = pd.read_csv(input_path)

    if summarize_only:
        schema = _summarize_schema(df_raw)
        pd.set_option("display.max_rows", 200)
        print(schema)
        raise typer.Exit(code=0)

    # Clean → write
    logger.info("Running basic_clean()…")
    df_proc = _basic_clean(df_raw)
    _write_processed(df_proc, output_path)


if __name__ == "__main__":
    app()
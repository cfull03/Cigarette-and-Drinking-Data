# File: addiction/plots/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

from loguru import logger
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

matplotlib.use("Agg")  # Why: headless-safe plotting
import matplotlib.pyplot as plt

from addiction.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer(help="Generate EDA plots from a processed dataset (matplotlib-only).")


# --------------------------- helpers (kept minimal) ---------------------------

def _load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)

def _infer_cols(df: pd.DataFrame, cat_max_unique: int) -> Tuple[List[str], List[str]]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    low_card_num = [c for c in num if df[c].nunique(dropna=True) <= cat_max_unique]
    cats = list(df.columns.difference(num)) + low_card_num
    seen = set()
    cats = [c for c in cats if (c not in seen) and (not seen.add(c))]
    num_true = [c for c in num if c not in low_card_num]
    return num_true, cats

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _safe_series(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan)

def _save_histograms(df: pd.DataFrame, cols: Iterable[str], outdir: Path, limit: int, bins: int) -> List[Path]:
    saved: List[Path] = []
    for col in list(cols)[:limit]:
        fig = plt.figure()
        _safe_series(df[col]).dropna().hist(bins=bins)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col); plt.ylabel("Frequency")
        out = outdir / f"hist_{col}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
    return saved

def _save_boxplots(df: pd.DataFrame, cols: Iterable[str], outdir: Path, limit: int) -> List[Path]:
    saved: List[Path] = []
    for col in list(cols)[:limit]:
        series = _safe_series(df[col]).dropna()
        if series.empty:
            continue
        fig = plt.figure()
        plt.boxplot(series, vert=True, labels=[col])
        plt.title(f"Boxplot: {col}")
        out = outdir / f"box_{col}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
    return saved

def _save_bars(df: pd.DataFrame, cats: Iterable[str], outdir: Path, limit: int, max_unique: int) -> List[Path]:
    saved: List[Path] = []
    small = [c for c in cats if df[c].nunique(dropna=True) <= max_unique][:limit]
    for col in small:
        counts = df[col].astype(str).fillna("NA").value_counts()
        fig = plt.figure()
        counts.plot(kind="bar")
        plt.title(f"Counts: {col}")
        plt.xlabel(col); plt.ylabel("Count")
        out = outdir / f"bar_{col}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
    return saved

def _save_corr(df: pd.DataFrame, num_cols: List[str], outdir: Path) -> List[Path]:
    if len(num_cols) < 2:
        return []
    corr = df[num_cols].corr(numeric_only=True)
    fig = plt.figure()
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title("Correlation (Pearson)")
    plt.colorbar()
    plt.tight_layout()
    out = outdir / "corr_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return [out]


# ------------------------------- CLI command -----------------------------------

@app.command()
def main(
    input_path: Path = typer.Option(PROCESSED_DATA_DIR / "dataset.features.csv", help="Processed dataset (.csv or .parquet)."),
    output_dir: Path = typer.Option(FIGURES_DIR / "eda", help="Directory to write figures into."),
    # plot toggles
    hist: bool = typer.Option(True, help="Generate histograms for numeric columns."),
    box: bool = typer.Option(True, help="Generate boxplots for numeric columns."),
    bars: bool = typer.Option(True, help="Generate bar charts for small-cardinality categoricals."),
    corr: bool = typer.Option(True, help="Generate Pearson correlation heatmap for numeric columns."),
    # limits / thresholds
    max_hist: int = typer.Option(12, min=1, help="Max numeric columns to plot histograms for."),
    max_box: int = typer.Option(10, min=1, help="Max numeric columns to boxplot."),
    max_bars: int = typer.Option(10, min=1, help="Max categorical columns to bar-plot."),
    cat_max_unique: int = typer.Option(20, min=2, help="Max unique values for a categorical column to be bar-plotted."),
    bins: int = typer.Option(30, min=5, help="Histogram bins."),
):
    """
    Generate standard EDA plots. Rules:
    - matplotlib only
    - one chart per figure
    - no explicit colors or styles
    """
    output_dir = _ensure_dir(output_dir)

    logger.info(f"Loading: {input_path}")
    df = _load_df(input_path)
    logger.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} cols")

    num_cols, cat_cols = _infer_cols(df, cat_max_unique=cat_max_unique)
    logger.info(f"Inferred columns -> numeric: {len(num_cols)}, categorical: {len(cat_cols)}")

    saved: List[Path] = []
    with tqdm(total=int(hist) + int(box) + int(bars) + int(corr), desc="Plotting") as pbar:
        if hist:
            saved += _save_histograms(df, num_cols, output_dir, limit=max_hist, bins=bins)
            pbar.update(1)
        if box:
            saved += _save_boxplots(df, num_cols, output_dir, limit=max_box)
            pbar.update(1)
        if bars:
            saved += _save_bars(df, cat_cols, output_dir, limit=max_bars, max_unique=cat_max_unique)
            pbar.update(1)
        if corr:
            saved += _save_corr(df, num_cols, output_dir)
            pbar.update(1)

    if saved:
        logger.success(f"Saved {len(saved)} figures to: {output_dir}")
        for p in sorted(saved):
            logger.info(p.as_posix())
    else:
        logger.warning("No figures were generated (check toggles or data types).")


if __name__ == "__main__":
    app()

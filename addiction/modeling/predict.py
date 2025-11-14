# filepath: addiction/modeling/predict.py
# [exp-001] - Contains methods modified/added in exp/001-smoking-trends-cf
from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from addiction.model import load_model

# optional; only used if a path is provided
try:
    from addiction.preprocessor import load_preprocessor, transform_df  # type: ignore
except Exception:  # pragma: no cover
    load_preprocessor = transform_df = None  # type: ignore

app = typer.Typer(add_completion=False, no_args_is_help=True)

__all__ = ["predict_df", "predict_file", "app"]


def _to_dense64(X: pd.DataFrame) -> np.ndarray:
    # why: sklearn estimators expect numeric arrays; ensure stable dtype
    arr = X.values
    if hasattr(arr, "toarray"):  # sparse
        arr = arr.toarray()
    return np.asarray(arr, dtype=np.float64)
    # [exp-001]


def _maybe_drop_target(df: pd.DataFrame, target: Optional[str]) -> pd.DataFrame:
    # why: avoid target leakage at inference
    if target and target in df.columns:
        return df.drop(columns=[target])
    return df
    # [exp-001]


def _maybe_apply_preprocessor(df: pd.DataFrame, preprocessor_path: Optional[Path]) -> pd.DataFrame:
    # why: enforce same encoding as training when available
    if preprocessor_path:
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        if load_preprocessor is None or transform_df is None:
            raise RuntimeError("Preprocessor module not available.")
        ct = load_preprocessor(preprocessor_path)
        return transform_df(df, ct)
    return df
    # [exp-001]


def predict_df(
    df_features: pd.DataFrame,
    model,
    *,
    preprocessor_path: Optional[Path] = None,
    target: Optional[str] = None,
    proba: bool = True,
) -> pd.DataFrame:
    """
    Public API: run inference on a DataFrame.
    Returns columns: id/row_id, pred, optional proba_1.
    """
    if df_features.empty:
        raise ValueError("Input features DataFrame is empty.")
    df_in = df_features.copy()

    # try keep an ID if available
    id_col = next((c for c in ("id", "ID", "row_id", "index") if c in df_in.columns), None)

    X_raw = _maybe_drop_target(df_in, target)
    X_proc = _maybe_apply_preprocessor(X_raw, preprocessor_path)
    if X_proc.shape[1] == 0:
        raise ValueError("No feature columns after preprocessing.")
    X = _to_dense64(X_proc)

    y_pred = model.predict(X)
    out = pd.DataFrame({"pred": y_pred})
    if proba and hasattr(model, "predict_proba"):
        out["proba_1"] = model.predict_proba(X)[:, 1]

    if id_col is not None:
        out.insert(0, id_col, df_in[id_col].values)
    else:
        out.insert(0, "row_id", np.arange(len(out), dtype=int))
    return out
    # [exp-001]


def predict_file(
    input_csv: Path,
    model_path: Path,
    *,
    output_csv: Path,
    preprocessor_path: Optional[Path] = None,
    target: Optional[str] = None,
    proba: bool = True,
) -> Path:
    """
    Public API: load CSV + model, run inference, write CSV, return output path.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {input_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    df = pd.read_csv(input_csv)
    model = load_model(model_path)

    logger.info("Running inference…")
    _ = [None for _ in tqdm(range(1), total=1)]  # keeps parity with logs
    out = predict_df(df, model, preprocessor_path=preprocessor_path, target=target, proba=proba)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    logger.success(f"Wrote predictions → {output_csv} ({out.shape[0]} rows)")

    # brief preview in logs
    with pd.option_context("display.max_rows", 5, "display.width", 120):
        logger.info(f"\n{out.head()}")
    return output_csv
    # [exp-001]


@app.command(name="main")
def main(
    input_csv: Path = typer.Option(
        Path("data/processed/features.csv"),
        help="CSV containing feature columns (may include target; it will be dropped).",
    ),
    model_path: Path = typer.Option(
        Path("artifacts/rf/model.joblib"),
        help="Path to trained model artifact.",
    ),
    output_csv: Path = typer.Option(
        Path("artifacts/rf/predictions.csv"),
        help="Where to write predictions CSV.",
    ),
    preprocessor_path: Optional[Path] = typer.Option(
        None,
        help="Optional path to fitted preprocessor.joblib used during training.",
    ),
    target: Optional[str] = typer.Option(
        None,
        help="Target column name if present in features CSV (will be dropped).",
    ),
    proba: bool = typer.Option(
        True,
        help="Also output positive-class probability if supported.",
    ),
) -> None:
    """CLI: run predictions and write CSV (Makefile-compatible)."""
    predict_file(
        input_csv=input_csv,
        model_path=model_path,
        output_csv=output_csv,
        preprocessor_path=preprocessor_path,
        target=target,
        proba=proba,
    )
    # [exp-001]


if __name__ == "__main__":
    app()

# filepath: addiction/predict.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from addiction.config import MODELS_DIR, PROCESSED_DATA_DIR
from addiction.model import load_model  # reuse trained model loader

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _to_dense64(X: pd.DataFrame) -> np.ndarray:
    # why: sklearn estimators expect numeric arrays; ensure stable dtype
    arr = X.values
    if hasattr(arr, "toarray"):  # sparse
        arr = arr.toarray()
    return np.asarray(arr, dtype=np.float64)


def _maybe_drop_target(df: pd.DataFrame, target: Optional[str]) -> pd.DataFrame:
    # why: avoid target leakage at inference
    if target and target in df.columns:
        return df.drop(columns=[target])
    return df


def _maybe_apply_preprocessor(df: pd.DataFrame, preprocessor_path: Optional[Path]) -> pd.DataFrame:
    # why: enforce same encoding as training when available
    if preprocessor_path:
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        from addiction.preprocessor import load_preprocessor, transform_df
        ct = load_preprocessor(preprocessor_path)
        return transform_df(df, ct)
    return df


@app.command(name="main")
def main(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "test_features.csv",
        help="CSV containing feature columns (may include target; it will be dropped).",
    ),
    model_path: Path = typer.Option(
        MODELS_DIR / "model.joblib",
        help="Path to trained model artifact.",
    ),
    predictions_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "test_predictions.csv",
        help="Where to write predictions CSV.",
    ),
    preprocessor_path: Optional[Path] = typer.Option(
        None,
        help="Optional path to fitted preprocessor.joblib used during training.",
    ),
    target: Optional[str] = typer.Option(
        "has_health_issues",
        help="Target column name if present in features CSV (will be dropped).",
    ),
    proba: bool = typer.Option(
        True,
        help="Also output positive-class probability if supported.",
    ),
) -> None:
    if not features_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading features: {features_path}")
    df_in = pd.read_csv(features_path)
    # Keep an ID if present for traceability
    id_col = None
    for cand in ("id", "ID", "row_id", "index"):
        if cand in df_in.columns:
            id_col = cand
            break

    X_raw = _maybe_drop_target(df_in, target)
    X_proc = _maybe_apply_preprocessor(X_raw, preprocessor_path)
    X = _to_dense64(X_proc)

    logger.info(f"Loaded model: {model_path}")
    model = load_model(model_path)

    logger.info("Running inference…")
    # tqdm mainly for parity with your logs; single-shot prediction is fast
    _ = [None for _ in tqdm(range(1), total=1)]
    y_pred = model.predict(X)
    out = pd.DataFrame({"pred": y_pred})

    if proba and hasattr(model, "predict_proba"):
        proba_1 = model.predict_proba(X)[:, 1]
        out["proba_1"] = proba_1

    if id_col is not None:
        out.insert(0, id_col, df_in[id_col].values)
    else:
        out.insert(0, "row_id", np.arange(len(out), dtype=int))

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(predictions_path, index=False)
    logger.success(f"Wrote predictions → {predictions_path} ({out.shape[0]} rows)")

    # brief preview in logs
    with pd.option_context("display.max_rows", 5, "display.width", 120):
        logger.info(f"\n{out.head()}")


if __name__ == "__main__":
    app()

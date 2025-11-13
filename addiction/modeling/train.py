# filepath: addiction/modeling/train.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

from loguru import logger
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import metrics as skm
import typer

from addiction.dataset import train_test_split_safe
from addiction.model import build_model, save_model

# optional; only used if a path is provided
try:
    from addiction.preprocessor import load_preprocessor, transform_df  # type: ignore
except Exception:  # pragma: no cover
    load_preprocessor = transform_df = None  # type: ignore

app = typer.Typer(add_completion=False, no_args_is_help=True)

__all__ = ["train_model", "app"]


def _assert_no_target_in_X(df: pd.DataFrame, target: str) -> None:
    # why: prevents leakage & mismatched ColumnTransformer columns
    if target in df.columns:
        raise ValueError(f"Target '{target}' unexpectedly present in feature matrix.")


def _to_dense64(X: pd.DataFrame | np.ndarray) -> npt.NDArray[np.float64]:
    from scipy import sparse as _sp  # local import to avoid hard dep at import time
    arr = X.toarray() if _sp.issparse(X) else np.asarray(X)
    return cast(npt.NDArray[np.float64], arr.astype(np.float64, copy=False))


def _to01(y: pd.Series | np.ndarray) -> npt.NDArray[np.int_]:
    arr = np.asarray(y)
    if arr.dtype == bool:
        return cast(npt.NDArray[np.int_], arr.astype(np.int_, copy=False))
    u = np.unique(arr)
    if set(u.tolist()) <= {0, 1}:
        return cast(npt.NDArray[np.int_], arr.astype(np.int_, copy=False))
    raise ValueError(f"Labels must be boolean or 0/1. Got uniques={u}")


def _apply_preprocessor_if_any(
    Xtr_raw: pd.DataFrame, Xte_raw: pd.DataFrame, preprocessor_path: Optional[Path]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # why: enforce identical encoding as used at training-time
    if preprocessor_path:
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        if load_preprocessor is None or transform_df is None:
            raise RuntimeError("Preprocessor module not available.")
        ct = load_preprocessor(preprocessor_path)
        Xtr = transform_df(Xtr_raw, ct)
        Xte = transform_df(Xte_raw, ct)
        return Xtr, Xte
    return Xtr_raw, Xte_raw


def _compute_metrics(
    y_true: npt.NDArray[np.int_],
    y_pred: npt.NDArray[np.int_],
    proba_pos: Optional[npt.NDArray[np.float64]] = None,
) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "precision": float(skm.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(skm.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(skm.f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "support": {"n": int(y_true.size), "pos_rate": float(y_true.mean())},
    }
    if proba_pos is not None and np.ndim(proba_pos) == 1:
        try:
            m["roc_auc"] = float(skm.roc_auc_score(y_true, proba_pos))
            m["log_loss"] = float(skm.log_loss(y_true, np.vstack([1 - proba_pos, proba_pos]).T))
        except Exception as e:
            logger.warning(f"Skipping probability-based metrics: {e}")
    return m


def train_model(
    df: pd.DataFrame,
    *,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    n_estimators: int = 500,
    max_depth: Optional[int] = 5,
    max_features: str = "sqrt",
    min_samples_leaf: int = 1,
    class_weight: Optional[str] = "balanced",
    n_jobs: int = -1,
    preprocessor_path: Optional[Path] = None,
) -> Tuple[
    Any,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    Dict[str, Any],
]:
    """
    Split → (optional) preprocess → fit RF via build_model → compute metrics.
    Returns: (clf, Xtr, Xte, ytr, yte, metrics)
    """
    if df.empty:
        raise ValueError("Training DataFrame is empty.")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Available: {list(df.columns)}")

    Xtr_raw, Xte_raw, ytr_raw, yte_raw = train_test_split_safe(
        df, target=target, test_size=test_size, random_state=random_state, stratify=stratify
    )
    _assert_no_target_in_X(Xtr_raw, target)
    _assert_no_target_in_X(Xte_raw, target)

    Xtr_df, Xte_df = _apply_preprocessor_if_any(Xtr_raw, Xte_raw, preprocessor_path)

    Xtr = _to_dense64(Xtr_df.values)
    Xte = _to_dense64(Xte_df.values)
    ytr = _to01(ytr_raw)
    yte = _to01(yte_raw)

    clf = build_model(
        Xtr,
        ytr,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    y_pred = clf.predict(Xte)
    proba_pos = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else None
    metrics = _compute_metrics(yte, y_pred, proba_pos)
    logger.info(f"Train pos rate={float(ytr.mean()):.4f} | Test pos rate={float(yte.mean()):.4f}")
    return clf, Xtr, Xte, ytr, yte, metrics


@app.command("main")
def main(
    input_csv: Path = typer.Option(
        Path("data/processed/features.csv"),
        help="CSV with features (+ target).",
        exists=True,
        readable=True,
    ),
    target: str = typer.Option(..., help="Target column name."),
    output_model: Path = typer.Option(
        Path("artifacts/rf/model.joblib"),
        help="Where to save the trained model.",
    ),
    output_metrics: Path = typer.Option(
        Path("artifacts/rf/metrics.json"),
        help="Where to write evaluation metrics JSON.",
    ),
    # split
    test_size: float = typer.Option(0.2, help="Test size fraction.", min=0.05, max=0.95),
    random_state: int = typer.Option(42, help="Random seed."),
    stratify: bool = typer.Option(True, help="Stratify by target when splitting."),
    # RF hyperparams
    n_estimators: int = typer.Option(500),
    max_depth: Optional[int] = typer.Option(5),
    max_features: str = typer.Option("sqrt"),
    min_samples_leaf: int = typer.Option(1, min=1),
    class_weight: Optional[str] = typer.Option("balanced"),
    n_jobs: int = typer.Option(-1),
    # preprocessing
    preprocessor_path: Optional[Path] = typer.Option(
        None,
        help="Fitted ColumnTransformer (fit on X only). If provided, apply to X.",
    ),
) -> None:
    """
    Train a RandomForest on tabular data, optionally using a saved preprocessor.
    Saves: model (.joblib) and metrics (.json).
    """
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded data: {input_csv} shape={df.shape}")

    clf, *_rest, metrics = train_model(
        df,
        target=target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        preprocessor_path=preprocessor_path,
    )

    output_model.parent.mkdir(parents=True, exist_ok=True)
    save_model(clf, output_model)

    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics, dtype="object").to_json(output_metrics, indent=2)

    logger.success(f"Saved model → {output_model}")
    logger.success(f"Wrote metrics → {output_metrics}")
    logger.info(f"Summary metrics: {metrics}")


if __name__ == "__main__":
    app()

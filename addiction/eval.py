# filepath: addiction/eval.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, cast

from loguru import logger
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import typer

from addiction.config import INTERIM_DATA_DIR
from addiction.dataset import train_test_split_safe
from addiction.model import load_model

app = typer.Typer(add_completion=False)

__all__ = ["evaluate", "save_metrics", "load_metrics"]

# ---------- Helpers ----------
def _to01(y: Any) -> npt.NDArray[np.int_]:
    ya = np.asarray(y)
    if ya.dtype == bool:
        return cast(npt.NDArray[np.int_], ya.astype(np.int_, copy=False))
    u = np.unique(ya)
    if set(u.tolist()) <= {0, 1}:
        return cast(npt.NDArray[np.int_], ya.astype(np.int_, copy=False))
    raise ValueError(f"Labels must be boolean or 0/1. Got uniques={u}")

def _to_dense(X: Any) -> npt.NDArray[np.float64]:
    if sparse.issparse(X):
        arr = X.toarray().astype(np.float64, copy=False)
        return cast(npt.NDArray[np.float64], arr)
    arr = np.asarray(X, dtype=np.float64)
    return cast(npt.NDArray[np.float64], arr)

# ---------- Public API ----------
def evaluate(model: Any, X_test: Any, y_test: Any) -> Dict[str, object]:
    """
    Compute ROC-AUC, Accuracy, Precision, Recall, F1, and Confusion Matrix.
    """
    Xd: npt.NDArray[np.float64] = _to_dense(X_test)
    y01: npt.NDArray[np.int_] = _to01(y_test)

    pred: npt.NDArray[np.int_] = cast(
        npt.NDArray[np.int_],
        np.asarray(model.predict(Xd), dtype=np.int_)
    )

    scores: Optional[npt.NDArray[np.float64]] = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(Xd)[:, 1]
        scores = cast(npt.NDArray[np.float64], np.asarray(p, dtype=np.float64))
    elif hasattr(model, "decision_function"):
        m = model.decision_function(Xd)
        scores = cast(npt.NDArray[np.float64], np.asarray(m, dtype=np.float64))

    auc: Optional[float] = float(roc_auc_score(y01, scores)) if scores is not None else None
    acc = float(accuracy_score(y01, pred))
    prec = float(precision_score(y01, pred, zero_division=0))
    rec = float(recall_score(y01, pred, zero_division=0))
    f1 = float(f1_score(y01, pred, zero_division=0))
    cm = confusion_matrix(y01, pred).tolist()

    return {
        "roc_auc": auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "n_samples": int(y01.shape[0]),
    }

def save_metrics(metrics: Dict[str, object], path: Path | str) -> Path:
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.success(f"Saved metrics → {out}")
    return out

def load_metrics(path: Path | str) -> Dict[str, object]:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Metrics file not found: {p}")
    with p.open() as f:
        data: Dict[str, object] = json.load(f)
    logger.info(f"Loaded metrics from {p}")
    return data

# ---------- CLI ----------
@app.command()
def eval(
    model_path: Path = typer.Option(Path("artifacts/rf/model.joblib"), help="Trained model path."),
    target: str = typer.Option(..., help="Target column in interim CSV."),
    input_csv: Path = typer.Option(INTERIM_DATA_DIR / "dataset.csv", help="Interim dataset CSV path."),
    test_size: float = typer.Option(0.2, help="Test size for split."),
    random_state: int = typer.Option(42, help="Random seed."),
    output_metrics: Path = typer.Option(Path("artifacts/rf/metrics.json"), help="Metrics JSON output."),
) -> None:
    """Load model + dataset, evaluate on TEST, save metrics JSON."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Interim CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded interim CSV: {input_csv} ({df.shape})")

    _, Xte_raw, _, yte = train_test_split_safe(
        df, target=target, test_size=test_size, random_state=random_state, stratify=True
    )
    Xte_mat: npt.NDArray[np.float64] = _to_dense(Xte_raw.values)

    model = load_model(model_path)
    metrics = evaluate(model, Xte_mat, yte)
    save_metrics(metrics, output_metrics)

    logger.info(f"Metrics: {metrics}")
    logger.success("Done.")

if __name__ == "__main__":
    app()
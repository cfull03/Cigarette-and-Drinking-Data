# filepath: addiction/eval.py
# [exp-001] - Contains methods modified/added in exp/001-smoking-trends-cf
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

from loguru import logger
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import typer

from addiction.config import PROCESSED_DATA_DIR
from addiction.dataset import train_test_split_safe
from addiction.model import load_model
from addiction.preprocessor import (  # why: encode categoricals consistently
    load_preprocessor,
    transform_df,
)

# optional import; only used to improve column names if available
try:
    from addiction.preprocessor import get_feature_names_after_preprocessor  # type: ignore
except Exception:  # pragma: no cover
    get_feature_names_after_preprocessor = None  # type: ignore

app = typer.Typer(add_completion=False, no_args_is_help=True)

__all__ = ["evaluate", "save_metrics", "load_metrics", "main"]

# ---------- helpers ----------
def _to_dense64(X: Any) -> npt.NDArray[np.float64]:
    from scipy import sparse as _sp
    arr: np.ndarray = X.toarray() if _sp.issparse(X) else np.asarray(X)
    arr64: np.ndarray = arr.astype(np.float64, copy=False)
    return cast(npt.NDArray[np.float64], arr64)
    # [exp-001]

def _to01(y: Any) -> npt.NDArray[np.int_]:
    arr: np.ndarray = np.asarray(y)
    if arr.dtype == bool:
        out: np.ndarray = arr.astype(np.int_, copy=False)
        return cast(npt.NDArray[np.int_], out)
    u = np.unique(arr)
    if set(u.tolist()) <= {0, 1}:
        out = arr.astype(np.int_, copy=False)
        return cast(npt.NDArray[np.int_], out)
    raise ValueError(f"Labels must be boolean or 0/1. Got uniques={u}")
    # [exp-001]

def _assert_has_target(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        cols_preview = ", ".join(map(str, list(df.columns[:12])))
        raise KeyError(
            f"Target column '{target}' not found in input CSV. "
            f"Eval expects the FEATURES file (with target), not a preprocessed X-only file.\n"
            f"Columns preview: [{cols_preview}{'...' if df.shape[1] > 12 else ''}]"
        )
    # [exp-001]

def _pos_class_index(model: Any) -> int:
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        return classes.index(1) if 1 in classes else (len(classes) - 1)
    return 1
    # [exp-001]

def _extract_feature_names(
    X_enc: Any, *, preprocessor_path: Optional[Path]
) -> Tuple[pd.Index, bool]:
    """
    Try to recover feature names from the transformed X. Returns (index, reliable_names?).
    """
    # If already a DataFrame with columns, use them
    if hasattr(X_enc, "columns"):
        cols = cast(pd.DataFrame, X_enc).columns
        return cols, True

    # If we have a preprocessor and helper to get names, try that
    if preprocessor_path and get_feature_names_after_preprocessor:
        try:
            ct = load_preprocessor(preprocessor_path)
            names = get_feature_names_after_preprocessor(ct)
            return pd.Index(names), True
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not recover feature names from preprocessor: {e}")

    # Fallback to numeric index
    n_cols = X_enc.shape[1] if hasattr(X_enc, "shape") else 0
    return pd.RangeIndex(n_cols), False

def _extract_feature_importances(
    model: Any, X_enc: Any, *, preprocessor_path: Optional[Path]
) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame with columns [feature, importance] if supported; else None.
    Supports tree-based models exposing `feature_importances_`.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None

    names, reliable = _extract_feature_names(X_enc, preprocessor_path=preprocessor_path)
    if len(importances) != len(names):
        # why: mismatched length indicates unknown column mapping; still write with RangeIndex
        logger.warning(
            f"feature_importances_ length ({len(importances)}) != feature names length ({len(names)}); "
            "falling back to RangeIndex."
        )
        names = pd.RangeIndex(len(importances))
        reliable = False

    df = pd.DataFrame({"feature": names.astype(str), "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    if not reliable:
        logger.warning("Feature names are inferred; verify mapping if you rely on exact names.")
    return df

# ---------- public api ----------
def evaluate(model: Any, X_test: Any, y_test: Any) -> Dict[str, object]:
    Xd: npt.NDArray[np.float64] = _to_dense64(X_test)
    y01: npt.NDArray[np.int_] = _to01(y_test)

    pred_arr: np.ndarray = np.asarray(model.predict(Xd), dtype=np.int_)
    pred: npt.NDArray[np.int_] = cast(npt.NDArray[np.int_], pred_arr)

    scores: Optional[npt.NDArray[np.float64]] = None
    if hasattr(model, "predict_proba"):
        idx = _pos_class_index(model)
        proba1: np.ndarray = model.predict_proba(Xd)[:, idx]
        scores_arr: np.ndarray = np.asarray(proba1, dtype=np.float64)
        scores = cast(npt.NDArray[np.float64], scores_arr)
    elif hasattr(model, "decision_function"):
        margin: np.ndarray = model.decision_function(Xd)
        scores_arr = np.asarray(margin, dtype=np.float64)
        scores = cast(npt.NDArray[np.float64], scores_arr)

    auc: Optional[float] = None
    try:
        if scores is not None:
            auc = float(roc_auc_score(y01, scores))
    except Exception as e:
        logger.warning(f"ROC AUC unavailable: {e}")

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
    # [exp-001]

def save_metrics(metrics: Dict[str, object], path: Path | str) -> Path:
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.success(f"Saved metrics → {out}")
    return out
    # [exp-001]

def load_metrics(path: Path | str) -> Dict[str, object]:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Metrics file not found: {p}")
    with p.open() as f:
        data: Dict[str, object] = json.load(f)
    logger.info(f"Loaded metrics from {p}")
    return data
    # [exp-001]

# ---------- single subcommand named "main" ----------
@app.command()
def main(
    model_path: Path = typer.Option(Path("artifacts/rf/model.joblib"), help="Trained model path."),
    target: str = typer.Option(..., help="Target column (must exist in CSV)."),
    input_csv: Path = typer.Option(
        PROCESSED_DATA_DIR / "features.csv",
        help="CSV including target column (features output).",
    ),
    test_size: float = typer.Option(0.2, help="Test size split."),
    random_state: int = typer.Option(42, help="Random seed."),
    output_metrics: Path = typer.Option(Path("artifacts/rf/metrics.json"), help="Metrics JSON output path."),
    output_importances: Optional[Path] = typer.Option(
        Path("artifacts/rf/feature_importances.csv"),
        help="Optional CSV with feature importances (if supported). Set empty to skip.",
    ),
    preprocessor_path: Optional[Path] = typer.Option(
        None,
        help="Path to fitted preprocessor.joblib. If provided, X is transformed before scoring.",
    ),
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded CSV for eval: {input_csv} ({df.shape})")
    _assert_has_target(df, target)

    # Hold-out split on the features CSV so we evaluate on unseen rows
    _, Xte_raw, _, yte_raw = train_test_split_safe(
        df, target=target, test_size=test_size, random_state=random_state, stratify=True
    )

    # Optional: apply the exact same preprocessor used at train time
    if preprocessor_path is not None:
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        ct = load_preprocessor(preprocessor_path)
        Xte_enc = transform_df(Xte_raw, ct)  # why: stringify/categorical columns → numeric features
    else:
        Xte_enc = Xte_raw  # assumes all-numeric

    Xte_mat: npt.NDArray[np.float64] = _to_dense64(
        Xte_enc.values if hasattr(Xte_enc, "values") else Xte_enc
    )
    yte: npt.NDArray[np.int_] = _to01(yte_raw)

    model = load_model(model_path)
    metrics = evaluate(model, Xte_mat, yte)
    save_metrics(metrics, output_metrics)

    # Optional: write feature importances if available
    if output_importances:
        try:
            fi_df = _extract_feature_importances(
                model,
                Xte_enc if hasattr(Xte_enc, "shape") else Xte_mat,
                preprocessor_path=preprocessor_path,
            )
            if fi_df is not None:
                output_importances.parent.mkdir(parents=True, exist_ok=True)
                fi_df.to_csv(output_importances, index=False)
                logger.success(
                    f"Wrote feature importances → {output_importances} (top5: {fi_df.head(5).to_dict('records')})"
                )
            else:
                logger.warning("Model does not expose feature_importances_; skipping CSV.")
        except Exception as e:
            logger.warning(f"Skipping feature importances due to error: {e}")

    logger.success("Done.")
    # [exp-001]

if __name__ == "__main__":
    app()

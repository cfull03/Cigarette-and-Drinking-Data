# --- filepath: addiction/model.py (add target guard; rest of file unchanged) ---
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, cast

from joblib import dump, load
from loguru import logger
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import typer

from addiction.config import INTERIM_DATA_DIR
from addiction.dataset import train_test_split_safe

app = typer.Typer(add_completion=False, no_args_is_help=True)

__all__ = ["build_model", "train_model", "save_model", "load_model"]

def _to_dense64(X: Any) -> npt.NDArray[np.float64]:
    from scipy import sparse as _sp
    arr: np.ndarray = X.toarray() if _sp.issparse(X) else np.asarray(X)
    arr64: np.ndarray = arr.astype(np.float64, copy=False)
    return cast(npt.NDArray[np.float64], arr64)

def _to01(y: Any) -> npt.NDArray[np.int_]:
    arr: np.ndarray = np.asarray(y)
    if arr.dtype == bool:
        return cast(npt.NDArray[np.int_], arr.astype(np.int_, copy=False))
    u = np.unique(arr)
    if set(u.tolist()) <= {0, 1}:
        return cast(npt.NDArray[np.int_], arr.astype(np.int_, copy=False))
    raise ValueError(f"Labels must be boolean or 0/1. Got uniques={u}")

def _apply_optional_preprocessor(
    Xtr_raw: pd.DataFrame,
    Xte_raw: pd.DataFrame,
    preprocessor_path: Optional[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if preprocessor_path and preprocessor_path.exists():
        from addiction.preprocessor import load_preprocessor, transform_df
        ct = load_preprocessor(preprocessor_path)
        Xtr = transform_df(Xtr_raw, ct)  # why: ensure identical encoding as fit-time
        Xte = transform_df(Xte_raw, ct)
        return Xtr, Xte
    return Xtr_raw, Xte_raw

def _assert_no_target_in_X(X: pd.DataFrame, target: str) -> None:
    # why: prevents leakage & ColumnTransformer column-mismatch
    if target in X.columns:
        raise ValueError(f"Target '{target}' unexpectedly present in X columns before preprocessing.")

def build_model(
    X_train: Any,
    y_train: Any,
    *,
    n_estimators: int = 500,
    max_depth: Optional[int] = 5,
    max_features: str = "sqrt",
    min_samples_leaf: int = 1,
    class_weight: Optional[str] = "balanced",
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomForestClassifier:
    Xd: npt.NDArray[np.float64] = _to_dense64(X_train)
    y01: npt.NDArray[np.int_] = _to01(y_train)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    logger.info("Fitting RandomForestClassifier…")
    clf.fit(Xd, y01)
    logger.success("Model trained.")
    return clf

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
) -> Tuple[RandomForestClassifier, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    Xtr_raw, Xte_raw, ytr_raw, yte_raw = train_test_split_safe(
        df, target=target, test_size=test_size, random_state=random_state, stratify=stratify
    )
    _assert_no_target_in_X(Xtr_raw, target)
    _assert_no_target_in_X(Xte_raw, target)

    # Apply preprocessor ONLY to X (after target was split out)
    Xtr_df, Xte_df = _apply_optional_preprocessor(Xtr_raw, Xte_raw, preprocessor_path)

    Xtr_mat: npt.NDArray[np.float64] = _to_dense64(Xtr_df.values)
    Xte_mat: npt.NDArray[np.float64] = _to_dense64(Xte_df.values)
    ytr: npt.NDArray[np.int_] = _to01(ytr_raw)
    yte: npt.NDArray[np.int_] = _to01(yte_raw)

    clf = build_model(
        Xtr_mat, ytr,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    logger.info(f"Train pos rate: {float(ytr.mean()):.4f} | Test pos rate: {float(yte.mean()):.4f}")
    return clf, Xtr_mat, Xte_mat, ytr, yte

def save_model(model: RandomForestClassifier, path: Path | str) -> Path:
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out)
    logger.success(f"Saved model → {out}")
    return out

def load_model(path: Path | str) -> RandomForestClassifier:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    logger.info(f"Loading model: {p}")
    model: RandomForestClassifier = load(p)
    return model

@app.command(name="main")
def main(
    target: str = typer.Option(..., help="Target column (must exist in the CSV)."),
    input_csv: Path = typer.Option(INTERIM_DATA_DIR / "dataset.csv", help="CSV including target column."),
    output_model: Path = typer.Option(Path("artifacts/rf/model.joblib"), help="Trained model output path."),
    test_size: float = typer.Option(0.2, help="Test size split."),
    random_state: int = typer.Option(42, help="Random seed."),
    n_estimators: int = typer.Option(500),
    max_depth: Optional[int] = typer.Option(5),
    max_features: str = typer.Option("sqrt"),
    min_samples_leaf: int = typer.Option(1),
    class_weight: Optional[str] = typer.Option("balanced"),
    n_jobs: int = typer.Option(-1),
    preprocessor_path: Optional[Path] = typer.Option(None, help="Path to fitted preprocessor.joblib (fit on X only)."),
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded CSV: {input_csv} ({df.shape})")

    model, *_ = train_model(
        df,
        target=target,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        preprocessor_path=preprocessor_path,
    )
    save_model(model, output_model)
    logger.success("Done.")

if __name__ == "__main__":
    app()

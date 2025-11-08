# filepath: addiction/model.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, cast

from joblib import dump, load
from loguru import logger
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
import typer

from addiction.config import INTERIM_DATA_DIR
from addiction.dataset import train_test_split_safe

app = typer.Typer(add_completion=False)

__all__ = ["build_model", "train_model", "save_model", "load_model"]


# -------------------------
# Helpers (no CLI exposure)
# -------------------------
def _to01(y: Any) -> np.ndarray:
    ya = np.asarray(y)
    if ya.dtype == bool:
        return ya.astype(int)
    u = np.unique(ya)
    if set(u.tolist()) <= {0, 1}:
        return ya.astype(int)
    raise ValueError(f"Labels must be boolean or 0/1. Got uniques={u}")

def _to_dense(X: Any) -> npt.NDArray[np.generic]:
    if sparse.issparse(X):
        return cast(npt.NDArray[np.generic], X.toarray())
    return cast(npt.NDArray[np.generic], np.asarray(X))


# -------------------------
# Public API
# -------------------------
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
    """
    Fit a RandomForestClassifier using project defaults.
    """
    Xd = _to_dense(X_train)
    y01 = _to01(y_train)

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
) -> Tuple[RandomForestClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split with train_test_split_safe, fit RF on TRAIN, return (model, Xtr, Xte, ytr, yte).
    """
    Xtr_raw, Xte_raw, ytr, yte = train_test_split_safe(
        df, target=target, test_size=test_size, random_state=random_state, stratify=stratify
    )
    Xtr_mat = _to_dense(Xtr_raw.values)
    Xte_mat = _to_dense(Xte_raw.values)
    clf = build_model(
        Xtr_mat,
        ytr,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    ytr01 = _to01(ytr)
    yte01 = _to01(yte)
    logger.info(f"Train pos rate: {ytr01.mean():.4f} | Test pos rate: {yte01.mean():.4f}")
    return clf, Xtr_mat, Xte_mat, ytr01, yte01


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


# -------------------------
# CLI
# -------------------------
@app.command()
def main(
    target: str = typer.Option(..., help="Target column in interim CSV."),
    input_csv: Path = typer.Option(
        INTERIM_DATA_DIR / "dataset.csv",
        help="Interim dataset (cleaned) CSV path.",
    ),
    output_model: Path = typer.Option(
        Path("artifacts/rf/model.joblib"),
        help="Where to write the trained model.",
    ),
    test_size: float = typer.Option(0.2, help="Test size for split."),
    random_state: int = typer.Option(42, help="Random seed."),
) -> None:
    """
    Train a RandomForest on the interim dataset and save the model.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Interim CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)  # type: ignore[name-defined]
    logger.info(f"Loaded interim CSV: {input_csv} ({df.shape})")

    model, *_ = train_model(
        df,
        target=target,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )
    save_model(model, output_model)
    logger.success("Done.")


if __name__ == "__main__":
    app()

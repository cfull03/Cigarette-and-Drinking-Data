# --- filepath: addiction/model.py ---
# [exp-001] - Contains methods modified/added in exp/001-smoking-trends-cf
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, cast

from joblib import dump, load
from loguru import logger
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)

__all__ = ["build_model", "save_model", "load_model", "app"]


# ----------------------------
# Helpers (local, for build)
# ----------------------------
def _to_dense64(X: Any) -> npt.NDArray[np.float64]:
    from scipy import sparse as _sp
    arr: np.ndarray = X.toarray() if _sp.issparse(X) else np.asarray(X)
    return cast(npt.NDArray[np.float64], arr.astype(np.float64, copy=False))
    # [exp-001]


def _to01(y: Any) -> npt.NDArray[np.int_]:
    arr: np.ndarray = np.asarray(y)
    if arr.dtype == bool:
        return cast(npt.NDArray[np.int_], arr.astype(np.int_, copy=False))
    u = np.unique(arr)
    if set(u.tolist()) <= {0, 1}:
        return cast(npt.NDArray[np.int_], arr.astype(np.int_, copy=False))
    raise ValueError(f"Labels must be boolean or 0/1. Got uniques={u}")
    # [exp-001]


# ----------------------------
# Public API
# ----------------------------
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
    """Fit and return a RandomForestClassifier."""
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
    # [exp-001]


def save_model(model: Any, path: Path | str) -> Path:
    """Persist a trained model with joblib."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out)
    logger.success(f"Saved model → {out}")
    return out
    # [exp-001]


def load_model(path: Path | str) -> Any:
    """Load a persisted model saved by `save_model`."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    logger.info(f"Loading model: {p}")
    return load(p)
    # [exp-001]


# ----------------------------
# CLI (delegates to train_model)
# ----------------------------
@app.command(name="main")
def main(
    target: str = typer.Option(..., help="Target column name."),
    input_csv: Path = typer.Option(
        Path("data/processed/features.csv"),
        help="CSV including feature columns and target.",
        exists=True,
        readable=True,
    ),
    output_model: Path = typer.Option(
        Path("artifacts/rf/model.joblib"),
        help="Trained model output path.",
    ),
    # split + preprocess are handled in train_model
    test_size: float = typer.Option(0.2, help="Test size split.", min=0.05, max=0.95),
    random_state: int = typer.Option(42, help="Random seed."),
    stratify: bool = typer.Option(True, help="Stratify by target when splitting."),
    preprocessor_path: Optional[Path] = typer.Option(
        None,
        help="Path to fitted preprocessor.joblib (fit on X only).",
    ),
    # RF hyperparams
    n_estimators: int = typer.Option(500),
    max_depth: Optional[int] = typer.Option(5),
    max_features: str = typer.Option("sqrt"),
    min_samples_leaf: int = typer.Option(1, min=1),
    class_weight: Optional[str] = typer.Option("balanced"),
    n_jobs: int = typer.Option(-1),
) -> None:
    """
    Thin CLI to kick off training via `addiction.modeling.train.train_model`, then save the model.
    Keeps `model.py` as the stable entry point some Make targets expect.
    """
    from addiction.modeling.train import train_model  # local import to avoid cycle

    df = pd.read_csv(input_csv)
    logger.info(f"Loaded data: {input_csv} shape={df.shape}")

    clf, *_matrices, metrics = train_model(
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

    save_model(clf, output_model)
    logger.success(f"Done. Model → {output_model}")
    logger.info(f"Summary metrics: {metrics}")
    # [exp-001]


if __name__ == "__main__":
    app()

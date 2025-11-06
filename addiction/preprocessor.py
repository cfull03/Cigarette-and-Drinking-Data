"""
scikit-learn preprocessing pipeline for DS/ML:

- Numeric: SimpleImputer(median) → StandardScaler
- Categorical: SimpleImputer(most_frequent) → OneHotEncoder (optional)
- Auto column inference with CLI overrides
- pandas output when supported; robust fallbacks otherwise
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, cast

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer

from addiction.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer(help="scikit-learn preprocessor (SimpleImputer + StandardScaler + optional OneHotEncoder).")

__all__ = [
    # column utilities
    "infer_columns",
    "infer_column_types",
    # builders & operations
    "build_preprocessor",
    "make_preprocessor",
    "fit_preprocessor",
    "transform_df",
    "get_feature_names_after_preprocessor",
    # persistence
    "save_preprocessor",
    "load_preprocessor",
]

# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------
def _csv_to_list(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    return [s.strip() for s in arg.split(",") if s.strip()]


def _get_bound_cols(ct: ColumnTransformer, name: str) -> list[str]:
    """Return currently bound columns for a transformer by name (mypy-safe)."""
    for n, _, cols in ct.transformers:
        if n == name:
            bound: Sequence[str] = cast(Sequence[str], cols)
            return list(bound)
    return []


def _rebind_columns(
    ct: ColumnTransformer,
    *,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> ColumnTransformer:
    """Update ColumnTransformer selections. Why: mypy-safe, avoids mutation surprises."""
    new_transformers: List[tuple] = []
    for name, trans, cols in ct.transformers:
        if name == "num":
            new_transformers.append((name, trans, list(numeric_cols)))
        elif name == "cat":
            new_transformers.append((name, trans, list(categorical_cols)))
        else:
            new_transformers.append((name, trans, cols))
    ct.transformers = new_transformers  # type: ignore[attr-defined]
    return ct


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def infer_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical columns."""
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num, cat


def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Alias for infer_columns(df)."""
    return infer_columns(df)


def build_preprocessor(
    *,
    numeric_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    encode_categoricals: bool = True,
) -> ColumnTransformer:
    """
    Factory for a ColumnTransformer with numeric + categorical branches.
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    if encode_categoricals:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)  # type: ignore[call-arg]
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", ohe),
            ]
        )
    else:
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])

    ncols = list(numeric_cols) if numeric_cols else []
    ccols = list(categorical_cols) if categorical_cols else []

    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, ncols),
            ("cat", cat_pipe, ccols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    try:
        ct.set_output(transform="pandas")  # sklearn>=1.2
    except Exception:
        pass
    return ct


# friendly alias to match other modules' naming
make_preprocessor = build_preprocessor


def fit_preprocessor(
    df: pd.DataFrame,
    *,
    numeric_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    encode_categoricals: bool = True,
) -> ColumnTransformer:
    """Build and fit the preprocessor on df (call on TRAIN ONLY to avoid leakage)."""
    if numeric_cols is None or categorical_cols is None:
        inf_num, inf_cat = infer_columns(df)
        numeric_cols = inf_num if numeric_cols is None else list(numeric_cols)
        categorical_cols = inf_cat if categorical_cols is None else list(categorical_cols)

    ct = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        encode_categoricals=encode_categoricals,
    )
    _rebind_columns(ct, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    ct.fit(df)
    return ct


def transform_df(
    df: pd.DataFrame,
    ct: ColumnTransformer,
    *,
    numeric_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Apply a fitted preprocessor to df. If cols provided, rebind selections first.
    """
    if numeric_cols is not None or categorical_cols is not None:
        current_num: List[str] = _get_bound_cols(ct, "num")
        current_cat: List[str] = _get_bound_cols(ct, "cat")

        bound_num: List[str] = list(numeric_cols) if numeric_cols is not None else current_num
        bound_cat: List[str] = list(categorical_cols) if categorical_cols is not None else current_cat
        _rebind_columns(ct, numeric_cols=bound_num, categorical_cols=bound_cat)

    out = ct.transform(df)

    if isinstance(out, pd.DataFrame):
        return out  # pandas output path

    # Fallback: construct names for numpy output
    feature_names: List[str] = get_feature_names_after_preprocessor(
        ct,
        numeric_cols=_get_bound_cols(ct, "num"),
        categorical_cols=_get_bound_cols(ct, "cat"),
    )
    return pd.DataFrame(out, columns=feature_names, index=df.index)


def get_feature_names_after_preprocessor(
    ct: ColumnTransformer,
    *,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> List[str]:
    """
    Recover feature names after a fitted ColumnTransformer (handles OHE if present).
    """
    names: List[str] = []
    names.extend(list(numeric_cols))

    try:
        transformers = dict(ct.named_transformers_)
        cat = transformers.get("cat")
        if cat not in ("drop", "passthrough", None):
            ohe = getattr(cat, "named_steps", {}).get("ohe")
            if ohe is not None:
                names.extend(ohe.get_feature_names_out(list(categorical_cols)).tolist())
            else:
                names.extend(list(categorical_cols))
        elif cat == "passthrough":
            names.extend(list(categorical_cols))
    except Exception:
        names.extend(list(categorical_cols))

    return names


# -----------------------------------------------------------------------------
# Persistence (Public API)
# -----------------------------------------------------------------------------
def save_preprocessor(ct: ColumnTransformer, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ct, p)
    logger.info(f"[preprocessor] saved → {p}")


def load_preprocessor(path: Path | str) -> ColumnTransformer:
    p = Path(path)
    ct: ColumnTransformer = joblib.load(p)
    logger.info(f"[preprocessor] loaded ← {p}")
    return ct


# -----------------------------
# CLI (dataset.py style)
# -----------------------------
@app.command()
def main(
    mode: str = typer.Option(..., help="fit | transform | fit-transform"),
    input_path: Path = typer.Option(PROCESSED_DATA_DIR / "dataset.csv", help="Input CSV."),
    output_path: Path = typer.Option(PROCESSED_DATA_DIR / "features.csv", help="Output CSV for transform/fit-transform."),
    model_path: Path = typer.Option(MODELS_DIR / "preprocessor.joblib", help="Where to save/load the fitted preprocessor."),
    num_cols: Optional[str] = typer.Option(None, help="Comma-separated numeric columns (override inference)."),
    cat_cols: Optional[str] = typer.Option(None, help="Comma-separated categorical columns (override inference)."),
    encode_cat: bool = typer.Option(True, "--encode-cat/--no-encode-cat", help="Enable OneHotEncoder for categoricals."),
) -> None:
    """
    fit: infer/build and fit preprocessor; save joblib.
    transform: load preprocessor; transform CSV; write features.
    fit-transform: fit then transform same CSV; save joblib and features.
    """
    if not input_path.exists():
        typer.echo(f"[ERROR] Input not found: {input_path}")
        raise typer.Exit(code=1)

    df = pd.read_csv(input_path)
    numeric_cols_list = _csv_to_list(num_cols)
    categorical_cols_list = _csv_to_list(cat_cols)

    if mode == "fit":
        logger.info("Fitting preprocessor…")
        ct = fit_preprocessor(
            df,
            numeric_cols=numeric_cols_list,
            categorical_cols=categorical_cols_list,
            encode_categoricals=encode_cat,
        )
        save_preprocessor(ct, model_path)
        logger.success(f"Saved preprocessor → {model_path}")

    elif mode == "transform":
        logger.info("Loading preprocessor and transforming…")
        ct = load_preprocessor(model_path)
        feats = transform_df(
            df,
            ct,
            numeric_cols=numeric_cols_list,
            categorical_cols=categorical_cols_list,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(output_path, index=False)
        logger.success(f"Wrote features → {output_path}")

    elif mode == "fit-transform":
        logger.info("Fitting, transforming, and saving…")
        ct = fit_preprocessor(
            df,
            numeric_cols=numeric_cols_list,
            categorical_cols=categorical_cols_list,
            encode_categoricals=encode_cat,
        )
        feats = transform_df(df, ct)  # columns bound during fit
        save_preprocessor(ct, model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(output_path, index=False)
        logger.success(f"Saved preprocessor → {model_path} and features → {output_path}")

    else:
        typer.echo(f"[ERROR] Unknown mode: {mode} (use fit | transform | fit-transform)")
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()

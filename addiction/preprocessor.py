# filepath: addiction/preprocessor.py
# [exp-001] - Contains methods modified/added in exp/001-smoking-trends-cf
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union, cast

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer

from addiction.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer(help="scikit-learn preprocessor (SimpleImputer + StandardScaler + OneHotEncoder).")

__all__ = [
    "infer_columns",
    "infer_column_types",
    "build_preprocessor",
    "make_preprocessor",
    "fit_preprocessor",
    "transform_df",
    "get_feature_names_after_preprocessor",
    "save_preprocessor",
    "load_preprocessor",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _csv_to_list(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    return [s.strip() for s in arg.split(",") if s.strip()]
    # [exp-001]

def _get_bound_cols(ct: ColumnTransformer, name: str) -> list[str]:
    for n, _, cols in ct.transformers:
        if n == name:
            bound: Sequence[str] = cast(Sequence[str], cols)
            return list(bound)
    return []
    # [exp-001]

def _drop_target_cols(df: pd.DataFrame, target: Optional[str]) -> pd.DataFrame:
    if not target:
        return df.copy()
    out = df.copy()
    pref = f"{target}_"
    to_drop = [c for c in out.columns if c == target or c.startswith(pref)]
    if to_drop:
        logger.info(f"[preprocessor] Dropping target-derived columns: {to_drop}")
        out = out.drop(columns=to_drop, errors="ignore")
    return out
    # [exp-001]

def _ensure_dataframe(
    X: Union[pd.DataFrame, np.ndarray, "sparse.spmatrix", Sequence[Sequence[object]]],
    index: Optional[pd.Index],
    columns: Optional[Iterable[str]],
) -> pd.DataFrame:
    cols = list(columns) if columns is not None else None
    if sparse.issparse(X):
        X = X.tocsr()
        return pd.DataFrame.sparse.from_spmatrix(X, index=index, columns=cols)
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X, index=index, columns=cols)
    if isinstance(X, pd.DataFrame):
        if cols is not None and list(X.columns) != cols:
            X = X.copy()
            X.columns = cols
        return X
    return pd.DataFrame(X, index=index, columns=cols)
    # [exp-001]

# -----------------------------------------------------------------------------
# Column inference & partitioning
# -----------------------------------------------------------------------------
def infer_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num, cat
    # [exp-001]

infer_column_types = infer_columns

def _partition_all_nan(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    some, all_nan = [], []
    for c in cols:
        (some if df[c].notna().any() else all_nan).append(c)
    return some, all_nan
    # [exp-001]

def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)  # type: ignore[call-arg]
    # [exp-001]

# -----------------------------------------------------------------------------
# Build / Fit / Transform
# -----------------------------------------------------------------------------
def build_preprocessor(
    *,
    numeric_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    encode_categoricals: bool = True,  # kept for CLI compatibility; currently ignored
) -> ColumnTransformer:
    """
    Deprecated low-info builder (kept for API). Prefer `fit_preprocessor(df, ...)` which
    partitions columns to avoid all-NaN median warnings.
    """
    # numeric: median → fallback constant(0) → scale
    num_pipe = Pipeline(
        steps=[
            ("imputer_median", SimpleImputer(strategy="median")),
            ("imputer_fallback", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )
    # categorical: most_frequent + OHE (sparse)
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", _make_ohe()),
        ]
    )
    ncols = list(numeric_cols) if numeric_cols else []
    ccols = list(categorical_cols) if categorical_cols else []
    ct = ColumnTransformer(
        transformers=[("num", num_pipe, ncols), ("cat", cat_pipe, ccols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ct

# alias
make_preprocessor = build_preprocessor

def fit_preprocessor(
    df: pd.DataFrame,
    *,
    numeric_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    encode_categoricals: bool = True,  # ignored
) -> ColumnTransformer:
    """
    Build a preprocessor that **avoids median-imputer warnings** by separating
    all-NaN columns into a constant-impute branch.
    """
    if numeric_cols is None or categorical_cols is None:
        inf_num, inf_cat = infer_columns(df)
        numeric_cols = inf_num if numeric_cols is None else list(numeric_cols)
        categorical_cols = inf_cat if categorical_cols is None else list(categorical_cols)

    num_some, num_all_nan = _partition_all_nan(df, numeric_cols)
    cat_some, cat_all_nan = _partition_all_nan(df, categorical_cols)

    if num_all_nan or cat_all_nan:
        logger.warning(
            "All-NaN columns detected; using constant imputation. "
            f"num_all_nan={num_all_nan} cat_all_nan={cat_all_nan}"
        )

    # Pipelines
    num_some_pipe = Pipeline(
        steps=[("imputer_median", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    num_all_nan_pipe = Pipeline(
        steps=[("imputer_const0", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler())]
    )
    cat_some_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", _make_ohe())]
    )
    cat_all_nan_pipe = Pipeline(
        steps=[("imputer_const", SimpleImputer(strategy="constant", fill_value="missing")), ("ohe", _make_ohe())]
    )

    transformers: List[tuple] = []
    if num_some:
        transformers.append(("num_median", num_some_pipe, list(num_some)))
    if num_all_nan:
        transformers.append(("num_const", num_all_nan_pipe, list(num_all_nan)))
    if cat_some:
        transformers.append(("cat_freq", cat_some_pipe, list(cat_some)))
    if cat_all_nan:
        transformers.append(("cat_const", cat_all_nan_pipe, list(cat_all_nan)))

    if not transformers:
        raise ValueError("No columns to preprocess.")

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    ct.fit(df)

    # Persist fitted column lists (flattened)
    setattr(ct, "_num_cols", list(numeric_cols))
    setattr(ct, "_cat_cols", list(categorical_cols))
    return ct
    # [exp-001]

def _feature_names_after_ohe(ct: ColumnTransformer) -> List[str]:
    try:
        return list(ct.get_feature_names_out())
    except Exception:
        num = cast(List[str], getattr(ct, "_num_cols", []))
        cat = cast(List[str], getattr(ct, "_cat_cols", []))
        names: List[str] = list(num)
        try:
            for name, trans, _ in getattr(ct, "transformers_", []):
                if "cat" in name and isinstance(trans, Pipeline) and "ohe" in trans.named_steps:
                    ohe = trans.named_steps["ohe"]
                    names.extend(list(ohe.get_feature_names_out(cat)))
            if len(names) == 0:
                raise RuntimeError
        except Exception:
            names = [f"f{i}" for i in range(ct.transform(np.zeros((1, 0))).shape[1])]
        return names

def transform_df(
    df: pd.DataFrame,
    ct: ColumnTransformer,
    *,
    numeric_cols: Optional[Sequence[str]] = None,      # ignored after fit
    categorical_cols: Optional[Sequence[str]] = None,  # ignored after fit
) -> pd.DataFrame:
    X = ct.transform(df)
    names = _feature_names_after_ohe(ct)

    width = X.shape[1] if hasattr(X, "shape") else len(names)
    if len(names) != width:
        logger.warning(
            "[preprocessor] Feature name count (%d) != transformed width (%d); falling back to default.",
            len(names), width,
        )
        names = None  # let pandas auto-range columns

    return _ensure_dataframe(X, index=df.index, columns=names)

def get_feature_names_after_preprocessor(ct: ColumnTransformer, *, numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> List[str]:
    return _feature_names_after_ohe(ct)

# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------
def save_preprocessor(ct: ColumnTransformer, path: Path | str) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ct, p)
    logger.info(f"[preprocessor] saved → {p}")

def load_preprocessor(path: Path | str) -> ColumnTransformer:
    p = Path(path); ct: ColumnTransformer = joblib.load(p)
    logger.info(f"[preprocessor] loaded ← {p}")
    return ct

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
@app.command()
def main(
    mode: str = typer.Option(..., help="fit | transform | fit-transform"),
    input_path: Path = typer.Option(PROCESSED_DATA_DIR / "dataset.csv", help="Input CSV."),
    output_path: Path = typer.Option(PROCESSED_DATA_DIR / "dataset.preprocessed.csv", help="Output CSV for transform/fit-transform."),
    model_path: Path = typer.Option(MODELS_DIR / "preprocessor.joblib", help="Where to save/load the fitted preprocessor."),
    num_cols: Optional[str] = typer.Option(None, help="Comma-separated numeric columns (override inference)."),
    cat_cols: Optional[str] = typer.Option(None, help="Comma-separated categorical columns (override inference)."),
    encode_cat: bool = typer.Option(True, "--encode-cat/--no-encode-cat", help="(kept for compat; ignored)."),
    target: Optional[str] = typer.Option(None, help="If provided and column exists, drop target (+ any '<target>_*') before fitting/transforming."),
) -> None:
    if not input_path.exists():
        typer.echo(f"[ERROR] Input not found: {input_path}")
        raise typer.Exit(code=1)

    df_raw = pd.read_csv(input_path)
    df = _drop_target_cols(df_raw, target)

    numeric_cols_list = _csv_to_list(num_cols) or None
    categorical_cols_list = _csv_to_list(cat_cols) or None

    if mode == "fit":
        logger.info("Fitting preprocessor…")
        ct = fit_preprocessor(
            df,
            numeric_cols=numeric_cols_list,
            categorical_cols=categorical_cols_list,
            encode_categoricals=True,
        )
        save_preprocessor(ct, model_path)
        logger.success(f"Saved preprocessor → {model_path}")

    elif mode == "transform":
        logger.info("Loading preprocessor and transforming…")
        ct = load_preprocessor(model_path)
        feats = transform_df(df, ct)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(output_path, index=False)
        logger.success(f"Wrote preprocessed features → {output_path}")

    elif mode == "fit-transform":
        logger.info("Fitting, transforming, and saving…")
        ct = fit_preprocessor(
            df,
            numeric_cols=numeric_cols_list,
            categorical_cols=categorical_cols_list,
            encode_categoricals=True,
        )
        feats = transform_df(df, ct)
        save_preprocessor(ct, model_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(output_path, index=False)
        logger.success(f"Saved preprocessor → {model_path} and preprocessed features → {output_path}")
    else:
        typer.echo(f"[ERROR] Unknown mode: {mode} (use fit | transform | fit-transform)")
        raise typer.Exit(code=2)

if __name__ == "__main__":
    app()

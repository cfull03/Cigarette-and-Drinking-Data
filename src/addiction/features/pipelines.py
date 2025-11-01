# file: src/addiction/features/pipelines.py
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Why: shared signature for train/evaluate to import.
PreOut = Tuple[ColumnTransformer, List[str], List[str]]


def infer_columns(
    df: pd.DataFrame,
    *,
    forced_numeric: Optional[Sequence[str]] = None,
    forced_categorical: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Decide numeric vs categorical columns.
    Explicit lists (if provided) win over dtype inference.
    """
    if forced_numeric is not None:
        num = [c for c in forced_numeric if c in df.columns]
        rest = [c for c in df.columns if c not in num]
        cat = [c for c in (forced_categorical or []) if c in rest] or [c for c in rest if c not in num]
        return num, cat

    num = df.select_dtypes(include=[np.number, "datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    cat = [c for c in df.columns if c not in num]
    if forced_categorical is not None:
        cat = [c for c in forced_categorical if c in df.columns]
        num = [c for c in df.columns if c not in cat]
    return num, cat


def build_preprocessor(
    X: pd.DataFrame,
    *,
    numeric_strategy: str = "median",
    scale: bool = True,
    with_mean: bool = False,  # keep sparse-friendly
    categorical_strategy: str = "most_frequent",
    one_hot: bool = True,
    handle_unknown: str = "ignore",
    min_frequency: Optional[int | float] = None,  # e.g., 0.01 or 10
    max_categories: Optional[int] = None,         # cap for OHE blowup
) -> PreOut:
    """
    Construct a ColumnTransformer for mixed-type tabular features.
    """
    num_cols, cat_cols = infer_columns(X)

    num_steps: List[tuple] = [("impute", SimpleImputer(strategy=numeric_strategy))]
    if scale:
        num_steps.append(("scale", StandardScaler(with_mean=with_mean)))

    cat_steps: List[tuple] = [("impute", SimpleImputer(strategy=categorical_strategy))]
    if one_hot and len(cat_cols) > 0:
        ohe = OneHotEncoder(
            handle_unknown=handle_unknown,
            sparse_output=True,
            min_frequency=min_frequency,
            max_categories=max_categories,
        )
        cat_steps.append(("ohe", ohe))

    num_pipe = Pipeline(steps=num_steps)
    cat_pipe = Pipeline(steps=cat_steps)

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,  # keep sparse when possible
    )
    return pre, num_cols, cat_cols


def build_preprocessor_from_config(
    cfg,
    df: pd.DataFrame,
) -> PreOut:
    """
    Config-aware builder. Reads:
      - data.id_column
      - features.drop_columns
      - features.categorical_columns / features.numeric_columns (optional)
      - preprocessing.numeric.imputer.strategy
      - preprocessing.numeric.scaler.standardize / with_mean
      - preprocessing.categorical.imputer.strategy
      - preprocessing.categorical.one_hot.[handle_unknown, sparse_output, min_frequency, max_categories]
    """
    # Lazy import to avoid hard coupling on utilities in case of standalone use
    try:
        from ..utilities.config import Config
        if not isinstance(cfg, Config):
            cfg = Config.load(cfg)  # accepts str/Path
    except Exception:
        pass  # assume cfg is mapping-like

    id_col = cfg.get("data.id_column")
    drop_cols = list(cfg.get("features.drop_columns", []) or [])
    if id_col:
        drop_cols.append(id_col)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    forced_cat = cfg.get("features.categorical_columns", []) or None
    forced_num = cfg.get("features.numeric_columns", []) or None
    num_cols, cat_cols = infer_columns(X, forced_numeric=forced_num, forced_categorical=forced_cat)

    # Numeric config
    num_strategy = str(cfg.get("preprocessing.numeric.imputer.strategy", "median"))
    do_scale = bool(cfg.get("preprocessing.numeric.scaler.standardize", True))
    with_mean = bool(cfg.get("preprocessing.numeric.scaler.with_mean", False))

    # Categorical config
    cat_strategy = str(cfg.get("preprocessing.categorical.imputer.strategy", "most_frequent"))
    handle_unknown = str(cfg.get("preprocessing.categorical.one_hot.handle_unknown", "ignore"))
    min_freq = cfg.get("preprocessing.categorical.one_hot.min_frequency", None)
    max_cats = cfg.get("preprocessing.categorical.one_hot.max_categories", None)

    num_steps: List[tuple] = [("impute", SimpleImputer(strategy=num_strategy))]
    if do_scale:
        num_steps.append(("scale", StandardScaler(with_mean=with_mean)))

    cat_steps: List[tuple] = [("impute", SimpleImputer(strategy=cat_strategy))]
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(
            handle_unknown=handle_unknown,
            sparse_output=True,
            min_frequency=min_freq,
            max_categories=max_cats,
        )
        cat_steps.append(("ohe", ohe))

    num_pipe = Pipeline(steps=num_steps)
    cat_pipe = Pipeline(steps=cat_steps)

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    return pre, num_cols, cat_cols

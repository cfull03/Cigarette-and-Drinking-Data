# file: src/addiction/data/loaders.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utilities.config import Config, load_config
from ..utilities.io import read_csv, raw_csv_path, to_interim


@dataclass(frozen=True)
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    target: str


def _as_config(cfg: Union[Config, str, Path]) -> Config:
    return cfg if isinstance(cfg, Config) else load_config(cfg)


# ----------------------------- core loaders -----------------------------

def load_raw(cfg: Union[Config, str, Path] = "config/config.yaml",
             usecols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    c = _as_config(cfg)
    path = raw_csv_path(c)
    dt_cols = c.get("data.datetime_columns", []) or []
    parse_dates = [col for col in dt_cols if isinstance(col, str)]
    return read_csv(path, usecols=usecols, parse_dates=parse_dates if parse_dates else None)


def feature_target(
    df: pd.DataFrame,
    target: str,
    *,
    drop_cols: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in DataFrame.")
    drops = set(drop_cols or [])
    drops.add(target)
    X = df.drop(columns=[c for c in drops if c in df.columns], errors="ignore")
    y = df[target]
    return X, y


def train_test_split_df(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    seed: int,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=(y if stratify else None))


def infer_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in df.columns if c not in num]
    return num, cat


# ----------------------------- high-level API -----------------------------

def make_splits_from_config(
    cfg: Union[Config, str, Path] = "config/config.yaml",
    *,
    save_to_interim: bool = False,
    save_prefix: str = "split",
) -> Dict[str, SplitResult]:
    """
    Build train/test splits for targets declared in config:
      data.target_regression, data.target_classification
    """
    c = _as_config(cfg)
    seed: int = int(c.get("project.seed", 42))
    test_size: float = float(c.get("data.test_size", 0.2))
    id_col: Optional[str] = c.get("data.id_column")
    drop_cols = [id_col] if id_col else []

    df = load_raw(c)
    results: Dict[str, SplitResult] = {}

    # Regression
    reg_tgt: Optional[str] = c.get("data.target_regression")
    if isinstance(reg_tgt, str) and reg_tgt in df.columns:
        reg_df = df.dropna(subset=[reg_tgt])
        Xr, yr = feature_target(reg_df, reg_tgt, drop_cols=drop_cols)
        Xtr, Xte, ytr, yte = train_test_split_df(Xr, yr, test_size=test_size, seed=seed, stratify=False)
        results["regression"] = SplitResult(Xtr, Xte, ytr, yte, reg_tgt)
        if save_to_interim:
            to_interim(pd.concat([Xtr, ytr.rename(reg_tgt)], axis=1), f"{save_prefix}_reg_train.csv", c)
            to_interim(pd.concat([Xte, yte.rename(reg_tgt)], axis=1), f"{save_prefix}_reg_test.csv", c)

    # Classification
    clf_tgt: Optional[str] = c.get("data.target_classification")
    if isinstance(clf_tgt, str) and clf_tgt in df.columns:
        clf_df = df.dropna(subset=[clf_tgt])
        Xc, yc = feature_target(clf_df, clf_tgt, drop_cols=drop_cols)
        stratify = bool(c.get("data.stratify_classification", True))
        Xtr, Xte, ytr, yte = train_test_split_df(Xc, yc, test_size=test_size, seed=seed, stratify=stratify)
        results["classification"] = SplitResult(Xtr, Xte, ytr, yte, clf_tgt)
        if save_to_interim:
            to_interim(pd.concat([Xtr, ytr.rename(clf_tgt)], axis=1), f"{save_prefix}_clf_train.csv", c)
            to_interim(pd.concat([Xte, yte.rename(clf_tgt)], axis=1), f"{save_prefix}_clf_test.csv", c)

    if not results:
        raise ValueError("No targets found. Set data.target_regression and/or data.target_classification in config.")
    return results

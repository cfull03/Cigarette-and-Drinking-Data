# Files to add/update

## File: `src/addiction_ds/train.py`
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# Prefer project IO utilities if available
try:
    from addiction_ds.io import load_cfg, save_model, get_paths
except Exception:  # fallback if helpers not present
    load_cfg = None  # type: ignore
    save_model = None  # type: ignore
    get_paths = None  # type: ignore
    import yaml  # type: ignore


# --------------------------- Config -----------------------------------------
@dataclass
class TrainConfig:
    random_state: int
    label: str
    features_numeric: list[str]
    features_categorical: list[str]
    paths: Dict[str, str]
    split_test_size: float
    split_stratify: bool
    model_name: str
    model_params: Dict[str, Any]

    @staticmethod
    def from_yaml(path: str = "configs/experiment.yaml") -> "TrainConfig":
        if load_cfg is not None:
            cfg = load_cfg(path)
        else:  # minimal fallback
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)  # type: ignore[name-defined]

        return TrainConfig(
            random_state=int(cfg.get("random_state", 42)),
            label=str(cfg["label"]),
            features_numeric=list(cfg["features"]["numeric"]),
            features_categorical=list(cfg["features"]["categorical"]),
            paths=dict(cfg.get("paths", {})),
            split_test_size=float(cfg.get("split", {}).get("test_size", 0.2)),
            split_stratify=bool(cfg.get("split", {}).get("stratify", True)),
            model_name=str(cfg.get("model", {}).get("name", "logistic_regression")),
            model_params=dict(cfg.get("model", {}).get("params", {"max_iter": 1000})),
        )


# --------------------------- Utils ------------------------------------------
ALLOWED = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "svc": SVC,  # we will force probability=True
    "sgd_classifier": SGDClassifier,
}


def _most_recent_csv(processed_dir: Path) -> Path | None:
    files = sorted(processed_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _pick_processed_inputs(cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df) using the most recent processed CSVs.

    Preference order:
    1) If both explicit train/val CSVs exist in cfg.paths, use them.
    2) Else, take the newest CSV from `data/processed/` and split it.
    """
    train_csv = Path(cfg.paths.get("train_csv", "data/processed/train.csv"))
    val_csv = Path(cfg.paths.get("val_csv", "data/processed/val.csv"))

    if train_csv.exists() and val_csv.exists():
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        return train_df, val_df

    processed_dir = Path(train_csv).parent if train_csv.parent.name else Path("data/processed")
    newest = _most_recent_csv(processed_dir)
    if newest is None:
        raise FileNotFoundError(f"No processed CSVs found in {processed_dir} — run your preprocessing first.")

    df = pd.read_csv(newest).drop_duplicates()

    # ensure label exists; if not, attempt to derive a default (customize if needed)
    if cfg.label not in df.columns and "smokes_per_day" in df.columns:
        df[cfg.label] = (df["smokes_per_day"].fillna(0) > 0).astype(int)

    if cfg.label not in df.columns:
        raise KeyError(f"Label '{cfg.label}' not found in {newest}. Update configs/experiment.yaml or preprocessing.")

    strat = df[cfg.label] if cfg.split_stratify else None
    train_df, val_df = train_test_split(
        df,
        test_size=cfg.split_test_size,
        random_state=cfg.random_state,
        stratify=strat,
    )
    return train_df, val_df


def _build_preprocessor(numeric: Iterable[str], categorical: Iterable[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(numeric)),
            ("cat", cat_pipe, list(categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_estimator(name: str, params: Dict[str, Any], random_state: int):
    if name not in ALLOWED:
        raise ValueError(f"Unsupported model '{name}'. Allowed: {sorted(ALLOWED)}")

    Est = ALLOWED[name]

    # default tweaks
    if name == "svc":
        params = {"probability": True, **params}
    if name == "sgd_classifier":
        params = {"loss": params.get("loss", "log_loss"), **params}

    # inject random_state where supported
    try:
        if "random_state" in Est().get_params():
            params = {"random_state": random_state, **params}
    except Exception:
        pass

    return Est(**params)


# --------------------------- Train ------------------------------------------

def main() -> None:
    cfg = TrainConfig.from_yaml()

    train_df, val_df = _pick_processed_inputs(cfg)

    X_cols = cfg.features_numeric + cfg.features_categorical
    y_col = cfg.label

    # Sanity checks on columns
    missing_train = [c for c in X_cols + [y_col] if c not in train_df.columns]
    missing_val = [c for c in X_cols + [y_col] if c not in val_df.columns]
    if missing_train:
        raise KeyError(f"Missing columns in train_df: {missing_train}")
    if missing_val:
        raise KeyError(f"Missing columns in val_df: {missing_val}")

    X_train, y_train = train_df[X_cols], train_df[y_col]
    X_val, y_val = val_df[X_cols], val_df[y_col]

    pre = _build_preprocessor(cfg.features_numeric, cfg.features_categorical)
    est = _make_estimator(cfg.model_name, cfg.model_params, cfg.random_state)

    pipe = Pipeline([
        ("pre", pre),
        ("model", est),
    ])
    pipe.fit(X_train, y_train)

    # Evaluate with probability or decision function
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        scores = pipe.predict_proba(X_val)[:, 1]
    elif hasattr(pipe.named_steps["model"], "decision_function"):
        s = np.asarray(pipe.decision_function(X_val), dtype=float)
        scores = (s - s.min()) / (s.max() - s.min() + 1e-12)
    else:
        raise RuntimeError("Estimator provides neither predict_proba nor decision_function.")

    auc = float(roc_auc_score(y_val, scores))
    print(json.dumps({"val_auc": round(auc, 4), "model": cfg.model_name}, indent=2))

    # Save artifacts using project helper if available; else fallback to joblib
    if save_model is not None and get_paths is not None:
        save_model(pipe, load_cfg("configs/experiment.yaml") if load_cfg else cfg.__dict__, name="latest", framework="sklearn")
        # timestamped copy
        import time
        stamp = time.strftime("%Y%m%d-%H%M%S")
        save_model(pipe, load_cfg("configs/experiment.yaml") if load_cfg else cfg.__dict__, name=f"{stamp}_{cfg.model_name}", framework="sklearn")
    else:
        # minimal fallback
        from joblib import dump
        models_dir = Path(cfg.paths.get("models_dir", "models"))
        models_dir.mkdir(parents=True, exist_ok=True)
        dump({"pipeline": pipe, "features": X_cols, "label": y_col}, models_dir / "latest.joblib")


if __name__ == "__main__":
    main()
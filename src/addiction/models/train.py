# file: src/addiction/models/train.py
from __future__ import annotations

import argparse
import json
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from ..data.loaders import make_splits_from_config
from ..features.pipelines import build_preprocessor_from_config
from ..utilities.config import Config, load_config
from ..utilities.io import save_figure, save_json, save_model
from ..utilities.logging import get_logger

# ------------------------- aliases for quick use -------------------------

_ALIAS_REG = {
    "rf": "sklearn.ensemble.RandomForestRegressor",
    "gbt": "sklearn.ensemble.GradientBoostingRegressor",
    "linreg": "sklearn.linear_model.LinearRegression",
    "ridge": "sklearn.linear_model.Ridge",
    "lasso": "sklearn.linear_model.Lasso",
    "svr": "sklearn.svm.SVR",
}
_ALIAS_CLF = {
    "rf": "sklearn.ensemble.RandomForestClassifier",
    "gbt": "sklearn.ensemble.GradientBoostingClassifier",
    "logreg": "sklearn.linear_model.LogisticRegression",
    "svc": "sklearn.svm.SVC",
}

EstimatorSpec = Union[str, Any, type]

# ------------------------- resolver & preprocessing -------------------------

def _import_object(dotted_path: str):
    mod, name = dotted_path.rsplit(".", 1)
    return getattr(import_module(mod), name)

def _resolve_estimator(task: str, spec: EstimatorSpec, params: Optional[Dict[str, Any]]) -> Any:
    """
    Accepts:
      - alias string ("rf", "logreg")
      - dotted path string ("sklearn.ensemble.RandomForestRegressor")
      - estimator class (e.g., RandomForestRegressor)
      - estimator instance (already constructed)
    Returns an estimator instance configured with params (if applicable).
    """
    if hasattr(spec, "fit") and hasattr(spec, "get_params"):
        return spec  # already an estimator instance

    if isinstance(spec, str):
        aliases = _ALIAS_REG if task == "regression" else _ALIAS_CLF
        dotted = aliases.get(spec.lower(), spec)
        cls = _import_object(dotted)
        return cls(**(params or {}))

    if isinstance(spec, type):
        return spec(**(params or {}))

    raise ValueError(f"Unsupported estimator spec for {task}: {spec!r}")

def _build_pipeline(cfg: Config, X: pd.DataFrame, estimator: Any) -> Pipeline:
    pre, _, _ = build_preprocessor_from_config(cfg, X)
    return Pipeline([("pre", pre), ("model", estimator)])

def _feature_names(pre: ColumnTransformer) -> List[str]:
    names: List[str] = []
    for name, trans, cols in pre.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            try:
                out = list(trans.get_feature_names_out(cols))
            except Exception:
                out = list(cols if isinstance(cols, list) else [cols])
        else:
            out = list(cols if isinstance(cols, list) else [cols])
        names.extend(out)
    return names

# ------------------------------ plotting ------------------------------

def _plot_reg_scatter(y_true: pd.Series, y_pred: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi])
    ax.set_title(title)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    return fig

def _plot_importances(est, feature_names: List[str], title: str, top_k: int = 20):
    if not hasattr(est, "feature_importances_"):
        return None
    importances = np.asarray(est.feature_importances_)
    idx = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, max(4, int(top_k * 0.35))))
    ax.barh(range(len(idx)), importances[idx][::-1])
    ax.set_yticks(range(len(idx)), [feature_names[i] for i in idx][::-1])
    ax.set_title(title)
    fig.tight_layout()
    return fig

# ------------------------- core training routines -------------------------

def train_regression(
    cfg: Union[Config, str],
    *,
    estimator: EstimatorSpec = "rf",
    params: Optional[Dict[str, Any]] = None,
    save_artifacts: bool = True,
):
    cfg = cfg if isinstance(cfg, Config) else load_config(cfg)
    log = get_logger("train.regression", cfg=cfg, to_file=True)

    splits = make_splits_from_config(cfg)
    if "regression" not in splits:
        log.info("No regression target configured; skipping.")
        return None

    sr = splits["regression"]
    est = _resolve_estimator("regression", estimator, params or cfg.get("models.regression.params", {}))
    pipe = _build_pipeline(cfg, sr.X_train, est)
    pipe.fit(sr.X_train, sr.y_train)

    pred = pipe.predict(sr.X_test)
    metrics = {
        "MAE": float(mean_absolute_error(sr.y_test, pred)),
        "MSE": float(mean_squared_error(sr.y_test, pred)),
        "R2": float(r2_score(sr.y_test, pred)),
    }
    out = {"estimator": pipe.named_steps["model"].__class__.__name__, "metrics": metrics, "target": sr.target}

    if save_artifacts:
        run = str(cfg.get("project.run_name", "run"))
        save_model(pipe, f"{run}_reg.joblib", cfg)
        save_json({"regression": out}, "metrics_regression.json", cfg, subdir="reports")
        try:
            if cfg.get("evaluation.save_plots", True):
                fig = _plot_reg_scatter(sr.y_test, pred, f"{sr.target}: True vs Pred")
                save_figure(fig, "reg_true_vs_pred.png", cfg); plt.close(fig)
                fn = _feature_names(pipe.named_steps["pre"])
                fig2 = _plot_importances(pipe.named_steps["model"], fn, "Regressor feature importances")
                if fig2:
                    save_figure(fig2, "reg_feature_importances.png", cfg); plt.close(fig2)
        except Exception as e:
            log.warning("Plotting skipped (regression): %s", e)

    log.info("Regression metrics: %s", json.dumps(metrics))
    return pipe, out

def train_classification(
    cfg: Union[Config, str],
    *,
    estimator: EstimatorSpec = "rf",
    params: Optional[Dict[str, Any]] = None,
    save_artifacts: bool = True,
):
    cfg = cfg if isinstance(cfg, Config) else load_config(cfg)
    log = get_logger("train.classification", cfg=cfg, to_file=True)

    splits = make_splits_from_config(cfg)
    if "classification" not in splits:
        log.info("No classification target configured; skipping.")
        return None

    sr = splits["classification"]
    est = _resolve_estimator("classification", estimator, params or cfg.get("models.classification.params", {}))
    pipe = _build_pipeline(cfg, sr.X_train, est)
    pipe.fit(sr.X_train, sr.y_train)

    pred = pipe.predict(sr.X_test)
    metrics = {
        "accuracy": float(accuracy_score(sr.y_test, pred)),
        "f1_weighted": float(f1_score(sr.y_test, pred, average="weighted")),
    }
    report = classification_report(sr.y_test, pred, output_dict=True)
    out = {"estimator": pipe.named_steps["model"].__class__.__name__, "metrics": metrics, "report": report, "target": sr.target}

    if save_artifacts:
        run = str(cfg.get("project.run_name", "run"))
        save_model(pipe, f"{run}_clf.joblib", cfg)
        save_json({"classification": out}, "metrics_classification.json", cfg, subdir="reports")
        try:
            if cfg.get("evaluation.save_plots", True):
                disp = ConfusionMatrixDisplay.from_predictions(sr.y_test, pred)
                disp.ax_.set_title("Confusion matrix")
                fig = disp.figure_
                fig.tight_layout()
                save_figure(fig, "clf_confusion_matrix.png", cfg); plt.close(fig)
                fn = _feature_names(pipe.named_steps["pre"])
                fig2 = _plot_importances(pipe.named_steps["model"], fn, "Classifier feature importances")
                if fig2:
                    save_figure(fig2, "clf_feature_importances.png", cfg); plt.close(fig2)
        except Exception as e:
            log.warning("Plotting skipped (classification): %s", e)

    log.info("Classification metrics: %s", json.dumps(metrics))
    return pipe, out

def train_all(
    cfg: Union[Config, str],
    *,
    reg_estimator: EstimatorSpec = "rf",
    clf_estimator: EstimatorSpec = "rf",
    reg_params: Optional[Dict[str, Any]] = None,
    clf_params: Optional[Dict[str, Any]] = None,
    save_artifacts: bool = True,
):
    reg = train_regression(cfg, estimator=reg_estimator, params=reg_params, save_artifacts=save_artifacts)
    clf = train_classification(cfg, estimator=clf_estimator, params=clf_params, save_artifacts=save_artifacts)
    return {"regression": reg, "classification": clf}

# -------------------------------- CLI wrapper --------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Train models; estimator can be alias, dotted path, class, or instance (via notebook).")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--task", choices=["all", "regression", "classification"], default="all")
    ap.add_argument("--reg-estimator", help="alias or dotted path for regression (e.g., rf, sklearn.linear_model.LinearRegression)")
    ap.add_argument("--clf-estimator", help="alias or dotted path for classification (e.g., rf, logreg)")
    ap.add_argument("--reg-params", help='JSON dict for regression params')
    ap.add_argument("--clf-params", help='JSON dict for classification params')
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger("train", cfg=cfg, to_file=True)

    if args.no_figures:
        cfg.d.setdefault("evaluation", {})
        cfg.d["evaluation"]["save_plots"] = False

    try:
        out: Dict[str, Any] = {}
        if args.task in ("all", "regression"):
            params = json.loads(args.reg_params) if args.reg_params else None
            est = args.reg_estimator or cfg.get("models.regression.estimator", "rf")
            _, o = train_regression(cfg, estimator=est, params=params, save_artifacts=True)
            out["regression"] = o
        if args.task in ("all", "classification"):
            params = json.loads(args.clf_params) if args.clf_params else None
            est = args.clf_estimator or cfg.get("models.classification.estimator", "rf")
            _, o = train_classification(cfg, estimator=est, params=params, save_artifacts=True)
            out["classification"] = o

        save_json(out, "metrics.json", cfg, subdir="reports")
        log.info("Training complete.")
        return 0
    except Exception as e:
        log.exception("Training failed: %s", e)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

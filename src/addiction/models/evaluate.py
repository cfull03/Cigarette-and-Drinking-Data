# file: src/addiction/models/evaluate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ..utilities.config import Config, load_config
from ..utilities.io import save_figure, save_json, load_model
from ..utilities.logging import get_logger
from ..data.loaders import load_raw, make_splits_from_config


# ------------------------------ helpers ------------------------------

def _model_paths(cfg: Config) -> Tuple[Path, Path]:
    run = str(cfg.get("project.run_name", "run"))
    reg = (cfg.paths.models_dir / f"{run}_reg.joblib").resolve()
    clf = (cfg.paths.models_dir / f"{run}_clf.joblib").resolve()
    return reg, clf


def _plot_reg_scatter(y_true: pd.Series, y_pred: np.ndarray, title: str) -> plt.Figure:
    # Why: simple quality visualization
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi])
    ax.set_title(title)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    return fig


# ------------------------------ evaluation ------------------------------

def eval_regression_full(cfg: Config, logger) -> Dict[str, Any]:
    reg_path, _ = _model_paths(cfg)
    if not reg_path.exists():
        logger.info("No regression model found at %s; skipping.", reg_path)
        return {}

    model = load_model(reg_path, cfg)
    df = load_raw(cfg)
    target = str(cfg.get("data.target_regression"))
    if target not in df.columns:
        logger.info("Regression target '%s' not in data; skipping.", target)
        return {}

    data = df.dropna(subset=[target])
    X, y = data.drop(columns=[target]), data[target]
    pred = model.predict(X)

    metrics = {
        "MAE": float(mean_absolute_error(y, pred)),
        "MSE": float(mean_squared_error(y, pred)),
        "R2": float(r2_score(y, pred)),
        "n": int(len(y)),
    }

    save_json({"regression": {"metrics": metrics}}, "evaluation_regression.json", cfg, subdir="reports")

    try:
        if cfg.get("evaluation.save_plots", True):
            fig = _plot_reg_scatter(y, pred, f"{target}: True vs Pred (full)")
            save_figure(fig, "eval_reg_true_vs_pred.png", cfg)
            plt.close(fig)
    except Exception as e:
        logger.warning("Regression plotting skipped: %s", e)

    logger.info("Regression evaluation: %s", json.dumps(metrics))
    return metrics


def eval_classification_full(cfg: Config, logger) -> Dict[str, Any]:
    _, clf_path = _model_paths(cfg)
    if not clf_path.exists():
        logger.info("No classification model found at %s; skipping.", clf_path)
        return {}

    model = load_model(clf_path, cfg)
    df = load_raw(cfg)
    target = str(cfg.get("data.target_classification"))
    if target not in df.columns:
        logger.info("Classification target '%s' not in data; skipping.", target)
        return {}

    data = df.dropna(subset=[target])
    X, y = data.drop(columns=[target]), data[target]
    pred = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_weighted": float(f1_score(y, pred, average="weighted")),
        "n": int(len(y)),
    }
    report = classification_report(y, pred, output_dict=True)

    save_json({"classification": {"metrics": metrics, "report": report}}, "evaluation_classification.json", cfg, subdir="reports")

    try:
        if cfg.get("evaluation.save_plots", True):
            disp = ConfusionMatrixDisplay.from_predictions(y, pred)
            disp.ax_.set_title("Confusion matrix (full)")
            fig = disp.figure_
            fig.tight_layout()
            save_figure(fig, "eval_clf_confusion_matrix.png", cfg)
            plt.close(fig)
    except Exception as e:
        logger.warning("Classification plotting skipped: %s", e)

    logger.info("Classification evaluation: %s", json.dumps(metrics))
    return {"metrics": metrics, "report": report}


def eval_on_splits(cfg: Config, logger) -> Dict[str, Any]:
    # Why: optionally evaluate on fresh train/test splits for comparability.
    out: Dict[str, Any] = {}
    reg_path, clf_path = _model_paths(cfg)
    splits = make_splits_from_config(cfg)

    if reg_path.exists() and "regression" in splits:
        m = load_model(reg_path, cfg)
        sr = splits["regression"]
        pr = m.predict(sr.X_test)
        out["regression"] = {
            "MAE": float(mean_absolute_error(sr.y_test, pr)),
            "MSE": float(mean_squared_error(sr.y_test, pr)),
            "R2": float(r2_score(sr.y_test, pr)),
            "n": int(len(sr.y_test)),
        }

    if clf_path.exists() and "classification" in splits:
        m = load_model(clf_path, cfg)
        sc = splits["classification"]
        pc = m.predict(sc.X_test)
        out["classification"] = {
            "accuracy": float(accuracy_score(sc.y_test, pc)),
            "f1_weighted": float(f1_score(sc.y_test, pc, average="weighted")),
            "n": int(len(sc.y_test)),
            "report": classification_report(sc.y_test, pc, output_dict=True),
        }

    if out:
        save_json({"splits": out}, "evaluation_splits.json", cfg, subdir="reports")
        logger.info("Split-based evaluation: %s", json.dumps(out))
    return out


# ---------------------------------- CLI ----------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate saved models on full dataset or fresh splits.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--task", choices=["all", "regression", "classification"], default="all")
    ap.add_argument("--mode", choices=["full", "splits"], default="full", help="Evaluate on full dataset targets or on fresh train/test splits")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger("evaluate", cfg=cfg, to_file=True)

    try:
        summary: Dict[str, Any] = {}
        if args.mode == "full":
            if args.task in ("all", "regression"):
                summary["regression"] = eval_regression_full(cfg, log)
            if args.task in ("all", "classification"):
                summary["classification"] = eval_classification_full(cfg, log)
            save_json({"full": summary}, "evaluation.json", cfg, subdir="reports")
        else:
            summary = eval_on_splits(cfg, log)
            # eval_on_splits already saved evaluation_splits.json
        log.info("Evaluation complete.")
        return 0
    except Exception as e:
        log.exception("Evaluation failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# File: src/addiction_ds/evaluate.py
# Purpose: Evaluate the latest trained sklearn pipeline on a CSV and
#          write metrics artifacts into ./reports.
#
# Usage (CLI):
#   python -m addiction_ds.evaluate \
#       --config configs/experiment.yaml \
#       --csv data/processed/val.csv            # optional; defaults to newest in data/processed
#       --model latest                          # optional
#       --reports-dir reports                   # optional
#       --threshold 0.5                         # optional

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# project IO helpers
from .io import load_cfg, load_model, get_paths  # type: ignore


def _newest_csv(directory: Path) -> Path:
    csvs = list(directory.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {directory}")
    return max(csvs, key=lambda p: p.stat().st_mtime)


essential_keys = ("features", "label")


def _scores_from_pipe(pipe: Any, X: pd.DataFrame) -> np.ndarray:
    model = getattr(pipe, "named_steps", {}).get("model", pipe)
    if hasattr(model, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]  # type: ignore[arg-type]
    if hasattr(model, "decision_function"):
        s = np.asarray(pipe.decision_function(X), dtype=float)  # type: ignore[arg-type]
        # min-max normalize for AUC parity
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
    raise RuntimeError("Estimator provides neither predict_proba nor decision_function.")


def evaluate_on_csv(
    config_path: str = "configs/experiment.yaml",
    csv_path: str | None = None,
    model_name: str = "latest",
    reports_dir: str = "reports",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    cfg = load_cfg(config_path)
    paths = get_paths(cfg)

    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    if csv_path is None:
        csv_p = _newest_csv(processed_dir)
    else:
        csv_p = Path(csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_p}")

    pipe = load_model(cfg, name=model_name, framework="sklearn")

    label = cfg.get("label") or "is_smoker"
    feats_num = list(cfg["features"]["numeric"])  # type: ignore[index]
    feats_cat = list(cfg["features"]["categorical"])  # type: ignore[index]
    feat_cols = feats_num + feats_cat

    df = pd.read_csv(csv_p)
    missing = [c for c in feat_cols + [label] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {csv_p}: {missing}")

    X = df[feat_cols]
    y = df[label]

    scores = _scores_from_pipe(pipe, X)
    preds = (scores >= threshold).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y, scores)),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }

    bundle: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": type(getattr(pipe, "named_steps", {}).get("model", pipe)).__name__,
        "threshold": threshold,
        "csv": str(csv_p),
        "metrics": metrics,
        "classification_report": classification_report(y, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # write artifacts
    p_json = reports / f"metrics_{stamp}.json"
    p_json.write_text(json.dumps(bundle, indent=2))
    (reports / "metrics_latest.json").write_text(json.dumps(bundle, indent=2))

    pd.DataFrame([metrics]).assign(model=bundle["model"], timestamp=bundle["timestamp"]).to_csv(
        reports / f"metrics_{stamp}.csv", index=False
    )

    pd.DataFrame(bundle["confusion_matrix"], columns=["pred_0", "pred_1"], index=["true_0", "true_1"]).to_csv(
        reports / f"confusion_matrix_{stamp}.csv"
    )

    (reports / f"classification_report_{stamp}.txt").write_text(bundle["classification_report"])  # type: ignore[arg-type]

    return {
        **bundle,
        "artifacts": {
            "json": str(p_json),
            "csv": str(reports / f"metrics_{stamp}.csv"),
            "cm_csv": str(reports / f"confusion_matrix_{stamp}.csv"),
            "report_txt": str(reports / f"classification_report_{stamp}.txt"),
        },
    }


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Evaluate trained pipeline and write reports")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--model", default="latest")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    res = evaluate_on_csv(
        config_path=args.config,
        csv_path=args.csv,
        model_name=args.model,
        reports_dir=args.reports_dir,
        threshold=args.threshold,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

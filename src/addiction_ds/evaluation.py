# File: src/addiction_ds/evaluate.py
"""Evaluate a trained sklearn pipeline on a CSV and write metrics artifacts.

- Auto-discovers latest CSV in processed dir when `--csv` omitted.
- Loads model by name (default: "latest") using project IO helpers.
- Writes JSON/CSV/TXT artifacts into a reports directory.
- Robust to config variations (processed vs processed_dir keys).
- Safe metrics when a class is missing (ROC AUC -> NaN instead of crash).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
from .io import get_paths, load_cfg, load_model  # type: ignore

__all__ = ["evaluate_on_csv", "cli"]


# ----------------------------- helpers --------------------------------------

def _newest_csv(directory: Path) -> Path:
    csvs = list(directory.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {directory}")
    return max(csvs, key=lambda p: p.stat().st_mtime)


def _scores_from_pipe(pipe: Any, X: pd.DataFrame) -> np.ndarray:
    model = getattr(pipe, "named_steps", {}).get("model", pipe)
    if hasattr(model, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]  # type: ignore[arg-type]
    if hasattr(model, "decision_function"):
        s = np.asarray(pipe.decision_function(X), dtype=float)  # type: ignore[arg-type]
        # min-max normalize for AUC parity
        s_min, s_max = float(np.min(s)), float(np.max(s))
        rng = s_max - s_min
        return (s - s_min) / (rng + 1e-12)
    raise RuntimeError("Estimator provides neither predict_proba nor decision_function.")


def _resolve_processed_dir(paths: dict[str, Any]) -> Path:
    # tolerate both keys from IO helper or raw config
    cand = (
        paths.get("processed_dir")
        or paths.get("processed")
        or "data/processed"
    )
    return Path(str(cand))


def _resolve_reports_dir(reports_dir: str | None, paths: dict[str, Any]) -> Path:
    if reports_dir:
        return Path(reports_dir)
    # optional: some configs may include reports_dir
    cand = paths.get("reports_dir") or "reports"
    return Path(str(cand))


# ------------------------------ core ----------------------------------------

def evaluate_on_csv(
    config_path: str = "configs/experiment.yaml",
    csv_path: str | None = None,
    model_name: str = "latest",
    reports_dir: str = "reports",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate model and persist metrics.

    Returns a dict containing metrics and artifact paths.
    """
    cfg = load_cfg(config_path)
    paths = get_paths(cfg)

    processed_dir = _resolve_processed_dir(paths)
    csv_p = Path(csv_path) if csv_path else _newest_csv(processed_dir)
    if not csv_p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_p}")

    pipe = load_model(cfg, name=model_name, framework="sklearn")

    # features/label
    label = (cfg.get("label") or "is_smoker")
    feats = cfg.get("features", {})
    feats_num = list(feats.get("numeric", []))
    feats_cat = list(feats.get("categorical", []))
    feat_cols = [*feats_num, *feats_cat]

    # read CSV (respect optional io.read_kwargs if present in merged cfg)
    read_kwargs = ((cfg.get("io") or {}).get("read_kwargs") or {}) if isinstance(cfg, dict) else {}
    df = pd.read_csv(csv_p, **read_kwargs)

    missing = [c for c in [*feat_cols, label] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {csv_p}: {missing}")

    X = df[feat_cols]
    y = df[label]

    # Convert boolean labels to ints for metrics consistency
    if y.dtype == bool:
        y = y.astype(int)

    scores = _scores_from_pipe(pipe, X)
    preds = (scores >= threshold).astype(int)

    # Metrics (safe roc-auc)
    try:
        roc = float(roc_auc_score(y, scores))
    except Exception:
        roc = float("nan")

    metrics = {
        "roc_auc": roc,
        "auc": roc,  # alias for compatibility with older checks
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }

    bundle: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": type(getattr(pipe, "named_steps", {}).get("model", pipe)).__name__,
        "threshold": threshold,
        "csv": str(csv_p),
        "metrics": metrics,
        "classification_report": classification_report(y, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }

    reports = _resolve_reports_dir(reports_dir, paths)
    reports.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # write artifacts
    p_json = reports / f"metrics_{stamp}.json"
    p_json.write_text(json.dumps(bundle, indent=2))
    (reports / "metrics_latest.json").write_text(json.dumps(bundle, indent=2))

    pd.DataFrame([metrics]).assign(model=bundle["model"], timestamp=bundle["timestamp"]).to_csv(
        reports / f"metrics_{stamp}.csv", index=False
    )

    pd.DataFrame(
        bundle["confusion_matrix"], columns=["pred_0", "pred_1"], index=["true_0", "true_1"]
    ).to_csv(reports / f"confusion_matrix_{stamp}.csv")

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


# ------------------------------- CLI ----------------------------------------

def cli(argv: list[str] | None = None) -> int:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Evaluate trained pipeline and write reports")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--model", default="latest")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    res = evaluate_on_csv(
        config_path=args.config,
        csv_path=args.csv,
        model_name=args.model,
        reports_dir=args.reports_dir,
        threshold=args.threshold,
    )
    print(json.dumps(res, indent=2))
    return 0


def main() -> None:  # pragma: no cover
    raise SystemExit(cli())


if __name__ == "__main__":  # pragma: no cover
    main()

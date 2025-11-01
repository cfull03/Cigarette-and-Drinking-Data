# file: src/addiction/models/predict.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utilities.config import Config, load_config
from ..utilities.io import load_model, to_interim, save_json
from ..utilities.logging import get_logger


def _model_paths(cfg: Config) -> Tuple[Path, Path]:
    run = str(cfg.get("project.run_name", "run"))
    reg = (cfg.paths.models_dir / f"{run}_reg.joblib").resolve()
    clf = (cfg.paths.models_dir / f"{run}_clf.joblib").resolve()
    return reg, clf


def _prepare_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Why: training pipelines expect same input columns (targets removed; id optional)
    drop_cols = []
    rid = cfg.get("data.id_column")
    if isinstance(rid, str) and rid in df.columns:
        drop_cols.append(rid)
    for key in ("data.target_regression", "data.target_classification"):
        tgt = cfg.get(key)
        if isinstance(tgt, str) and tgt in df.columns:
            drop_cols.append(tgt)
    return df.drop(columns=drop_cols, errors="ignore")


def predict_file(
    cfg: Config,
    *,
    input_csv: Path,
    kind: str = "both",  # "reg" | "clf" | "both"
    include_proba: bool = False,
    save_output: bool = True,
    output_name: Optional[str] = None,
) -> Dict[str, Any]:
    log = get_logger("predict", cfg=cfg, to_file=True)

    # Load models if available
    reg_path, clf_path = _model_paths(cfg)
    want_reg = kind in ("reg", "both")
    want_clf = kind in ("clf", "both")

    model_reg = load_model(reg_path, cfg) if (want_reg and reg_path.exists()) else None
    model_clf = load_model(clf_path, cfg) if (want_clf and clf_path.exists()) else None

    if want_reg and model_reg is None:
        log.warning("Regression model not found at %s; skipping.", reg_path)
    if want_clf and model_clf is None:
        log.warning("Classification model not found at %s; skipping.", clf_path)
    if (want_reg and model_reg is None) and (want_clf and model_clf is None):
        raise FileNotFoundError("No requested models were found. Train first.")

    # Load input
    df = pd.read_csv(input_csv)
    X = _prepare_features(df, cfg)

    out = df.copy()
    meta: Dict[str, Any] = {"input": str(input_csv), "rows": int(len(out)), "predictions": []}

    # Regression
    reg_tgt = cfg.get("data.target_regression")
    if model_reg is not None:
        yhat = model_reg.predict(X)
        col = f"pred_{reg_tgt or 'regression'}"
        out[col] = yhat
        meta["predictions"].append({"kind": "regression", "column": col, "model_path": str(reg_path)})

    # Classification
    clf_tgt = cfg.get("data.target_classification")
    if model_clf is not None:
        yhat = model_clf.predict(X)
        col = f"pred_{clf_tgt or 'classification'}"
        out[col] = yhat
        meta["predictions"].append({"kind": "classification", "column": col, "model_path": str(clf_path)})

        if include_proba and hasattr(model_clf, "predict_proba"):
            proba = model_clf.predict_proba(X)
            classes = getattr(model_clf, "classes_", None)
            if classes is None:
                classes = list(range(proba.shape[1]))
            for i, cls in enumerate(classes):
                out[f"proba_{clf_tgt}_{cls}" if clf_tgt else f"proba_cls_{cls}"] = proba[:, i]

    # Save outputs
    saved_path: Optional[Path] = None
    if save_output:
        name = output_name or "predictions.csv"
        saved_path = to_interim(out, name, cfg)
        meta["output_csv"] = str(saved_path)

    # Save summary json beside reports
    save_json(meta, "predict_summary.json", cfg, subdir="reports")
    return {"output_path": str(saved_path) if saved_path else None, "summary": meta}


# --------------------------------- CLI ---------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Batch predictions using saved models.")
    ap.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    ap.add_argument("--input", help="Input CSV path; defaults to config.paths.raw_csv")
    ap.add_argument("--kind", choices=["reg", "clf", "both"], default="both", help="Which model(s) to apply")
    ap.add_argument("--proba", action="store_true", help="Include classification probabilities when available")
    ap.add_argument("--no-save", action="store_true", help="Do not write predictions CSV; print summary only")
    ap.add_argument("--output-name", help="Base filename for predictions (timestamp auto-appended)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger("predict.cli", cfg=cfg, to_file=True)

    try:
        input_csv = Path(args.input) if args.input else cfg.paths.raw_csv
        res = predict_file(
            cfg,
            input_csv=input_csv,
            kind=args.kind,
            include_proba=args.proba,
            save_output=not args.no_save,
            output_name=args.output_name,
        )
        log.info("Prediction complete: %s", json.dumps(res["summary"]))
        if res["output_path"]:
            print(res["output_path"])
        return 0
    except Exception as e:
        log.exception("Prediction failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

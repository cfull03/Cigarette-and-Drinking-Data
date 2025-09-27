
# file: tests/test_train_predict_evaluate.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from addiction.utilities.config import load_config
from addiction.utilities.logging import get_logger
from addiction.models.train import train_all
from addiction.models.evaluate import eval_on_splits
from addiction.models.predict import predict_file

CFG_TMPL = """\
project:
  run_name: "testrun"
  seed: 123
paths:
  raw_csv: "{raw}"
  processed_dir: "{root}/data/processed"
  interim_dir: "{root}/data/interm"
  models_dir: "{root}/models"
  reports_dir: "{root}/reports"
  figures_dir: "{root}/reports/figures"
data:
  id_column: "id"
  target_regression: "smokes_per_day"
  target_classification: "has_health_issues"
  test_size: 0.25
  stratify_classification: true
preprocessing:
  numeric:
    imputer: {{"strategy": "median"}}
    scaler: {{"standardize": true, "with_mean": false}}
  categorical:
    imputer: {{"strategy": "most_frequent"}}
    one_hot: {{"handle_unknown": "ignore"}}
models:
  regression:
    estimator: "rf"
    params: {{"n_estimators": 10, "random_state": 0}}
  classification:
    estimator: "rf"
    params: {{"n_estimators": 10, "random_state": 0}}
"""

def make_synth(tmp_path: Path):
    n = 120
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "id": np.arange(n),
        "age": rng.integers(18, 70, size=n),
        "bmi": rng.normal(27, 4, size=n),
        "gender": rng.choice(["Male","Female"], size=n),
        "city": rng.choice(["A","B","C"], size=n),
        "smokes_per_day": rng.integers(0, 20, size=n),
        "has_health_issues": rng.choice([0,1], size=n, p=[0.6, 0.4]),
    })
    raw = tmp_path / "data" / "raw" / "raw.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw, index=False)
    return raw

def make_cfg(tmp_path: Path, raw: Path):
    cfg_path = tmp_path / "config" / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_text = CFG_TMPL.format(root=tmp_path.as_posix(), raw=raw.as_posix())
    cfg_path.write_text(cfg_text, encoding="utf-8")
    return load_config(cfg_path)

def test_train_eval_predict_flow(tmp_path: Path):
    raw = make_synth(tmp_path)
    cfg = make_cfg(tmp_path, raw)

    # Train both models (fast, tiny forests)
    out = train_all(cfg, save_artifacts=True)
    # models saved
    reg_model = (cfg.paths.models_dir / f"{cfg.get('project.run_name')}_reg.joblib")
    clf_model = (cfg.paths.models_dir / f"{cfg.get('project.run_name')}_clf.joblib")
    assert reg_model.exists() and clf_model.exists()

    # Evaluate on splits
    eval_out = eval_on_splits(cfg, logger=get_logger("tmp", cfg=cfg, to_file=False))
    assert "regression" in eval_out and "classification" in eval_out

    # Predict on raw
    res = predict_file(cfg, input_csv=raw, kind="both", include_proba=False, save_output=True, output_name="pred.csv")
    assert res["output_path"] is not None
    pred_csv = Path(res["output_path"])
    assert pred_csv.exists()
    # Check prediction columns present
    pdf = pd.read_csv(pred_csv)
    assert any(c.startswith("pred_smokes_per_day") for c in pdf.columns)
    assert any(c.startswith("pred_has_health_issues") for c in pdf.columns)
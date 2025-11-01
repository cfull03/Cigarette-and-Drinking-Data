# file: tests/test_pipelines.py
import pandas as pd
from pathlib import Path
from addiction.utilities.config import load_config
from addiction.features.pipelines import build_preprocessor_from_config

def make_cfg(tmp_path: Path):
    cfg_path = tmp_path / "config" / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(f"""
project: {{ run_name: "testrun" }}
paths:
  raw_csv: "{(tmp_path/'data'/'raw'/'raw.csv').as_posix()}"
  processed_dir: "{(tmp_path/'data'/'processed').as_posix()}"
  interim_dir: "{(tmp_path/'data'/'interm').as_posix()}"
  models_dir: "{(tmp_path/'models').as_posix()}"
  reports_dir: "{(tmp_path/'reports').as_posix()}"
  figures_dir: "{(tmp_path/'reports'/'figures').as_posix()}"
preprocessing:
  numeric:
    imputer: {{"strategy": "median"}}
    scaler: {{"standardize": true, "with_mean": false}}
  categorical:
    imputer: {{"strategy": "most_frequent"}}
    one_hot: {{"handle_unknown": "ignore"}}
""", encoding="utf-8")
    return load_config(cfg_path)

def test_build_preprocessor_from_config(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    df = pd.DataFrame({
        "age": [30, 40, None],
        "bmi": [25.1, 30.0, 27.5],
        "gender": ["Male", "Female", None],
        "city": ["A", "B", "C"],
    })
    pre, num, cat = build_preprocessor_from_config(cfg, df)
    assert set(num).issubset(df.columns)
    assert set(cat).issubset(df.columns)
    Xt = pre.fit_transform(df)
    # Should be 3 rows
    assert Xt.shape[0] == 3
# file: tests/test_io.py
import re
import joblib
import pandas as pd
from pathlib import Path
from addiction.utilities.config import load_config
from addiction.utilities.io import to_interim, save_model, load_model, save_json, save_text

def make_cfg(tmp_path: Path):
    cfg_path = tmp_path / "config" / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(f"""
project:
  run_name: "testrun"
paths:
  raw_csv: "{(tmp_path/'data'/'raw'/'raw.csv').as_posix()}"
  processed_dir: "{(tmp_path/'data'/'processed').as_posix()}"
  interim_dir: "{(tmp_path/'data'/'interm').as_posix()}"
  models_dir: "{(tmp_path/'models').as_posix()}"
  reports_dir: "{(tmp_path/'reports').as_posix()}"
  figures_dir: "{(tmp_path/'reports'/'figures').as_posix()}"
""", encoding="utf-8")
    return load_config(cfg_path)

def test_to_interim_and_artifacts(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    out = to_interim(df, "small.csv", cfg)
    assert out.exists()
    assert out.parent == cfg.paths.interim_dir
    # name has timestamp suffix
    assert re.match(r"^small_\d{8}-\d{6}\.csv$", out.name)

    # model round-trip
    model_obj = {"hello": "world"}
    mpath = save_model(model_obj, "toy.joblib", cfg)
    assert mpath.exists()
    loaded = load_model("toy.joblib", cfg)
    assert loaded == model_obj

    # json/text
    jpath = save_json({"ok": True}, "x.json", cfg, subdir="reports")
    tpath = save_text("hi", "y.txt", cfg, subdir="reports")
    assert jpath.exists() and tpath.exists()
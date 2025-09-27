# file: tests/test_config.py
import json
from pathlib import Path

from addiction.utilities.config import load_config

CFG_TEXT = """\
project:
  name: "test-proj"
  run_name: "testrun"
paths:
  raw_csv: "{root}/data/raw/raw.csv"
  processed_dir: "{root}/data/processed"
  interim_dir: "{root}/data/interm"
  models_dir: "{root}/models"
  reports_dir: "{root}/reports"
  figures_dir: "{root}/reports/figures"
"""

def test_load_config_paths(tmp_path: Path):
    cfg_path = tmp_path / "config" / "config.yaml"
    (tmp_path / "config").mkdir(parents=True)
    # Fill placeholders
    cfg_path.write_text(CFG_TEXT.replace("{root}", str(tmp_path)), encoding="utf-8")

    cfg = load_config(cfg_path)
    # Paths resolve under tmp root
    assert cfg.paths.raw_csv.as_posix().startswith(tmp_path.as_posix())
    # Ensure dirs helper works
    cfg.paths.ensure_dirs()
    assert cfg.paths.processed_dir.exists()
    assert cfg.paths.interim_dir.exists()
    assert cfg.paths.reports_dir.exists()
    assert cfg.paths.figures_dir.exists()
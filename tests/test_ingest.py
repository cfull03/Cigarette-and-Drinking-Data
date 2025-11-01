# file: tests/test_ingest.py
from pathlib import Path
import pandas as pd
import yaml
from addiction.utilities.config import load_config
from addiction.data.ingest import scan_interim

SCHEMA = {
  "primary_key": ["id"],
  "fields": [
    {"name":"id","type":"integer","nullable": False},
    {"name":"age","type":"integer","nullable": False, "stats":{"min": 18, "max": 120}},
    {"name":"gender","type":"string","nullable": False, "enum": ["Male","Female"]},
    {"name":"smokes_per_day","type":"integer","nullable": False, "stats":{"min": 0, "max": 40}},
  ]
}

CFG_TMPL = """\
project: {{ run_name: "ingest" }}
paths:
  raw_csv: "{raw}"
  processed_dir: "{root}/data/processed"
  interim_dir: "{root}/data/interm"
  models_dir: "{root}/models"
  reports_dir: "{root}/reports"
  figures_dir: "{root}/reports/figures"
"""

def setup_ingest_env(tmp_path: Path):
    # config
    cfg_path = tmp_path / "config" / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(CFG_TMPL.format(root=tmp_path.as_posix(), raw=(tmp_path/'data'/'raw'/'raw.csv').as_posix()), encoding="utf-8")
    # schema
    schema_path = tmp_path / "config" / "schema.yaml"
    yaml.safe_dump(SCHEMA, schema_path.open("w", encoding="utf-8"))
    # interim file with some bad rows
    interm = tmp_path / "data" / "interm"
    interm.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "id":[1,2,2,4,5],
        "age":[25, 17, 30, 200, 40],  # 17<min, 200>max
        "gender":["Male", "Alien", "Female", "Female", "Male"],  # "Alien" invalid enum
        "smokes_per_day":[5, 3, 2, 1, 0],
    })
    src = interm / "incoming.csv"
    df.to_csv(src, index=False)
    return cfg_path, schema_path, src

def test_scan_interim_with_quarantine(tmp_path: Path):
    cfg_path, schema_path, src = setup_ingest_env(tmp_path)
    cfg = load_config(cfg_path)
    out = scan_interim(cfg, schema_path=schema_path)
    # Processed at least our file
    assert len(out) >= 1
    # Marker exists
    marker = src.with_suffix(src.suffix + ".done.json")
    assert marker.exists()
    mark = json.loads(marker.read_text())
    # Cleaned output exists
    clean_path = Path(mark["output_clean"])
    assert clean_path.exists()
    # Rejects saved due to enum and range failures + PK dupes
    assert mark["output_rejects"] is not None
    assert Path(mark["output_rejects"]).exists()
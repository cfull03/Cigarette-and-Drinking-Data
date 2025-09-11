# File: tests/test_validate_ok.py
"""Minimal-but-solid tests for the validator.

Covers:
- Pure function API: validate_df (pass and fail cases)
- CLI: exit code 0 on pass, 2 on failure

These tests do NOT depend on your repo's real data; they build temp CSV/CFG/SCHEMA.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd

from addiction_ds.validate import validate_df


def test_validate_df_pass_basic():
    schema = {
        "columns": {
            "id": {"type": "integer", "required": True, "unique": True},
            "group": {"type": "string", "allowed": ["a", "b"]},
            "score": {"type": "number", "min": 0, "max": 100},
        },
        "primary_key": ["id"],
        "min_rows": 1,
    }
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "group": ["a", "b", "a"],
            "score": [10.0, 90.5, 42.0],
        }
    )
    ok, errs = validate_df(df, schema)
    assert ok, f"expected pass, got: {errs}"


def test_validate_df_fail_missing_and_duplicates():
    schema = {
        "columns": {
            "id": {"type": "integer", "required": True},
            "name": {"type": "string", "required": True},
        },
        "primary_key": ["id"],
        "min_rows": 1,
    }
    df = pd.DataFrame(
        {
            "id": [1, 1, None],  # duplicate + null in PK
            # missing required column "name"
        }
    )
    ok, errs = validate_df(df, schema)
    assert not ok
    assert any("primary_key contains duplicates" in e for e in errs) or any(
        "duplicates" in e for e in errs
    )
    assert any("primary_key contains nulls" in e for e in errs) or any("null" in e for e in errs)
    assert any("Missing required column: name" in e for e in errs)


def _write_yaml(p: Path, data: str) -> None:
    p.write_text(data, encoding="utf-8")


def _write_csv(p: Path, rows):
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


def test_cli_exit_codes(tmp_path: Path):
    # files
    cfg = tmp_path / "configs" / "default.yaml"
    sch = tmp_path / "configs" / "schema.yaml"
    data_ok = tmp_path / "data" / "raw" / "ok.csv"
    data_bad = tmp_path / "data" / "raw" / "bad.csv"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    data_ok.parent.mkdir(parents=True, exist_ok=True)

    # simple schema: id int PK, group in {a,b}
    _write_yaml(
        sch,
        """
columns:
  id:
    type: integer
    required: true
  group:
    type: string
    allowed: [a, b]
primary_key: [id]
min_rows: 1
        """.strip(),
    )

    # config prefers sample but also carries raw path (we'll pass --input explicitly anyway)
    _write_yaml(
        cfg,
        f"""
project_name: test
paths:
  raw: {data_ok}
  sample_input: {tmp_path / 'data' / 'sample' / 'sample.csv'}
  schema: {sch}
        """.strip(),
    )

    # good CSV
    _write_csv(data_ok, [["id", "group"], [1, "a"], [2, "b"]])
    # bad CSV (duplicate id, group not allowed)
    _write_csv(data_bad, [["id", "group"], [1, "x"], [1, "a"]])

    py = sys.executable
    mod = "addiction_ds.validate"

    # OK run → exit 0
    r_ok = subprocess.run(
        [py, "-m", mod, "--config", str(cfg), "--schema", str(sch), "--input", str(data_ok)],
        cwd=tmp_path,
    )
    assert r_ok.returncode == 0

    # BAD run → exit 2
    r_bad = subprocess.run(
        [py, "-m", mod, "--config", str(cfg), "--schema", str(sch), "--input", str(data_bad)],
        cwd=tmp_path,
    )
    assert r_bad.returncode == 2

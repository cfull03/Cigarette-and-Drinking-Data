# File: tests/test_validate_ok.py
import yaml
import pandas as pd
from addiction_ds.validate import validate_df


def _synth_row(schema: dict) -> pd.DataFrame:
    cols = schema.get("columns", {})
    row = {}
    for name, spec in cols.items():
        t = str(spec.get("type", "string")).lower()
        if t in {"int", "integer"}:
            row[name] = int(max(1, spec.get("min", 1)))
        elif t in {"float", "double", "number", "numeric"}:
            row[name] = float(spec.get("min", 0.0))
        elif t in {"bool", "boolean"}:
            row[name] = True
        elif t in {"date", "datetime", "datetime64"}:
            row[name] = "2020-01-01"
        else:
            allowed = spec.get("allowed")
            row[name] = (allowed[0] if isinstance(allowed, list) and allowed else "ok")
    return pd.DataFrame([row])


def test_validate_schema_roundtrip():
    with open("configs/schema.yaml", "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    df = _synth_row(schema)
    ok, errs = validate_df(df, schema)
    assert ok, f"Expected valid row, got errors: {errs}"
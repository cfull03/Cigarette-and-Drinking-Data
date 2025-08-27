#!/usr/bin/env python3
"""
Validate a CSV against a YAML schema (cleaned-data contract).
Exit code: 0 = OK, 2 = failed validation.

Schema keys supported (top-level):
- strict: bool
- allow_extra: bool
- primary_key: str
- unique: [str]
- min_rows: int
- columns: { <name>: {type, required, unique, min, max, allowed, regex, max_len, format} }

Why: enforce a stable shape for cleaned data before training.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml


# ----------------------------- schema I/O ------------------------------------
def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError("Schema root must be a mapping (YAML object)")
    return data


# ----------------------------- checks ----------------------------------------
def _check_series_type(s: pd.Series, spec: Dict[str, Any]) -> List[str]:
    """Return list of errors for a single column against spec.
    Only implement constraints we actually use; keep it simple and readable.
    """
    errs: List[str] = []
    t = (spec.get("type") or "").lower()

    if spec.get("required") and s.isna().any():
        errs.append(f"{int(s.isna().sum())} missing values where required")

    if spec.get("unique") and s.duplicated().any():
        errs.append(f"{int(s.duplicated().sum())} duplicate values present")

    if t in {"integer", "number"}:
        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.isna().all() and s.notna().any():
            errs.append("cannot coerce to numeric")
        if t == "integer":
            non_int_mask = (~coerced.isna()) & (coerced % 1 != 0)
            if non_int_mask.any():
                errs.append(f"{int(non_int_mask.sum())} values are non-integers")
        mn, mx = spec.get("min"), spec.get("max")
        if mn is not None and (coerced.dropna() < mn).any():
            errs.append(f"values < min {mn}")
        if mx is not None and (coerced.dropna() > mx).any():
            errs.append(f"values > max {mx}")

    elif t == "date":
        fmt = spec.get("format")
        parsed = pd.to_datetime(s, errors="coerce", format=fmt)
        if parsed.isna().all() and s.notna().any():
            errs.append("cannot parse dates with given format")
        mn, mx = spec.get("min"), spec.get("max")
        if mn and (parsed.dropna() < pd.to_datetime(mn)).any():
            errs.append(f"dates < min {mn}")
        if mx and (parsed.dropna() > pd.to_datetime(mx)).any():
            errs.append(f"dates > max {mx}")

    elif t in {"string", "category"}:
        allowed = spec.get("allowed")
        if allowed is not None:
            vals = s.dropna().astype(str)
            bad = vals[~vals.isin([str(a) for a in allowed])]
            if not bad.empty:
                errs.append(f"{bad.size} values not in allowed set")
        regex = spec.get("regex")
        if regex:
            vals = s.dropna().astype(str)
            mask = vals.str.match(re.compile(regex))
            if (~mask).any():
                errs.append(f"{int((~mask).sum())} values fail regex {regex}")

    max_len = spec.get("max_len")
    if max_len is not None:
        vals = s.dropna().astype(str)
        too_long = vals.str.len() > int(max_len)
        if too_long.any():
            errs.append(f"{int(too_long.sum())} values exceed max_len {max_len}")

    return errs


def validate_df(df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    spec_cols: Dict[str, Any] = schema.get("columns", {})
    strict = bool(schema.get("strict", False))
    allow_extra = bool(schema.get("allow_extra", False))

    # min_rows
    min_rows = schema.get("min_rows")
    if isinstance(min_rows, int) and len(df) < min_rows:
        errors.append(f"row count {len(df)} < min_rows {min_rows}")

    # required columns present
    for col, col_spec in spec_cols.items():
        if col_spec.get("required") and col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # no extras when strict and not allow_extra
    if strict and not allow_extra:
        extras = [c for c in df.columns if c not in spec_cols]
        if extras:
            errors.append(f"Unexpected columns present: {sorted(extras)}")

    # per-column checks
    for col, col_spec in spec_cols.items():
        if col not in df.columns:
            continue
        col_errs = _check_series_type(df[col], col_spec)
        if col_errs:
            errors.append(f"[{col}] " + "; ".join(col_errs))

    # table-level uniqueness / PK
    pk = schema.get("primary_key")
    if pk and pk in df.columns:
        if df[pk].isna().any() or df[pk].duplicated().any():
            errors.append(f"Primary key '{pk}' has missing or duplicate values")

    for ucol in schema.get("unique", []) or []:
        if ucol in df.columns and df[ucol].duplicated().any():
            errors.append(f"Column '{ucol}' must be unique")

    return (len(errors) == 0), errors


# ----------------------------- CLI -------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate a CSV against a YAML schema")
    p.add_argument("--schema", default="configs/schema.yaml", help="Path to YAML schema")
    p.add_argument("--csv", required=True, help="Path to CSV to validate")
    p.add_argument("--sep", default=",", help="CSV delimiter (default ',')")
    p.add_argument("--encoding", default="utf-8", help="CSV encoding")
    p.add_argument("--quiet", action="store_true", help="Print only pass/fail")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    schema = load_schema(Path(args.schema))
    df = pd.read_csv(args.csv, sep=args.sep, encoding=args.encoding)

    ok, errs = validate_df(df, schema)
    if ok:
        if not args.quiet:
            print("✅ Validation passed")
        return 0

    print("❌ Validation failed:")
    for e in errs:
        print(" -", e)
    return 2


# if __name__ == "__main__":
    raise SystemExit(main())

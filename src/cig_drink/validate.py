#!/usr/bin/env python3
"""Data validation CLI for CSVs against a YAML schema.

Exit codes:
  0 = OK
  2 = validation failed

Schema (YAML) supported keys:
- strict: bool                       # if true, reject any columns not in schema
- allow_extra: bool                  # if true, permit extra columns even when strict is false
- primary_key: str | list[str]       # PK columns must be present, non-null, unique as a tuple
- unique: list[str]                  # columns that must be globally unique
- min_rows: int                      # minimum required rows
- columns:                           # per-column rules
    <name>:
      type: integer|number|string|boolean|datetime|category
      required: bool
      unique: bool
      min: number                    # for numeric
      max: number                    # for numeric
      allowed: [..]                  # for categorical/string
      regex: str                     # pattern for string content
      max_len: int                   # max string length
      format: datetime|date          # attempt parse with pandas.to_datetime

Why: enforce stable, explicit contracts before modeling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import re
import sys
from typing import Any, Iterable

import pandas as pd
import pandas.api.types as pat
import yaml


# ----------------------------- helpers -------------------------------------


def load_yaml(path: Path) -> dict:
    """Load YAML safely; return {} on empty."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def choose_input_path(cfg: dict, cli_input: Path | None) -> Path:
    """Pick input CSV based on CLI → config.paths.sample_input → config.paths.raw."""
    if cli_input is not None:
        return cli_input
    paths = (cfg or {}).get("paths", {})
    for key in ("sample_input", "raw"):
        p = paths.get(key)
        if p and Path(p).exists():
            return Path(p)
    raise FileNotFoundError(
        "No usable input CSV. Provide --input or set paths.sample_input/raw in configs/default.yaml."
    )


# ----------------------------- type checks ----------------------------------


def _dtype_ok(series: pd.Series, expected: str) -> bool:
    e = expected.lower()
    if e in {"int", "integer"}:
        return pat.is_integer_dtype(series) or pat.is_bool_dtype(series)  # bool is subclass of int
    if e in {"float", "double", "number", "numeric"}:
        return pat.is_numeric_dtype(series)
    if e in {"str", "string"}:
        return pat.is_string_dtype(series) or series.dtype == "object"
    if e in {"bool", "boolean"}:
        return pat.is_bool_dtype(series)
    if e in {"date", "datetime", "datetime64"}:
        return pat.is_datetime64_any_dtype(series)
    if e in {"category", "categorical"}:
        return pat.is_categorical_dtype(series) or pat.is_object_dtype(series)
    return True  # unknown label: don't block


@dataclass
class ColumnIssue:
    column: str
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.column}: {self.message}"


# ----------------------------- validators -----------------------------------


def _validate_column(name: str, s: pd.Series, spec: dict) -> list[ColumnIssue]:
    errs: list[ColumnIssue] = []
    t = str(spec.get("type", "")).strip().lower()

    # required
    if spec.get("required") and s.isna().any():
        errs.append(ColumnIssue(name, f"{int(s.isna().sum())} missing values where required"))

    # unique
    if spec.get("unique") and s.duplicated().any():
        errs.append(ColumnIssue(name, f"{int(s.duplicated().sum())} duplicate values present"))

    # type enforcement (non-destructive)
    if t and not _dtype_ok(s, t):
        # Try a safe coercion to see if the data is salvageable
        if t in {"integer", "int", "number", "numeric", "float", "double"}:
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.isna().all() and s.notna().any():
                errs.append(ColumnIssue(name, "cannot coerce to numeric"))
            if t in {"integer", "int"}:
                non_int_mask = (~coerced.isna()) & (coerced % 1 != 0)
                if non_int_mask.any():
                    errs.append(ColumnIssue(name, f"{int(non_int_mask.sum())} values are non-integers"))
        elif t in {"datetime", "date", "datetime64"}:
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.isna().all() and s.notna().any():
                errs.append(ColumnIssue(name, "cannot parse as datetime"))
        # For string/category we rely on allowed/regex checks below

    # numeric range
    if t in {"integer", "int", "float", "double", "number", "numeric"} and pat.is_numeric_dtype(s):
        if "min" in spec:
            below = s.dropna() < spec["min"]
            if below.any():
                errs.append(ColumnIssue(name, f"{int(below.sum())} values below min {spec['min']}"))
        if "max" in spec:
            above = s.dropna() > spec["max"]
            if above.any():
                errs.append(ColumnIssue(name, f"{int(above.sum())} values above max {spec['max']}"))

    # string length
    if t in {"str", "string"}:
        max_len = spec.get("max_len")
        if isinstance(max_len, int):
            too_long = s.dropna().astype(str).map(len) > max_len
            if too_long.any():
                errs.append(ColumnIssue(name, f"{int(too_long.sum())} strings exceed max_len {max_len}"))

    # regex
    regex = spec.get("regex")
    if regex:
        patt = re.compile(str(regex))
        bad = s.dropna().astype(str).map(lambda v: not bool(patt.fullmatch(v)))
        if bad.any():
            errs.append(ColumnIssue(name, f"{int(bad.sum())} values fail regex {regex!r}"))

    # allowed set (categorical/string)
    allowed = spec.get("allowed")
    if allowed:
        invalid = set(pd.Series(s.dropna().unique()).astype(str)) - {str(x) for x in allowed}
        if invalid:
            errs.append(ColumnIssue(name, f"unexpected categories {sorted(invalid)}"))

    return errs


def validate_df(df: pd.DataFrame, schema: dict) -> tuple[bool, list[str]]:
    """Apply global + per-column rules; return (ok, errors)."""
    errors: list[str] = []

    strict = bool(schema.get("strict", False))
    allow_extra = bool(schema.get("allow_extra", True))

    spec_cols: dict[str, dict] = schema.get("columns", {}) or {}

    # min_rows
    min_rows = schema.get("min_rows")
    if isinstance(min_rows, int) and len(df) < min_rows:
        errors.append(f"row count {len(df)} < min_rows {min_rows}")

    # required columns present
    for col, col_spec in spec_cols.items():
        if col_spec.get("required") and col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # extras policy
    if strict and not allow_extra:
        extras = [c for c in df.columns if c not in spec_cols]
        if extras:
            errors.append(f"extra columns present: {extras}")

    # primary key
    pk = schema.get("primary_key")
    if pk:
        pk_cols = [pk] if isinstance(pk, str) else list(pk)
        for k in pk_cols:
            if k not in df.columns:
                errors.append(f"primary_key column missing: {k}")
        if not any(e.startswith("primary_key column missing") for e in errors):
            subset = df[pk_cols]
            if subset.isna().any().any():
                errors.append("primary_key contains nulls")
            if subset.duplicated().any():
                errors.append("primary_key contains duplicates")

    # top-level unique constraints
    for u in schema.get("unique", []) or []:
        if u not in df.columns:
            errors.append(f"unique column missing: {u}")
        else:
            col = df[u]
            if col.isna().any():
                errors.append(f"{u}: has nulls but must be unique")
            if col.duplicated().any():
                errors.append(f"{u}: contains duplicates")

    # per-column checks
    for name, spec in spec_cols.items():
        if name not in df.columns:
            # missing already handled for required; skip here
            continue
        s = df[name]
        for issue in _validate_column(name, s, spec or {}):
            errors.append(str(issue))

    return (len(errors) == 0, errors)


# ----------------------------- CLI -----------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate CSV against schema/default YAMLs")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--schema", type=Path, default=Path("configs/schema.yaml"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    sch = load_yaml(args.schema)
    in_path = choose_input_path(cfg, args.input)

    df = pd.read_csv(in_path)

    ok, errs = validate_df(df, sch)
    if ok:
        if not args.quiet:
            print("✅ Validation passed")
        return 0

    print("❌ Validation failed:")
    for e in errs:
        print(" -", e)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

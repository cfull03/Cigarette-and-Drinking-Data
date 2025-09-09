"""Lightweight CSV validator driven by YAML schema.

Changes:
- Input discovery now supports directories (`paths.raw_dir`, `paths.sample_dir`, `paths.processed_dir`) and falls back to latest CSV.
- Respects `nullable:` in schema (treats `nullable: false` as required non-null).
- Reads CSV with `io.read_kwargs` from `configs/default.yaml` when present.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import re
from typing import Any, Dict, Iterable, Tuple, List

import pandas as pd
import pandas.api.types as pat
import yaml


# -------------------------- config helpers --------------------------

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _is_csv(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".csv"


def _latest_csv_in_dir(d: Path) -> Path | None:
    if not d or not d.exists() or not d.is_dir():
        return None
    files = sorted(d.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _as_path(x: Any) -> Path | None:
    if not x:
        return None
    return Path(str(x))


def _choose_input(cfg: dict, cli_input: Path | None) -> Path:
    """Pick an input CSV.

    Strategy:
    1) If CLI path is provided: use file directly; if it's a dir, pick newest CSV inside.
    2) Try explicit file paths in config: `paths.sample_input`, `paths.raw`.
    3) Try directories in config: `paths.processed_dir`, `paths.raw_dir`, `paths.sample_dir` (pick newest CSV).
    4) Fallback to repo defaults: `data/processed`, `data/raw`, `data/sample` (newest CSV).
    """
    # 1) CLI override
    if cli_input is not None:
        p = Path(cli_input)
        if p.is_dir():
            cand = _latest_csv_in_dir(p)
            if cand:
                return cand
            raise FileNotFoundError(f"No CSV files found in directory: {p}")
        if _is_csv(p):
            return p
        raise FileNotFoundError(f"Input path is not a CSV file: {p}")

    paths = (cfg or {}).get("paths", {})

    # 2) explicit files
    for key in ("sample_input", "raw"):
        p = _as_path(paths.get(key))
        if p and _is_csv(p):
            return p

    # 3) configured directories (preference: processed -> raw -> sample)
    for key in ("processed_dir", "raw_dir", "sample_dir"):
        d = _as_path(paths.get(key))
        cand = _latest_csv_in_dir(d) if d else None
        if cand:
            return cand

    # 4) repo default folders
    for d in map(Path, ["data/processed", "data/raw", "data/sample"]):
        cand = _latest_csv_in_dir(d)
        if cand:
            return cand

    raise FileNotFoundError(
        "No usable input CSV. Provide --input or ensure CSV exists under data/processed, data/raw, or data/sample.\n"
        "Alternatively set one of: paths.sample_input (file), paths.raw (file), paths.processed_dir/raw_dir/sample_dir (dirs) in configs/default.yaml."
    )


# -------------------------- dtype & column checks --------------------------

def _is_categorical_dtype(series: pd.Series) -> bool:
    # Avoid deprecation warnings by checking dtype object
    try:
        return pat.is_categorical_dtype(series.dtype)  # type: ignore[arg-type]
    except Exception:
        return pat.is_categorical_dtype(series)


def _dtype_ok(s: pd.Series, expected: str) -> bool:
    e = expected.lower()
    if e in {"int", "integer"}:  # pandas treats bool as int-like
        return pat.is_integer_dtype(s) or pat.is_bool_dtype(s)
    if e in {"float", "double", "number", "numeric"}:
        return pat.is_numeric_dtype(s)
    if e in {"str", "string"}:
        return pat.is_string_dtype(s) or s.dtype == "object"
    if e in {"bool", "boolean"}:
        return pat.is_bool_dtype(s)
    if e in {"date", "datetime", "datetime64"}:
        return pat.is_datetime64_any_dtype(s)
    if e in {"category", "categorical"}:
        return _is_categorical_dtype(s) or pat.is_object_dtype(s)
    return True


def _required_non_null(spec: dict) -> bool:
    # Prefer explicit `nullable`; fall back to `required` for backward compat
    if "nullable" in spec:
        return not bool(spec.get("nullable"))
    return bool(spec.get("required"))


def _validate_col(name: str, s: pd.Series, spec: dict) -> list[str]:
    errs: list[str] = []
    t = str(spec.get("type", "")).strip().lower()

    # required / unique
    if _required_non_null(spec) and s.isna().any():
        errs.append(f"{name}: {int(s.isna().sum())} missing values where required")
    if spec.get("unique") and s.duplicated().any():
        errs.append(f"{name}: {int(s.duplicated().sum())} duplicate values present")

    # type enforcement with safe coercion checks
    numeric_expected = t in {"integer", "int", "number", "numeric", "float", "double"}
    if t and not _dtype_ok(s, t):
        if numeric_expected:
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.isna().all() and s.notna().any():
                errs.append(f"{name}: cannot coerce to numeric")
            if t in {"integer", "int"}:
                non_int = (~coerced.isna()) & (coerced % 1 != 0)
                if non_int.any():
                    errs.append(f"{name}: {int(non_int.sum())} values are non-integers")
        elif t in {"datetime", "date", "datetime64"}:
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.isna().all() and s.notna().any():
                errs.append(f"{name}: cannot parse as datetime")
        # strings/categories validated below via regex/allowed/max_len

    # numeric ranges (use coerced values when expected numeric)
    if numeric_expected:
        num = s if pat.is_numeric_dtype(s) else pd.to_numeric(s, errors="coerce")
        if "min" in spec:
            below = num.dropna() < spec["min"]
            if below.any():
                errs.append(f"{name}: {int(below.sum())} values below min {spec['min']}")
        if "max" in spec:
            above = num.dropna() > spec["max"]
            if above.any():
                errs.append(f"{name}: {int(above.sum())} values above max {spec['max']}")

    # string constraints
    if t in {"str", "string"}:
        max_len = spec.get("max_len")
        if isinstance(max_len, int):
            too_long = s.dropna().astype(str).map(len) > max_len
            if too_long.any():
                errs.append(f"{name}: {int(too_long.sum())} strings exceed max_len {max_len}")

    # regex
    regex = spec.get("regex")
    if regex:
        patt = re.compile(str(regex))
        bad = s.dropna().astype(str).map(lambda v: not bool(patt.fullmatch(v)))
        if bad.any():
            errs.append(f"{name}: {int(bad.sum())} values fail regex {regex!r}")

    # allowed categories
    allowed = spec.get("allowed")
    if allowed:
        allowed_str = {str(x) for x in allowed}
        invalid = set(pd.Series(s.dropna().unique()).astype(str)) - allowed_str
        if invalid:
            errs.append(f"{name}: unexpected categories {sorted(invalid)}")

    return errs


def _parse_pk(pk_spec: Any) -> Tuple[List[str], bool]:
    """Return (columns, optional) from schema's primary_key.

    Supports:
      - "id"
      - ["id", "source"]
      - {columns: ["id"], optional: true}
    """
    if not pk_spec:
        return ([], False)
    if isinstance(pk_spec, dict):
        cols = pk_spec.get("columns", [])
        if isinstance(cols, (str, Path)):
            cols = [str(cols)]
        elif isinstance(cols, Iterable):
            cols = [str(c) for c in cols]
        else:
            cols = []
        optional = bool(pk_spec.get("optional", False))
        return (cols, optional)
    if isinstance(pk_spec, (list, tuple, set)):
        return ([str(c) for c in pk_spec], False)
    if isinstance(pk_spec, (str, Path)):
        return ([str(pk_spec)], False)
    return ([], False)


# -------------------------- main validation --------------------------

def validate_df(df: pd.DataFrame, schema: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    spec_cols: dict[str, dict] = schema.get("columns", {}) or {}

    # global checks
    min_rows = schema.get("min_rows")
    if isinstance(min_rows, int) and len(df) < min_rows:
        errors.append(f"row count {len(df)} < min_rows {min_rows}")

    # required columns present (derive from `nullable` when provided)
    for col, col_spec in spec_cols.items():
        required_present = _required_non_null(col_spec)
        if required_present and col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # primary key (supports optional)
    pk_cols, pk_optional = _parse_pk(schema.get("primary_key"))
    if pk_cols:
        missing_cols = [c for c in pk_cols if c not in df.columns]
        if missing_cols:
            if not pk_optional:
                for c in missing_cols:
                    errors.append(f"primary_key column missing: {c}")
        else:
            subset = df[pk_cols]
            if pk_optional:
                mask = subset.notna().all(axis=1)
                if mask.any() and subset[mask].duplicated().any():
                    errors.append("primary_key duplicates among non-null rows")
            else:
                if subset.isna().any().any():
                    errors.append("primary_key contains nulls")
                if subset.duplicated().any():
                    errors.append("primary_key contains duplicates")

    # per-column validations
    for name, spec in spec_cols.items():
        if name not in df.columns:
            continue
        errors.extend(_validate_col(name, df[name], spec or {}))

    return (len(errors) == 0, errors)


# -------------------------- CLI --------------------------

def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate CSV against schema/default YAMLs")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--schema", type=Path, default=Path("configs/schema.yaml"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.config)
    sch = _load_yaml(args.schema)
    csv_path = _choose_input(cfg, args.input)

    # Respect io.read_kwargs when present
    read_kwargs = ((cfg.get("io") or {}).get("read_kwargs") or {}) if isinstance(cfg, dict) else {}
    df = pd.read_csv(csv_path, **read_kwargs)

    ok, errs = validate_df(df, sch)

    if ok:
        if not args.quiet:
            print(f"✅ Validation passed | file={csv_path}")
        return 0
    print(f"❌ Validation failed | file={csv_path}")
    for e in errs:
        print(" -", e)
    return 2


if __name__ == "__main__":
    raise SystemExit(cli())

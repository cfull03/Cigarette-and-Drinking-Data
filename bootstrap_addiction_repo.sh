#!/usr/bin/env bash
# File: bootstrap_addiction_repo.sh
# Purpose: Create a professional, lean SOLO DS repository wired to your CSV.
# Usage:
#   bash bootstrap_addiction_repo.sh [RAW_CSV_PATH]
# Example:
#   bash bootstrap_addiction_repo.sh data/raw/addiction_population_data.csv
set -euo pipefail

RAW_CSV_PATH=${1:-data/raw/addiction_population_data.csv}
PKG_NAME=addiction_ds
APP_NAME=addiction-ds
KERNEL=${PKG_NAME}

# --- Directories ---
mkdir -p .vscode configs data/{raw,interim,processed,external,sample} notebooks reports/figures src/${PKG_NAME}

# --- .gitignore ---
cat > .gitignore << 'GIT'
# OS/Editors
.DS_Store
.vscode/
.idea/

# Python
__pycache__/
*.py[cod]
.venv/

# Build
build/
dist/
*.egg-info/
.eggs/

# Tool caches
.pytest_cache/
.ruff_cache/

# Data & outputs
/data/*
!/data/sample/
!/data/sample/**
/models/*
/reports/figures/*
GIT

# --- pyproject.toml ---
cat > pyproject.toml << 'TOML'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "addiction-ds"
version = "0.1.0"
description = "Professional solo data-science project for addiction dataset"
readme = "README.md"
requires-python = ">=3.10"
authors = [{ name = "Christian Fullerton" }]
license = { text = "MIT" }
keywords = ["data-science", "reproducibility", "solo"]
classifiers = [
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
]
dependencies = [
  "pandas>=2.1",
  "pyyaml>=6.0",
  "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
  "ruff>=0.6",
  "black>=24.4",
  "ipykernel>=6.29",
]

[project.scripts]
ad-validate = "addiction_ds.validate:cli"

[tool.setuptools]
package-dir = { "" = "src" }

[tool.setuptools.packages.find]
where = ["src"]
include = ["addiction_ds*"]

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]
TOML

# --- Makefile ---
cat > Makefile << 'MAKE'
# File: Makefile
SHELL := /bin/bash
PYTHON ?= python
PIP ?= pip
CONFIG ?= configs/default.yaml
SCHEMA ?= configs/schema.yaml
INPUT ?=
KERNEL ?= addiction_ds

.PHONY: help install env format lint validate clean

help:
	@echo "Targets: install env format lint validate clean"

install:
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PIP) install -e '.[dev]'

env:
	$(PYTHON) -m ipykernel install --user --name=$(KERNEL) || true

format:
	black .

lint:
	ruff check .

validate:
	@if [ -n "$(INPUT)" ]; then \
		$(PYTHON) -m addiction_ds.validate --config $(CONFIG) --schema $(SCHEMA) --input "$(INPUT)"; \
	else \
		$(PYTHON) -m addiction_ds.validate --config $(CONFIG) --schema $(SCHEMA); \
	fi

clean:
	rm -rf .ruff_cache .pytest_cache build dist **/__pycache__ *.egg-info
MAKE

# --- configs/default.yaml ---
cat > configs/default.yaml << YAML
project_name: addiction_ds
random_state: 42

paths:
  data_dir: data
  raw: ${RAW_CSV_PATH}
  sample_input: data/sample/sample.csv
  reports_dir: reports
  schema: configs/schema.yaml
YAML

# --- placeholder schema (will be overwritten if RAW exists) ---
cat > configs/schema.yaml << 'YAML'
columns:
  id:    {type: integer, required: true, min: 1}
  name:  {type: string,  required: true,  max_len: 100}
  age:   {type: integer, min: 0, max: 120}
primary_key: [id]
min_rows: 1
YAML

# --- package init ---
cat > src/${PKG_NAME}/__init__.py << 'PY'
__all__ = ["__version__"]
__version__ = "0.1.0"
PY

# --- validator ---
cat > src/${PKG_NAME}/validate.py << 'PY'
from __future__ import annotations
from pathlib import Path
import argparse
import re
from typing import Any

import pandas as pd
import pandas.api.types as pat
import yaml


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _choose_input(cfg: dict, cli_input: Path | None) -> Path:
    if cli_input is not None:
        return cli_input
    paths = (cfg or {}).get("paths", {})
    for k in ("sample_input", "raw"):
        val = paths.get(k)
        if val and Path(val).exists():
            return Path(val)
    raise FileNotFoundError("No usable input CSV. Provide --input or set paths.sample_input/raw in configs/default.yaml.")


def _dtype_ok(s: pd.Series, expected: str) -> bool:
    e = expected.lower()
    if e in {"int", "integer"}: return pat.is_integer_dtype(s) or pat.is_bool_dtype(s)
    if e in {"float", "double", "number", "numeric"}: return pat.is_numeric_dtype(s)
    if e in {"str", "string"}: return pat.is_string_dtype(s) or s.dtype == "object"
    if e in {"bool", "boolean"}: return pat.is_bool_dtype(s)
    if e in {"date", "datetime", "datetime64"}: return pat.is_datetime64_any_dtype(s)
    if e in {"category", "categorical"}: return pat.is_categorical_dtype(s) or pat.is_object_dtype(s)
    return True


def _validate_col(name: str, s: pd.Series, spec: dict) -> list[str]:
    errs: list[str] = []
    t = str(spec.get("type", "")).strip().lower()

    if spec.get("required") and s.isna().any():
        errs.append(f"{name}: {int(s.isna().sum())} missing values where required")

    if spec.get("unique") and s.duplicated().any():
        errs.append(f"{name}: {int(s.duplicated().sum())} duplicate values present")

    if t and not _dtype_ok(s, t):
        if t in {"integer", "int", "number", "numeric", "float", "double"}:
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

    if t in {"integer", "int", "float", "double", "number", "numeric"} and pat.is_numeric_dtype(s):
        if "min" in spec:
            below = s.dropna() < spec["min"]
            if below.any():
                errs.append(f"{name}: {int(below.sum())} values below min {spec['min']}")
        if "max" in spec:
            above = s.dropna() > spec["max"]
            if above.any():
                errs.append(f"{name}: {int(above.sum())} values above max {spec['max']}")

    if t in {"str", "string"}:
        max_len = spec.get("max_len")
        if isinstance(max_len, int):
            too_long = s.dropna().astype(str).map(len) > max_len
            if too_long.any():
                errs.append(f"{name}: {int(too_long.sum())} strings exceed max_len {max_len}")

    regex = spec.get("regex")
    if regex:
        patt = re.compile(str(regex))
        bad = s.dropna().astype(str).map(lambda v: not bool(patt.fullmatch(v)))
        if bad.any():
            errs.append(f"{name}: {int(bad.sum())} values fail regex {regex!r}")

    allowed = spec.get("allowed")
    if allowed:
        invalid = set(pd.Series(s.dropna().unique()).astype(str)) - {str(x) for x in allowed}
        if invalid:
            errs.append(f"{name}: unexpected categories {sorted(invalid)}")

    return errs


def validate_df(df: pd.DataFrame, schema: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []

    spec_cols: dict[str, dict] = schema.get("columns", {}) or {}

    min_rows = schema.get("min_rows")
    if isinstance(min_rows, int) and len(df) < min_rows:
        errors.append(f"row count {len(df)} < min_rows {min_rows}")

    for col, col_spec in spec_cols.items():
        if col_spec.get("required") and col not in df.columns:
            errors.append(f"Missing required column: {col}")

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

    for name, spec in spec_cols.items():
        if name not in df.columns:
            continue
        errors.extend(_validate_col(name, df[name], spec or {}))

    return (len(errors) == 0, errors)


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

    df = pd.read_csv(csv_path)
    ok, errs = validate_df(df, sch)

    if ok:
        if not args.quiet:
            print("✅ Validation passed")
        return 0
    print("❌ Validation failed:")
    for e in errs:
        print(" -", e)
    return 2


if __name__ == "__main__":
    raise SystemExit(cli())
PY

# --- README ---
cat > README.md << 'MD'
# addiction-ds (professional solo data-science repo)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e '.[dev]'

# Put your CSV at: data/raw/addiction_population_data.csv
# Or pass a custom path when generating the repo: bash bootstrap_addiction_repo.sh path/to.csv

# Validate using configs/default.yaml and configs/schema.yaml
make validate
# or
ad-validate --input data/raw/addiction_population_data.csv
```

## Layout
- `configs/` — `default.yaml` (paths), `schema.yaml` (data contract)
- `data/` — raw/interim/processed/external/sample (large files ignored by git)
- `src/addiction_ds/` — reusable code; `validate.py` CLI (`ad-validate`)
- `notebooks/` — EDA/modeling; keep heavy logic in `src/`
- `Makefile` — install, lint, format, validate
- `pyproject.toml` — packaging & deps
MD

# --- Optional: infer schema if CSV exists ---
if [[ -f "${RAW_CSV_PATH}" ]]; then
python - << 'PY'
from pathlib import Path
import pandas as pd, yaml, numpy as np

raw = Path("${RAW_CSV_PATH}")
df = pd.read_csv(raw)

schema = {"columns": {}, "min_rows": 1}
# primary key candidate
for col in df.columns:
    if df[col].isna().sum() == 0 and df[col].nunique(dropna=True) == len(df):
        schema["primary_key"] = [col]
        break

for col in df.columns:
    s = df[col]
    spec = {}
    if str(s.dtype).startswith("int"):
        spec["type"] = "integer"
    elif str(s.dtype).startswith("float"):
        spec["type"] = "number"
    elif str(s.dtype) in ("bool", "boolean"):
        spec["type"] = "boolean"
    else:
        spec["type"] = "string"

    if s.isna().sum() == 0:
        spec["required"] = True

    if spec["type"] in {"integer", "number"}:
        spec["min"] = float(np.nanmin(s.values))
        spec["max"] = float(np.nanmax(s.values))

    # Keep allowed set only for low-cardinality strings/bools
    if spec["type"] in {"string", "boolean"}:
        uniq = s.dropna().unique()
        if len(uniq) <= 20:
            spec["allowed"] = [str(x) for x in sorted(uniq.tolist())]
        if spec["type"] == "string" and not s.dropna().empty:
            spec["max_len"] = int(s.dropna().astype(str).map(len).max())

    schema["columns"][col] = spec

out = Path("configs/schema.yaml")
out.write_text(yaml.dump(schema, sort_keys=False), encoding="utf-8")
print(f"Wrote inferred schema to {out}")
PY
fi

# End of bootstrap
printf "\n✔ Repo created. Next steps:\n"
printf "  1) Ensure your raw CSV exists at: %s\n" "$RAW_CSV_PATH"
printf "  2) Create venv → install deps → validate:\n"
printf "     python -m venv .venv && source .venv/bin/activate\n"
printf "     python -m pip install -U pip setuptools wheel\n"
printf "     pip install -e '.[dev]'\n"
printf "     make validate\n"

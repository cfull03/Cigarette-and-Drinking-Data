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

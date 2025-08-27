#!/usr/bin/env bash
set -euo pipefail

# Ensure repo root
if [[ ! -d "." ]]; then
  echo "Run from the repository root" >&2
  exit 1
fi

# Directories
mkdir -p configs \
         data/{raw,external,interim,processed,sample} \
         docs \
         models \
         notebooks \
         reports/figures \
         scripts \
         src/cig_drink/models \
         tests

# Move & rename existing notebooks (if present)
if [[ -f "Data Cleaning/Cigarettes & Drinking Data Cleaning.ipynb" ]]; then
  mv -n "Data Cleaning/Cigarettes & Drinking Data Cleaning.ipynb" notebooks/01_data_cleaning.ipynb
fi
if [[ -f "Machine Learning/Machine Learning Model.ipynb" ]]; then
  mv -n "Machine Learning/Machine Learning Model.ipynb" notebooks/02_ml_baseline.ipynb
fi
if [[ -f "Deep Learning/Neural Model.ipynb" ]]; then
  mv -n "Deep Learning/Neural Model.ipynb" notebooks/03_dl_model.ipynb
fi
# Remove now-empty dirs
for d in "Data Cleaning" "Machine Learning" "Deep Learning"; do
  [[ -d "$d" ]] && rmdir "$d" 2>/dev/null || true
done

# Move & normalize CSVs (if present)
if [[ -f "addiction_population_data.csv" ]]; then
  mv -n "addiction_population_data.csv" data/raw/
fi
if [[ -f "Updated Cigarette & Drinking Data.csv" ]]; then
  mv -n "Updated Cigarette & Drinking Data.csv" "data/raw/updated_cigarette_drinking_data.csv"
fi

# Small sample for quick peeks
RAW_ANY=$(ls data/raw/*.csv 2>/dev/null | head -n1 || true)
if [[ -n "${RAW_ANY}" ]]; then
  head -n 101 "${RAW_ANY}" > data/sample/sample_100_rows.csv || true
fi

# README
cat > README.md << 'EOR'
# Cigarette & Drinking Data — Reproducible ML repo

A clean, reproducible structure for data cleaning, classical ML, and deep learning on cigarette/alcohol-related datasets.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # or: conda create -n cgd python=3.11
pip install -r requirements.txt
pre-commit install  # optional but recommended
make data           # run toy data pipeline (edit configs/paths.yaml)
make train-ml       # run baseline ML training script


# Cigarette & Drinking Data — Reproducible ML repo

A clean, reproducible structure for data cleaning, classical ML, and deep learning on cigarette/alcohol-related datasets.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # or: conda create -n cgd python=3.11
pip install -r requirements.txt
pre-commit install  # optional but recommended
make data           # run toy data pipeline (edit configs/paths.yaml)
make train-ml       # run baseline ML training script


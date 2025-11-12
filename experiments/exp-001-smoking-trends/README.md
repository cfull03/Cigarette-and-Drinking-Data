# exp-001 — Smoking trends (RF baseline)

**Goal.** RandomForest baseline on engineered features with robust preprocessing (safe all-NaN handling).

- **Manifest:** `experiment.yml`
- **Branch:** `exp/001-smoking-trends-cf` @ `<commit-sha>`
- **Target column:** `has_health_issues`
- **Artifacts:** `artifacts/rf/model.joblib` (LFS or external URI), `artifacts/rf/metrics.json`, `artifacts/rf/feature_importances.csv`, `artifacts/rf/predictions.csv` (if produced)
- **Figures:** `experiments/exp-001-smoking-trends/results/figures/`

## Reproduce
```bash
conda env update -n cigarette-and-drinking-data -f environment.yml --prune
conda activate cigarette-and-drinking-data
pip install -e .

make data
make features
make preprocess
make build
make eval
# optional
make predict

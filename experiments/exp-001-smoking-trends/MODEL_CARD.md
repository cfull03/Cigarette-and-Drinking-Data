# Model Card — exp-001 Smoking Trends (RandomForest Baseline)

**Model ID**: `exp-001-smoking-trends`  
**Owner**: Christian Fullerton  
**Created**: 2025-11-11  
**Repository**: https://github.com/cfull03/Cigarette-and-Drinking-Data  
**Branch/Commit**: `exp/001-smoking-trends-cf` @ `b2b7a589fd233f50385312a14f2694fae478b3ff`  
**Artifact**: `artifacts/rf/model.joblib` (Git LFS or external URI)

---

## 1) Overview

This model is a **RandomForestClassifier** baseline that predicts the binary target **`has_health_issues`** from engineered features derived from an addiction/health dataset. It uses a robust `ColumnTransformer` that safely handles all-NaN columns and unseen categories.

- **Goal**: Establish a clean, reproducible baseline and a reference preprocessor.
- **Scope**: Single dataset, single split (train/test via `train_test_split_safe`), single algorithm with transparent hyperparameters.

---

## 2) Intended Use & Users

- **Intended use**: Exploratory analysis, baseline benchmarking, and feature/preprocessing validation.
- **Not for**: Clinical or policy decision-making. Outputs are *not* medical advice.
- **Primary users**: Student/research contributors to this repository; reviewers evaluating experiment PRs.

---

## 3) Data

- **Raw data**: `data/raw/addiction_population_data.csv`
- **Processed features**: `data/processed/features.csv` (includes target `has_health_issues`)
- **Splits**: `train_test_split_safe(test_size=0.2, stratify=True, random_state=42)`
- **Data notes**:
  - Potential for demographic proxies of sensitive attributes.
  - Ensure license and PII handling comply with project standards (see `DATA_CARD.md`).

---

## 4) Preprocessing

- **Module**: `addiction.preprocessor`
- **Saved preprocessor**: `models/preprocessor.joblib`
- **Strategy**:
  - `numeric_some`: median impute → `StandardScaler`
  - `numeric_all_nan`: constant(0.0) → `StandardScaler`
  - `categorical_some`: most_frequent → `OneHotEncoder(handle_unknown="ignore")`
  - `categorical_all_nan`: constant("missing") → `OneHotEncoder(handle_unknown="ignore")`
- **Safeguards**: Drops target and any `<target>_*` derived columns before fit/transform.

---

## 5) Model Details

- **Algorithm**: `RandomForestClassifier`
- **Hyperparameters** (from `experiment.yml`):
  - `n_estimators=500`, `max_depth=5`, `max_features="sqrt"`, `min_samples_leaf=1`
  - `class_weight="balanced"`, `n_jobs=-1`, `random_state=42`
- **Training module**: `addiction.modeling.train`
- **Output**: `artifacts/rf/model.joblib`

---

## 6) Evaluation

- **Evaluation module**: `addiction.eval`
- **Metrics file**: `artifacts/rf/metrics.json`
- **Expected keys**: `accuracy, f1, precision, recall, roc_auc, confusion_matrix, n_samples`
- **How to reproduce**:
  ```bash
  conda env update -n cigarette-and-drinking-data -f environment.yml --prune
  conda activate cigarette-and-drinking-data
  pip install -e .
  make data && make features && make preprocess && make build && make eval

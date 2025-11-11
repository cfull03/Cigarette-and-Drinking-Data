# Data Card — exp-001 Smoking Trends

**Experiment ID:** `exp-001-smoking-trends`  
**Owner:** Christian Fullerton  
**Created:** 2025-11-11  
**Repo:** https://github.com/cfull03/Cigarette-and-Drinking-Data  
**Code Commit:** b2b7a589fd233f50385312a14f2694fae478b3ff

---

## 1) Dataset Summary
This experiment predicts the binary target **`has_health_issues`** from demographic and lifestyle features related to smoking/alcohol behavior. The pipeline reads a single tabular CSV, engineers features, applies a robust `ColumnTransformer`, and trains a RandomForest baseline.

- **Raw data:** `data/raw/addiction_population_data.csv`
- **Interim data:** `data/interim/dataset.csv`
- **Features (includes target):** `data/processed/features.csv`
- **Preprocessed design matrix:** `data/processed/dataset.preprocessed.csv`  
- **Target column:** `has_health_issues` (0/1)

> Note: This card documents data as used in **exp-001** at the code state above. If you regenerate features on a different commit or with different settings, update this card accordingly.

---

## 2) Source, License, and Consent
- **Source:** Local project CSV (`data/raw/addiction_population_data.csv`).  
- **License/Usage:** _(fill in)_ e.g., “Internal academic use only. Redistribution prohibited.”  
- **Consent/Privacy:** Confirm no personally identifiable information (PII) is present. If PII exists, ensure de-identification before processing or sharing.

---

## 3) Schema (practical view)
A formal schema file is **not** required for this experiment. At minimum:

- **Target:** `has_health_issues` (integer/binary)  
- **Features:** mix of numeric and categorical columns 


---

## 4) Preprocessing & Feature Engineering
- **Module:** `addiction.preprocessor`
- **Saved preprocessor:** `models/preprocessor.joblib`
- **Strategies:**
  - **Numeric (some missing):** median imputation → `StandardScaler`
  - **Numeric (all NaN):** constant(0.0) → `StandardScaler`
  - **Categorical (some missing):** most_frequent → `OneHotEncoder(handle_unknown="ignore")`
  - **Categorical (all NaN):** constant("missing") → `OneHotEncoder(handle_unknown="ignore")`
- **Safeguards:** The preprocessor drops the target and any derived `<target>_*` columns prior to fit/transform.

---

## 5) Splits & Sampling
- **Split method:** stratified 80/20 (hold-out) via `train_test_split_safe`
- **Stratify:** `True` on `has_health_issues`
- **Random state:** `42`
- **Notes:** No resampling/oversampling performed in exp-001; class imbalance is handled via model `class_weight="balanced"`.

---

## 6) Quality Checks
- **File integrity:** ensure CSVs load without parse errors (run `tests/test_data.py`).
- **Column presence:** target exists and is binary (0/1).
- **Missingness:** imputation strategies above; verify no all-NaN columns remain post-transform.
- **Leakage guard:** any columns derived from target are removed before fit/transform.

---

## 8) Reproducibility
**Exact commands:**
```bash
# environment
conda env update -n cigarette-and-drinking-data -f environment.yml --prune
conda activate cigarette-and-drinking-data
pip install -e .

# pipeline
make data
make features
make preprocess

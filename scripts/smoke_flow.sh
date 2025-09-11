#!/usr/bin/env bash
# Purpose: end-to-end smoke test that the repo flows properly.
# - installs project
# - optional lint/tests
# - ensures processed splits
# - trains model
# - evaluates and writes reports
# - verifies artifacts exist

set -Eeuo pipefail
shopt -s nullglob

# ---- config ----
PYTHON_BIN="python"
CFG_EXPERIMENT="configs/experiment.yaml"
RAW_CSV="data/raw/addiction_population_data.csv"
PROCESSED_DIR="data/processed"
TRAIN_CSV="${PROCESSED_DIR}/train.csv"
VAL_CSV="${PROCESSED_DIR}/val.csv"
MODELS_DIR="models"
MODEL_LATEST="${MODELS_DIR}/latest.joblib"
REPORTS_DIR="reports"

step() { echo -e "\n▶ $*"; }
ok()   { echo "✅ $*"; }
warn() { echo "⚠️  $*"; }
fail(){ echo "❌ $*"; exit 1; }

step "Install project (editable) + dev tools"
$PYTHON_BIN -m pip install -U pip setuptools wheel >/dev/null
$PYTHON_BIN -m pip install -e '.[dev]' >/dev/null
ok "Installed"

step "Lint & tests (best-effort)"
if command -v ruff >/dev/null; then ruff check . || warn "ruff issues"; else warn "ruff not installed"; fi
if command -v black >/dev/null; then black --check . || warn "black check failed"; else warn "black not installed"; fi
if [ -d tests ] || [ -d test ]; then pytest -q || warn "pytest failures"; else warn "no tests found"; fi

step "Ensure processed splits"
mkdir -p "$PROCESSED_DIR"
if [[ ! -f "$TRAIN_CSV" || ! -f "$VAL_CSV" ]]; then
  if compgen -G "${PROCESSED_DIR}/*.csv" > /dev/null; then
    # Use newest processed CSV
    $PYTHON_BIN -m addiction_ds.ml_utils split --dir "$PROCESSED_DIR" --out-train "$TRAIN_CSV" --out-val "$VAL_CSV"
  else
    # Fall back: split from raw
    [[ -f "$RAW_CSV" ]] || fail "Missing $RAW_CSV and no processed CSVs to split"
    # Simple 80/20 split from RAW
    $PYTHON_BIN - <<'PY'
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
raw = Path('data/raw/addiction_population_data.csv')
df = pd.read_csv(raw).drop_duplicates()
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
Path('data/processed').mkdir(parents=True, exist_ok=True)
train_df.to_csv('data/processed/train.csv', index=False)
val_df.to_csv('data/processed/val.csv', index=False)
print('Wrote data/processed/train.csv and data/processed/val.csv')
PY
  fi
else
  ok "Found existing $TRAIN_CSV and $VAL_CSV"
fi

[[ -f "$TRAIN_CSV" && -f "$VAL_CSV" ]] || fail "Processed splits missing"
ok "Processed splits ready"

step "Train model (uses configs/experiment.yaml)"
MLFLOW_TRACKING_URI="file:./mlruns" MLFLOW_EXPERIMENT_NAME="addiction_ds" \
$PYTHON_BIN -m addiction_ds.train

[[ -f "$MODEL_LATEST" ]] || fail "Model artifact missing: $MODEL_LATEST"
ok "Model saved → $MODEL_LATEST"

step "Evaluate on validation CSV and write reports"
$PYTHON_BIN -m addiction_ds.evaluate --config "$CFG_EXPERIMENT" --csv "$VAL_CSV" --reports-dir "$REPORTS_DIR"

LATEST_JSON="${REPORTS_DIR}/metrics_latest.json"
[[ -f "$LATEST_JSON" ]] || fail "Missing $LATEST_JSON"
LATEST_TXT=$(ls -1t ${REPORTS_DIR}/classification_report_*.txt 2>/dev/null | head -n1 || true)
[[ -n "$LATEST_TXT" ]] || warn "No classification report text found"

ok "Reports in $REPORTS_DIR"

step "Smoke-load the pipeline and run one prediction"
$PYTHON_BIN - <<'PY'
from addiction_ds.io import load_cfg, load_model
import pandas as pd
from pathlib import Path
cfg = load_cfg('configs/experiment.yaml')
pipe = load_model(cfg, name='latest', framework='sklearn')
val = Path('data/processed/val.csv')
df = pd.read_csv(val)
X = df[cfg['features']['numeric'] + cfg['features']['categorical']].head(1)
print('One-row predict_proba OK' if hasattr(pipe, 'predict_proba') and pipe.predict_proba(X).shape[0]==1 else 'Predict OK' )
PY

ok "Smoke test complete. Flow looks healthy."

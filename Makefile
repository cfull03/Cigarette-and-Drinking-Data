# File: Makefile
# Purpose: Clean, working MLOps targets using your package + ml_utils splitter

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
CONFIG ?= configs/default.yaml
EXPERIMENT_CFG ?= configs/experiment.yaml
SCHEMA ?= configs/schema.yaml
INPUT ?=
# Prefer addiction_ds; tolerate addition_ds; fall back to cig_drink
MODULE ?= $(shell if [ -d src/addiction_ds ]; then echo addiction_ds; \
                 elif [ -d src/addition_ds ]; then echo addition_ds; \
                 elif [ -d src/cig_drink ]; then echo cig_drink; \
                 else echo addiction_ds; fi)
KERNEL ?= $(MODULE)

# --- Data & artifacts ---
RAW ?= data/raw/addiction_population_data.csv
PROCESSED_DIR ?= data/processed
TRAIN ?= $(PROCESSED_DIR)/train.csv
VAL ?= $(PROCESSED_DIR)/val.csv
MODELS_DIR ?= models
MODEL ?= $(MODELS_DIR)/latest.joblib
REPORTS_DIR ?= reports

# --- MLflow (optional local store) ---
MLFLOW_TRACKING_URI ?= file:./mlruns
EXPERIMENT ?= addiction_ds
API_HOST ?= 0.0.0.0
API_PORT ?= 8000

EVAL_MODEL ?= latest
THRESHOLD ?= 0.5


# Constrain BLAS threads for reproducibility, especially in CI
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1

.PHONY: default help venv install env format format-check lint test validate validate_latest clean sample clean_data \
	split train eval model serve mlflow-ui ci smoke smoke-scripts smoke-pipeline smoke-full

default: help

help:
	@echo "Targets: venv install env format format-check lint test validate validate_latest clean sample clean_data"
	@echo "MLOps: split train eval model serve mlflow-ui ci"
	@echo "Smokes: smoke smoke-scripts smoke-pipeline smoke-full"
	@echo "Vars: MODULE=$(MODULE) CONFIG=$(CONFIG) SCHEMA=$(SCHEMA) INPUT=$(INPUT)"
	@echo "Data: PROCESSED_DIR=$(PROCESSED_DIR) TRAIN=$(TRAIN) VAL=$(VAL)"
	@echo "Artifacts: MODEL=$(MODEL) REPORTS_DIR=$(REPORTS_DIR)"
	@echo "Split args passthrough: make split ARGS='--csv path/to.csv --stratify-col is_smoker'"

venv:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && python -m pip install -U pip setuptools wheel || true

install:
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -e '.[dev]'

env:
	$(PYTHON) -m ipykernel install --user --name=$(KERNEL) || true

format:
	ruff check . --fix
	black .
	nbqa ruff notebooks --fix || true
	nbqa black notebooks || true
	@echo "format done"

format-check:
	ruff check .
	nbqa ruff notebooks || true
	black --check .

lint:
	ruff check .
	nbqa ruff notebooks || true

# Run tests if any test files exist
TEST_GLOB := \( -name "test_*.py" -o -name "*_test.py" \)

test:
	@if ( [ -d tests ] || [ -d test ] ) && (find tests test -type f $(TEST_GLOB) 2>/dev/null | grep -q . ); then \
	  pytest -q ; \
	else \
	  echo "No tests found in ./tests or ./test — skipping" ; \
	fi

validate:
	@if [ -n "$(INPUT)" ]; then \
	  $(PYTHON) -m $(MODULE).validate --config $(CONFIG) --schema $(SCHEMA) --input "$(INPUT)" ; \
	else \
	  $(PYTHON) -m $(MODULE).validate --config $(CONFIG) --schema $(SCHEMA) ; \
	fi

validate_latest:
	@if ls -1t $(PROCESSED_DIR)/*.csv >/dev/null 2>&1; then \
	  PROC=$$(ls -1t $(PROCESSED_DIR)/*.csv | head -n1); \
	  echo "Validating: $$PROC"; \
	  $(MAKE) validate INPUT="$$PROC"; \
	else \
	  echo "No processed CSV in $(PROCESSED_DIR)" >&2; exit 1; \
	fi

clean:
	rm -rf .ruff_cache .pytest_cache build dist *.egg-info mlruns
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

# --- Optional data helpers (guarded) ----------------------------------------
sample:
	@if [ -f scripts/sample.py ]; then \
		echo "Generating sample dataset..."; \
		$(PYTHON) scripts/sample.py --module $(MODULE) --config $(CONFIG) $(if $(N),--n $(N),); \
		echo "Done."; \
	else \
		echo "scripts/sample.py not found — skipping"; \
	fi

clean_data:
	@if [ -f scripts/clean.py ]; then \
		if [ -n "$(INPUT)" ]; then \
			$(PYTHON) scripts/clean.py --module $(MODULE) --config $(CONFIG) --input "$(INPUT)" --stem auto; \
		else \
			$(PYTHON) scripts/clean.py --module $(MODULE) --config $(CONFIG) --glob 'data/raw/*.csv' --glob 'data/sample/*.csv' --stem auto; \
		fi; \
	else \
		echo "scripts/clean.py not found — skipping"; \
	fi

# --- MLOps pipeline ---------------------------------------------------------
# Split CSV → TRAIN/VAL using ml_utils (defaults to newest CSV in PROCESSED_DIR)
# Pass custom args with ARGS, e.g.: make split ARGS='--csv data/processed/my.csv --stratify-col is_smoker'
split:
	@mkdir -p $(PROCESSED_DIR)
	$(PYTHON) -m $(MODULE).ml_utils split --dir $(PROCESSED_DIR) --out-train $(TRAIN) --out-val $(VAL) $(ARGS)

# Train via your module; it will use TRAIN/VAL if present, else fall back to newest processed CSV
train:
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) MLFLOW_EXPERIMENT_NAME=$(EXPERIMENT) \
	$(PYTHON) -m $(MODULE).train

# Evaluate latest model on VAL (or newest processed) and write reports
# Auto-detect evaluate vs evaluation module name

# Evaluate latest model on VAL (if present) or newest processed CSV.
# Uses src/$(MODULE)/evaluate.py or src/$(MODULE)/evaluation.py if present;
# otherwise falls back to python -m $(MODULE).evaluate.
eval:
	@CSV_ARG=""; \
	if [ -n "$(VAL)" ] && [ -f "$(VAL)" ]; then CSV_ARG="--csv $(VAL)"; fi; \
	if [ -f "src/$(MODULE)/evaluate.py" ]; then \
		echo ">> Using $(MODULE).evaluate"; \
		$(PYTHON) -m $(MODULE).evaluate --config $(EXPERIMENT_CFG) $$CSV_ARG --model $(EVAL_MODEL) --reports-dir $(REPORTS_DIR) --threshold $(THRESHOLD); \
	elif [ -f "src/$(MODULE)/evaluation.py" ]; then \
		echo ">> Using $(MODULE).evaluation"; \
		$(PYTHON) -m $(MODULE).evaluation --config $(EXPERIMENT_CFG) $$CSV_ARG --model $(EVAL_MODEL) --reports-dir $(REPORTS_DIR) --threshold $(THRESHOLD); \
	else \
		echo ">> No src module file found. Trying $(MODULE).evaluate..."; \
		$(PYTHON) -m $(MODULE).evaluate --config $(EXPERIMENT_CFG) $$CSV_ARG --model $(EVAL_MODEL) --reports-dir $(REPORTS_DIR) --threshold $(THRESHOLD) || \
		( echo "Neither $(MODULE).evaluate nor $(MODULE).evaluation are available." >&2; exit 1 ); \
	fi

# Convenience pipeline
model: split train eval

# Dev server (only if app/server.py exists)
serve:
	@if [ -f app/server.py ]; then \
	  uvicorn app.server:app --reload --host $(API_HOST) --port $(API_PORT); \
	else \
	  echo "app/server.py not found — skipping serve"; \
	fi

mlflow-ui:
	mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --host 0.0.0.0 --port 5000

# --- Smoke targets (optional, align with tests I provided) ------------------
smoke-scripts:
	@if [ -f tests/scripts/test_scripts_smoke.py ]; then pytest -q tests/scripts/test_scripts_smoke.py; else echo "skip"; fi

smoke-pipeline:
	@if [ -f tests/e2e/test_pipeline_smoke.py ]; then pytest -q tests/e2e/test_pipeline_smoke.py; else echo "skip"; fi

smoke-full:
	@if [ -f tests/e2e/test_full_flow_smoke.py ]; then pytest -q tests/e2e/test_full_flow_smoke.py; else echo "skip"; fi

smoke: smoke-scripts smoke-pipeline smoke-full

# CI-like pipeline
ci: clean install lint format-check validate_latest test smoke
	@echo "CI-like run complete."

# File: Makefile
# Purpose: Clean, working MLOps targets using your package + ml_utils splitter

SHELL := /bin/bash
PYTHON ?= python
PIP ?= $(PYTHON) -m pip
CONFIG ?= configs/default.yaml
EXPERIMENT_CFG ?= configs/experiment.yaml
SCHEMA ?= configs/schema.yaml
INPUT ?=
MODULE ?= $(shell if [ -d src/addiction_ds ]; then echo addiction_ds; elif [ -d src/cig_drink ]; then echo cig_drink; else echo addiction_ds; fi)
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

.PHONY: default help venv install env format format-check lint test validate validate_latest clean sample clean_data \
	split train eval model serve mlflow-ui ci

default: help

help:
	@echo "Targets: venv install env format format-check lint test validate validate_latest clean sample clean_data"
	@echo "MLOps: split train eval model serve mlflow-ui ci"
	@echo "Vars: MODULE=$(MODULE) CONFIG=$(CONFIG) SCHEMA=$(SCHEMA) INPUT=$(INPUT)"
	@echo "Data: PROCESSED_DIR=$(PROCESSED_DIR) TRAIN=$(TRAIN) VAL=$(VAL)"
	@echo "Artifacts: MODEL=$(MODEL) REPORTS_DIR=$(REPORTS_DIR)"
	@echo "Split args passthrough: make split ARGS='--csv path/to.csv --stratify-col is_smoker'"

venv:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && python -m pip install -U pip setuptools wheel

install:
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -e '.[dev]'

env:
	$(PYTHON) -m ipykernel install --user --name=$(KERNEL) || true

format:
	black .

format-check:
	black --check .

lint:
	ruff check .

test:
	if ( [ -d tests ] || [ -d test ] ) && (find tests test -type f \( -name "test_*.py" -o -name "*_test.py" \) 2>/dev/null | grep -q . ); then \
	  pytest -q ; \
	else \
	  echo "No tests found in ./tests or ./test — skipping" ; \
	fi

validate:
	if [ -n "$(INPUT)" ]; then \
	  $(PYTHON) -m $(MODULE).validate --config $(CONFIG) --schema $(SCHEMA) --input "$(INPUT)"; \
	else \
	  $(PYTHON) -m $(MODULE).validate --config $(CONFIG) --schema $(SCHEMA); \
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

eval:
	$(PYTHON) -m $(MODULE).evaluate --config $(EXPERIMENT_CFG) --csv $(VAL) --reports-dir $(REPORTS_DIR)


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

# CI-like pipeline
ci: clean install lint format-check validate_latest test
	@echo "CI-like run complete."

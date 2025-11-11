# filepath: Makefile
#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

PROJECT_NAME       := cigarette-and-drinking-data
PYTHON_VERSION     := 3.12
RUN                := conda run -n $(PROJECT_NAME)
PY                 := $(RUN) python

# Artifacts
INTERIM       := $(strip data/interim/dataset.csv)
FEATURES_CSV  := $(strip data/processed/features.csv)              # with target
PREP_INPUT    := $(strip data/interim/dataset.prep.csv)
PREP_OUTPUT   := $(strip data/processed/dataset.preprocessed.csv)  # X-only (no target)
PREP_MODEL    ?= $(strip models/preprocessor.joblib)

# Modeling knobs
TARGET      ?= has_health_issues
NUM_COLS    ?=
CAT_COLS    ?=
ENCODE_CAT  ?= 1

# Model artifacts
MODEL_DIR       := $(strip artifacts/rf)
MODEL_PATH      := $(strip $(MODEL_DIR)/model.joblib)
METRICS_PATH    := $(strip $(MODEL_DIR)/metrics.json)
PREDICTIONS_CSV := $(strip $(MODEL_DIR)/predictions.csv)

# Helpers
_mkdirs = mkdir -p $(dir $(strip $1))

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: requirements
requirements: update_environment

.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -f "$(PREP_INPUT)"

.PHONY: format
format:
	ruff check --fix
	ruff format

.PHONY: lint
lint:
	ruff format --check
	ruff check

.PHONY: test
test:
	$(PY) -m pytest -q

.PHONY: create_environment
create_environment:
	conda env create -f environment.yml || true
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	$(RUN) python -m pip install -e .

.PHONY: update_environment
update_environment:
	conda env update -n $(PROJECT_NAME) -f environment.yml --prune
	$(RUN) python -m pip install -e .

.PHONY: doctor
doctor:
	@echo ">>> Verifying interpreter and pip inside env: $(PROJECT_NAME)"
	@$(RUN) python -V
	@$(RUN) python -c "import sys; print(sys.executable)"
	@$(RUN) python -m pip -V
	@echo "OK if Python is 3.12.x and pip path is inside .../envs/$(PROJECT_NAME)/"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Data: raw -> interim (idempotent; reuses existing)
.PHONY: data
data: requirements
	@$(call _mkdirs,$(INTERIM))
	@if [ -f "$(INTERIM)" ]; then \
		echo "✔ Using existing $(INTERIM)"; \
	else \
		echo "→ Building interim dataset → $(INTERIM)"; \
		$(RUN) addiction-dataset --output-path "$(INTERIM)"; \
	fi

## Features-only: INTERIM -> FEATURES_CSV (feature engineering; keeps target)
.PHONY: features
features: data
	@$(call _mkdirs,$(FEATURES_CSV))
	@echo "→ Applying features (target=$(TARGET)) $(INTERIM) → $(FEATURES_CSV)"
	$(RUN) addiction-features \
		--input-path  "$(INTERIM)" \
		--output-path "$(FEATURES_CSV)" \
		$(if $(TARGET),--target "$(TARGET)",)

## Preprocess: INTERIM -> PREP_OUTPUT (cleanup + preprocessor; drops target)
.PHONY: preprocess
preprocess: data
	@$(call _mkdirs,$(PREP_INPUT))
	@$(call _mkdirs,$(PREP_OUTPUT))
	@$(call _mkdirs,$(PREP_MODEL))
	@echo "→ Preparing preprocess input (drop target=$(TARGET))"
	$(RUN) addiction-features \
		--input-path  "$(INTERIM)" \
		--output-path "$(PREP_INPUT)" \
		$(if $(TARGET),--target "$(TARGET)",) \
		--only basic_cleanup
	@echo "→ Preprocessing $(PREP_INPUT) → $(PREP_OUTPUT)"
	$(RUN) addiction-preprocessor \
		--mode fit-transform \
		--input-path  "$(PREP_INPUT)" \
		--output-path "$(PREP_OUTPUT)" \
		--model-path "$(PREP_MODEL)" \
		$(if $(NUM_COLS),--num-cols "$(NUM_COLS)",) \
		$(if $(CAT_COLS),--cat-cols "$(CAT_COLS)",) \
		$(if $(filter 1,$(ENCODE_CAT)),--encode-cat,--no-encode-cat) \
		$(if $(TARGET),--target "$(TARGET)",)
	@rm -f "$(PREP_INPUT)" || true

## Build (train): uses console script (applies preprocessor internally)
.PHONY: build
build: features preprocess
	@mkdir -p "$(MODEL_DIR)"
	@echo "→ Training model (target=$(TARGET)) on $(FEATURES_CSV) → $(MODEL_PATH)"
	$(RUN) addiction-train \
		--target "$(TARGET)" \
		--input-csv "$(FEATURES_CSV)" \
		--output-model "$(MODEL_PATH)" \
		--preprocessor-path "$(PREP_MODEL)"

## Predict: write predictions using saved model + preprocessor
.PHONY: predict
predict: build
	@mkdir -p "$(MODEL_DIR)"
	@echo "→ Predicting on $(FEATURES_CSV) → $(PREDICTIONS_CSV)"
	$(RUN) addiction-predict \
		--input-csv "$(FEATURES_CSV)" \
		--model-path "$(MODEL_PATH)" \
		--preprocessor-path "$(PREP_MODEL)" \
		--target "$(TARGET)" \
		--output-csv "$(PREDICTIONS_CSV)"

## Evaluate: use same preprocessor as training to handle categoricals
.PHONY: eval
eval: build
	@mkdir -p "$(dir $(METRICS_PATH))"
	$(RUN) addiction-eval \
		--model-path "$(MODEL_PATH)" \
		--preprocessor-path "$(PREP_MODEL)" \
		--target "$(TARGET)" \
		--input-csv "$(FEATURES_CSV)" \
		--output-metrics "$(METRICS_PATH)"

.PHONY: train-eval
train-eval: build eval

.PHONY: all
all: train-eval

## Remove artifacts to rebuild clean
.PHONY: reset
reset:
	rm -f "$(INTERIM)" "$(FEATURES_CSV)" "$(PREP_INPUT)" "$(PREP_OUTPUT)"
	rm -rf "$(MODEL_DIR)"
	@echo "✖ Removed $(INTERIM), $(FEATURES_CSV), $(PREP_INPUT), $(PREP_OUTPUT), $(MODEL_DIR)"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

## Show this help
.PHONY: help
help:
	@$(PY) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

## Debug: print resolved vars
.PHONY: echo-vars
echo-vars:
	@echo "ENV=$(PROJECT_NAME)"
	@echo "INTERIM=$(INTERIM)"
	@echo "FEATURES_CSV=$(FEATURES_CSV)"
	@echo "PREP_OUTPUT=$(PREP_OUTPUT)"
	@echo "PREP_MODEL=$(PREP_MODEL)"
	@echo "MODEL_PATH=$(MODEL_PATH)"
	@echo "METRICS_PATH=$(METRICS_PATH)"
	@echo "PREDICTIONS_CSV=$(PREDICTIONS_CSV)"
	@echo "TARGET=$(TARGET)"
	@echo "NUM_COLS=$(NUM_COLS)"
	@echo "CAT_COLS=$(CAT_COLS)"
	@echo "ENCODE_CAT=$(ENCODE_CAT)"

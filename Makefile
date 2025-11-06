#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME      = cigarette-and-drinking-data
PYTHON_VERSION    = 3.12
PYTHON_INTERPRETER= python

# Artifacts
INTERIM    := data/interim/dataset.csv
PROCESSED  := data/processed/dataset.csv
PREP_INPUT := data/interim/dataset.prep.csv
PREP_MODEL ?= models/preprocessor.joblib

# Modeling knobs
TARGET     ?= has_health_issues   # target to drop (also drops any '<target>_*')
NUM_COLS   ?=
CAT_COLS   ?=
ENCODE_CAT ?= 1                   # 1=enable OHE, 0=disable

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune

## Delete compiled files and temp preprocess input
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -f "$(PREP_INPUT)"

## Lint using ruff (use `make format` to auto-fix)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Auto-fix lint and format with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests

## Create conda environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Build INTERIM (raw -> interim). Reuse if present; else build from raw.
.PHONY: data
data: requirements
	@if [ -f "$(INTERIM)" ]; then \
		echo "✔ Using existing $(INTERIM)"; \
	else \
		echo "→ Building interim dataset from raw → $(INTERIM)"; \
		$(PYTHON_INTERPRETER) addiction/dataset.py \
			--output-path "$(INTERIM)"; \
	fi

## Feature engineering: INTERIM -> PROCESSED (drops TARGET + its OHEs)
.PHONY: features
features: data
	@echo "→ Applying features (target=$(TARGET)) $(INTERIM) → $(PROCESSED)"; \
	$(PYTHON_INTERPRETER) addiction/features.py \
		--input-path  "$(INTERIM)" \
		--output-path "$(PROCESSED)" \
		$(if $(strip $(TARGET)),--target "$(TARGET)",)

## Preprocess: INTERIM -> PROCESSED
## 1) Create temp PREP_INPUT by dropping target/OHEs (no other features)
## 2) Fit+transform -> PROCESSED
.PHONY: preprocess
preprocess: data
	@echo "→ Preparing preprocess input by dropping target (target=$(TARGET))"; \
	$(PYTHON_INTERPRETER) addiction/features.py \
		--input-path  "$(INTERIM)" \
		--output-path "$(PREP_INPUT)" \
		$(if $(strip $(TARGET)),--target "$(TARGET)",) \
		--only basic_cleanup; \
	echo "→ Preprocessing $(PREP_INPUT) → $(PROCESSED)"; \
	$(PYTHON_INTERPRETER) addiction/preprocessor.py \
		--mode fit-transform \
		--input-path  "$(PREP_INPUT)" \
		--output-path "$(PROCESSED)" \
		--model-path "$(PREP_MODEL)" \
		$(if $(NUM_COLS),--num-cols "$(NUM_COLS)",) \
		$(if $(CAT_COLS),--cat-cols "$(CAT_COLS)",) \
		$(if $(filter 1,$(ENCODE_CAT)),--encode-cat,--no-encode-cat); \
	rm -f "$(PREP_INPUT)" || true

## Full pipeline: raw -> interim -> features -> processed(preprocessed)
.PHONY: all
all: data features preprocess

## Remove artifacts to rebuild clean
.PHONY: reset
reset:
	rm -f "$(INTERIM)" "$(PROCESSED)" "$(PREP_INPUT)"
	@echo "✖ Removed $(INTERIM), $(PROCESSED), $(PREP_INPUT)"

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
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

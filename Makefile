#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = cigarette-and-drinking-data
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset (produces data/processed/dataset.csv)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) addiction/dataset.py


# -------- Feature engineering (addiction/features.py) --------
IN ?= data/processed/dataset.csv
OUT ?= data/processed/features.csv

## Build engineered features (IN -> OUT)
.PHONY: features
features: data
	$(PYTHON_INTERPRETER) addiction/features.py \
		--input-path "$(IN)" \
		--output-path "$(OUT)"


# -------- Preprocessing (addiction/preprocessor.py) --------
PREP_IN ?= data/processed/dataset.csv
PREP_OUT ?= data/processed/features_preprocessed.csv
PREP_MODEL ?= models/preprocessor.joblib
NUM_COLS ?=
CAT_COLS ?=
ENCODE_CAT ?= 1   # 1=enable OHE, 0=disable

## Fit preprocessor and transform in one step (PREP_IN -> PREP_OUT)
.PHONY: preprocess
preprocess: data
	$(PYTHON_INTERPRETER) addiction/preprocessor.py \
		--mode fit-transform \
		--input-path "$(PREP_IN)" \
		--output-path "$(PREP_OUT)" \
		--model-path "$(PREP_MODEL)" \
		$(if $(NUM_COLS),--num-cols "$(NUM_COLS)",) \
		$(if $(CAT_COLS),--cat-cols "$(CAT_COLS)",) \
		$(if $(filter 1,$(ENCODE_CAT)),--encode-cat,--no-encode-cat)


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

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

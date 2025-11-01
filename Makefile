#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = cigarette-and-drinking-data
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


##### Makefile (drop-in replacement for CCDS virtualenvwrapper flow) #####
.PHONY: help create_environment requirements data lint format test clean python

# --- venv paths ---
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(PY) -m pip

help:
	@echo "Targets:"
	@echo "  make create_environment  # create .venv (python3 -m venv)"
	@echo "  make requirements        # pip install -r requirements.txt"
	@echo "  make data                # run raw -> engineered features"
	@echo "  make test                # run tests"
	@echo "  make lint                # run linters"
	@echo "  make format              # auto-format"
	@echo "  make clean               # clean caches/artifacts"

python:
	@echo "Using:"; $(PY) -V || echo "Run: make create_environment && source .venv/bin/activate"

create_environment:
	@echo "Creating venv at $(VENV)"
	python3 -m venv $(VENV)
	@echo ">>> Activate with: source $(VENV)/bin/activate"

requirements: | create_environment
	@echo "Installing requirements with $(PY)"
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

# ---- pipeline targets (adjust paths if yours differ) ----
data: | requirements
	$(PY) -m addiction.data.dataset \
	  --input-path data/raw/addiction_population_data.csv \
	  --output-path data/processed/dataset.features.csv

# If you keep a separate features step; otherwise remove this target
features: | requirements
	$(PY) -m addiction.features.features \
	  --input-path data/processed/dataset.features.csv \
	  --output-path data/processed/features.parquet

test: | requirements
	$(PY) -m pytest -q

lint: | requirements
	-$(PY) -m ruff check .
	-$(PY) -m mypy addiction || true

format: | requirements
	-$(PY) -m ruff check . --fix
	-$(PY) -m black .

clean:
	rm -rf .pytest_cache .mypy_cache __pycache__ */__pycache__ .ruff_cache artifacts reports/figures *.egg-info
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.DS_Store" -delete
###########################################################################

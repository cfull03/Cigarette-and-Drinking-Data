SHELL := /bin/bash
PYTHON ?= python
PIP ?= $(PYTHON) -m pip
CONFIG ?= configs/default.yaml
SCHEMA ?= configs/schema.yaml
INPUT ?=
MODULE ?= $(shell if [ -d src/addiction_ds ]; then echo addiction_ds; elif [ -d src/cig_drink ]; then echo cig_drink; else echo addiction_ds; fi)
KERNEL ?= $(MODULE)

.PHONY: default help venv install env format format-check lint test validate validate_latest clean sample clean_data ci use-ssh use-https show-remote

default: help

help:
	@echo "Targets: venv install env format format-check lint test validate validate_latest clean sample clean_data ci"
	@echo "Vars: MODULE=$(MODULE) CONFIG=$(CONFIG) SCHEMA=$(SCHEMA) INPUT=$(INPUT)"

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
	@if ls -1t data/processed/*.csv >/dev/null 2>&1; then \
	  PROC=$$(ls -1t data/processed/*.csv | head -n1); \
	  echo "Validating: $$PROC"; \
	  $(MAKE) validate INPUT="$$PROC"; \
	else \
	  echo "No processed CSV in data/processed" >&2; exit 1; \
	fi

clean:
	rm -rf .ruff_cache .pytest_cache build dist *.egg-info
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

# --- Data helpers ---
# Usage: make sample           # defaults to 100 rows
#        make sample N=200     # choose row count
sample:
	echo "Generating sample dataset..."
	$(PYTHON) scripts/sample.py --module $(MODULE) --config $(CONFIG) $(if $(N),--n $(N),)
	echo "Done."

# Clean data pipeline (single file or all CSVs)
# Usage:
#   make clean_data                 # cleans all CSVs in data/raw + data/sample
#   make clean_data INPUT=path.csv  # cleans a specific file
clean_data:
	if [ -n "$(INPUT)" ]; then \
		$(PYTHON) scripts/clean.py --module $(MODULE) --config $(CONFIG) --input "$(INPUT)" --stem auto; \
	else \
		$(PYTHON) scripts/clean.py --module $(MODULE) --config $(CONFIG) --glob 'data/raw/*.csv' --glob 'data/sample/*.csv' --stem auto; \
	fi

# --- Remote helpers ---
REPO ?= $(shell git config --get remote.origin.url | awk -F'[:/]' '/github.com/{print $$4"/"$$5}' | sed 's/\.git$$//')

use-ssh:
	if [ -z "$(REPO)" ]; then echo "Could not detect repo from remote.origin.url"; exit 1; fi
	git remote set-url origin git@github.com:$(REPO).git
	git remote -v

use-https:
	if [ -z "$(REPO)" ]; then echo "Could not detect repo from remote.origin.url"; exit 1; fi
	git remote set-url origin https://github.com/$(REPO).git
	git remote -v

show-remote:
	git remote -v

# CI-like pipeline locally
ci: clean install lint format-check clean_data validate_latest test
	@echo "CI-like run complete."

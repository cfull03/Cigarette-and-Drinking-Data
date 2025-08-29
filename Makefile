# File: Makefile
SHELL := /bin/bash
PYTHON ?= python
PIP ?= $(PYTHON) -m pip
CONFIG ?= configs/default.yaml
SCHEMA ?= configs/schema.yaml
INPUT ?=
# auto-detect package module
MODULE ?= $(shell if [ -d src/addiction_ds ]; then echo addiction_ds; elif [ -d src/cig_drink ]; then echo cig_drink; else echo cig_drink; fi)
KERNEL ?= $(MODULE)

.PHONY: help venv install env format lint test validate clean

help:
	@echo "Targets: venv install env format lint test validate clean"
	@echo "Vars: MODULE=$(MODULE) CONFIG=$(CONFIG) SCHEMA=$(SCHEMA) INPUT=$(INPUT)"

# optional: create .venv to avoid mixing Anaconda/base with project deps
venv:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && python -m pip install -U pip setuptools wheel

install:
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -e '.[dev]'

env:
	$(PYTHON) -m ipykernel install --user --name=$(KERNEL) || true

format:
	black .

lint:
	ruff check .

test:
	@if [ -d tests ] && (find tests -type f -name "test_*.py" -o -name "*_test.py" | grep -q .); then \
		pytest -q ; \
	else \
		echo "No tests found in ./tests — skipping" ; \
	fi

validate:
	@if [ -n "$(INPUT)" ]; then \
		$(PYTHON) -m $(MODULE).validate --config $(CONFIG) --schema $(SCHEMA) --input "$(INPUT)"; \
	else \
		$(PYTHON) -m $(MODULE).validate --config $(CONFIG) --schema $(SCHEMA); \
	fi

clean:
	rm -rf .ruff_cache .pytest_cache build dist *.egg-info
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

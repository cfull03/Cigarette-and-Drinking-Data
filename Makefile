# file: Makefile

# -------- Settings --------
PY ?= python
PIP ?= pip
VENV ?= .venv
ACT := . $(VENV)/bin/activate

CONFIG ?= config/config.yaml
RAW ?= $(shell grep -m1 'raw_csv:' $(CONFIG) | awk '{print $$2}')

# -------- Phony targets --------
.PHONY: help setup setup-dev install clean train evaluate predict test lint format coverage freeze

help:
	@echo "Targets:"
	@echo "  setup        - Create venv and install runtime deps"
	@echo "  setup-dev    - Create venv and install dev deps (extras)"
	@echo "  install      - Install package in editable mode"
	@echo "  train        - Train models (config: $(CONFIG))"
	@echo "  evaluate     - Evaluate saved models"
	@echo "  predict      - Run batch predictions on RAW csv"
	@echo "  test         - Run pytest"
	@echo "  lint         - Run flake8 checks"
	@echo "  format       - Run black + isort"
	@echo "  coverage     - Run tests with coverage report"
	@echo "  freeze       - Export environment to requirements-freeze.txt"
	@echo "  clean        - Remove caches, build, and pyc files"

# -------- Environment --------
$(VENV)/bin/activate:
	$(PY) -m venv $(VENV)
	$(ACT) && $(PIP) install -U pip

setup: $(VENV)/bin/activate
	$(ACT) && $(PIP) install -e . -r requirements.txt

setup-dev: $(VENV)/bin/activate
	$(ACT) && $(PIP) install -e ".[dev]" -r requirements.txt

install:
	$(ACT) && $(PIP) install -e .

# -------- Pipeline --------
train:
	$(ACT) && addiction-train --config $(CONFIG)

evaluate:
	$(ACT) && addiction-evaluate --config $(CONFIG)

predict:
	$(ACT) && addiction-predict --config $(CONFIG) --input $(RAW)

# -------- Quality --------
test:
	$(ACT) && pytest -q

lint:
	$(ACT) && flake8 src tests

format:
	$(ACT) && isort src tests && black src tests

coverage:
	$(ACT) && coverage run -m pytest && coverage report -m

freeze:
	$(ACT) && pip freeze > requirements-freeze.txt

ingest:
	$(ACT) && addiction-ingest --config $(CONFIG) --schema config/schema.yaml

# -------- Cleanup --------
clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + || true
	find . -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" -delete || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist || true

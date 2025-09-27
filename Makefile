# file: Makefile

# -------- Settings --------
PY ?= python
VENV ?= .venv
VENV_BIN := $(VENV)/bin
VENV_PY  := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip

CONFIG ?= config/config.yaml
RAW ?= $(shell grep -m1 'raw_csv:' $(CONFIG) | awk '{print $$2}')

# -------- Phony --------
.PHONY: help setup setup-dev install clean train evaluate predict test lint format coverage freeze ingest watch

help:
	@echo "setup        - create venv + install runtime deps"
	@echo "setup-dev    - create venv + install dev deps"
	@echo "install      - pip install -e . (uses venv)"
	@echo "train/evaluate/predict/ingest/watch - pipeline targets"
	@echo "test/lint/format/coverage/freeze/clean"

# -------- Environment --------
$(VENV_BIN)/python:
	$(PY) -m venv $(VENV)
	$(VENV_PY) -m pip install -U pip

setup: $(VENV_BIN)/python
	$(VENV_PIP) install -e . -r requirements.txt

setup-dev: $(VENV_BIN)/python
	$(VENV_PIP) install -e ".[dev]" -r requirements.txt

install: $(VENV_BIN)/python
	$(VENV_PIP) install -e .

# -------- Pipeline --------
train:
	$(VENV_BIN)/addiction-train --config $(CONFIG)

evaluate:
	$(VENV_BIN)/addiction-evaluate --config $(CONFIG)

predict:
	$(VENV_BIN)/addiction-predict --config $(CONFIG) --input $(RAW)

ingest:
	$(VENV_BIN)/addiction-ingest --config $(CONFIG) --schema config/schema.yaml

watch:
	$(VENV_BIN)/addiction-watch --config $(CONFIG) --schema config/schema.yaml

# -------- Quality --------
test:
	$(VENV_BIN)/pytest -q

lint:
	$(VENV_BIN)/flake8 src tests

format:
	$(VENV_BIN)/isort src tests && $(VENV_BIN)/black src tests

coverage:
	$(VENV_BIN)/coverage run -m pytest && $(VENV_BIN)/coverage report -m

freeze:
	$(VENV_PIP) freeze > requirements-freeze.txt

# -------- Cleanup --------
clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + || true
	find . -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" -delete || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist || true

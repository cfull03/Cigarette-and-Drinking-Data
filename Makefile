# -------- Settings --------
PY ?= python
VENV ?= .venv
VENV_BIN := $(VENV)/bin
VENV_PY  := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip

CONFIG ?= config/config.yaml
RAW ?= $(shell grep -m1 'raw_csv:' $(CONFIG) | awk '{print $$2}')

# Ingest/Watch transform flags
OUTLIERS ?= iqr
IQR_K ?= 1.5
SAVE_CSV ?= 0          # set to 1 to also write CSV
NO_PARQUET ?= 0        # set to 1 to disable parquet

# Docker
IMAGE ?= cigarette-and-drinking-data:latest
COMPOSE ?= docker compose

# -------- Phony --------
.PHONY: help setup setup-dev install clean train evaluate predict test lint format coverage freeze ingest watch \
        docker-build docker-watch docker-ingest docker-shell docker-logs docker-down

help:
	@echo "setup        - create venv + install runtime deps"
	@echo "setup-dev    - create venv + install dev deps"
	@echo "install      - pip install -e . (uses venv)"
	@echo "train/evaluate/predict/ingest/watch - pipeline targets"
	@echo "docker-build - build the image"
	@echo "docker-watch - run watcher service (compose)"
	@echo "docker-ingest- run one-off ingest job (compose)"
	@echo "docker-shell - interactive shell inside image"
	@echo "docker-logs  - tail watcher logs"
	@echo "docker-down  - stop services"
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

# -------- Pipeline (venv) --------
train:
	$(VENV_BIN)/addiction-train --config $(CONFIG)

evaluate:
	$(VENV_BIN)/addiction-evaluate --config $(CONFIG)

predict:
	$(VENV_BIN)/addiction-predict --config $(CONFIG) --input $(RAW)

ingest:
	$(VENV_BIN)/addiction-ingest --config $(CONFIG) --schema config/schema.yaml \
		--outliers $(OUTLIERS) --iqr-k $(IQR_K) $(if $(filter 1,$(SAVE_CSV)),--save-csv) $(if $(filter 1,$(NO_PARQUET)),--no-parquet)

watch:
	$(VENV_BIN)/addiction-watch --config $(CONFIG) --schema config/schema.yaml \
		--outliers $(OUTLIERS) --iqr-k $(IQR_K) $(if $(filter 1,$(SAVE_CSV)),--save-csv) $(if $(filter 1,$(NO_PARQUET)),--no-parquet)

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

# -------- Docker / Compose --------
docker-build:
	$(COMPOSE) build

docker-watch:
	OUTLIERS=$(OUTLIERS) IQR_K=$(IQR_K) SAVE_CSV=$(SAVE_CSV) NO_PARQUET=$(NO_PARQUET) $(COMPOSE) up -d watcher

docker-ingest:
	OUTLIERS=$(OUTLIERS) IQR_K=$(IQR_K) SAVE_CSV=$(SAVE_CSV) NO_PARQUET=$(NO_PARQUET) $(COMPOSE) run --rm ingest-once

docker-shell:
	docker run --rm -it -v $$(pwd):/app -w /app $(IMAGE) bash

docker-logs:
	$(COMPOSE) logs -f watcher

docker-down:
	$(COMPOSE) down

# -------- Cleanup --------
clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + || true
	find . -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" -delete || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist || true

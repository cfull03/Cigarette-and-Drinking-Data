# file: Makefile (hybrid: local venv by default; container when DOCKER=1)

# ---------- Config ----------
DOCKER ?= 0                 # set DOCKER=1 to run inside container
IMAGE  ?= cadd-data:latest
DC     ?= docker compose

PY ?= python
VENV ?= .venv
VENV_BIN := $(VENV)/bin
VENV_PY  := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip

CONFIG ?= config/config.yaml
RAW ?= $(shell grep -m1 'raw_csv:' $(CONFIG) | awk '{print $$2}')
STAMP := $(VENV)/.deps.stamp

# ---------- Local env (used when DOCKER=0) ----------
$(VENV_BIN)/python:
	$(PY) -m venv $(VENV)
	$(VENV_PY) -m pip install -U pip setuptools wheel certifi

$(STAMP): $(VENV_BIN)/python requirements.txt pyproject.toml
	$(VENV_PIP) install -e . -r requirements.txt --prefer-binary
	@touch $(STAMP)

# ---------- Docker build ----------
.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE) .

# ---------- Command selectors ----------
ifeq ($(DOCKER),1)
RUN_TRAIN     = $(DC) run --rm train
RUN_EVAL      = $(DC) run --rm evaluate
RUN_PREDICT   = INPUT=$(RAW) $(DC) run --rm predict
RUN_INGEST    = $(DC) run --rm ingest
RUN_WATCH     = $(DC) up watch
RUN_TEST      = $(DC) run --rm test
RUN_LINT      = $(DC) run --rm test bash -lc "flake8 src tests"
RUN_FORMAT    = $(DC) run --rm test bash -lc "isort src tests && black src tests"
RUN_COVERAGE  = $(DC) run --rm test bash -lc "coverage run -m pytest && coverage report -m"
else
RUN_TRAIN     = $(VENV_BIN)/addiction-train --config $(CONFIG)
RUN_EVAL      = $(VENV_BIN)/addiction-evaluate --config $(CONFIG)
RUN_PREDICT   = $(VENV_BIN)/addiction-predict --config $(CONFIG) --input $(RAW)
RUN_INGEST    = $(VENV_BIN)/addiction-ingest --config $(CONFIG) --schema config/schema.yaml
RUN_WATCH     = $(VENV_BIN)/addiction-watch --config $(CONFIG) --schema config/schema.yaml
RUN_TEST      = $(VENV_BIN)/pytest -q
RUN_LINT      = $(VENV_BIN)/flake8 src tests
RUN_FORMAT    = $(VENV_BIN)/isort src tests && $(VENV_BIN)/black src tests
RUN_COVERAGE  = $(VENV_BIN)/coverage run -m pytest && $(VENV_BIN)/coverage report -m
endif

# ---------- Public targets ----------
.PHONY: help setup setup-dev install train evaluate predict ingest watch test lint format coverage clean

help:
	@echo "Use DOCKER=1 to run targets inside the container (default: local venv)."
	@echo "Examples:"
	@echo "  make setup           | DOCKER=0 (local)"
	@echo "  DOCKER=1 make test   | run tests in container"
	@echo "  make docker-build    | build image"
	@echo ""
	@echo "Targets: setup, setup-dev, install, train, evaluate, predict, ingest, watch, test, lint, format, coverage, clean"

# Local-only setup (ignored when DOCKER=1)
setup: $(STAMP)
setup-dev: $(STAMP)
install: $(STAMP)

train:
	$(RUN_TRAIN)

evaluate:
	$(RUN_EVAL)

predict:
	$(RUN_PREDICT)

ingest:
	$(RUN_INGEST)

watch:
	$(RUN_WATCH)

test:
	$(RUN_TEST)

lint:
	$(RUN_LINT)

format:
	$(RUN_FORMAT)

coverage:
	$(RUN_COVERAGE)

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + || true
	find . -name "*.py[co]" -delete || true
	rm -rf .pytest_cache htmlcov .coverage build dist || true

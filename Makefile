# File: Makefile
# Usage: make <target>
# Shell and common variables
SHELL := /bin/bash
PYTHON ?= python
PIP ?= pip
CONFIG ?= configs/default.yaml
SCHEMA ?= configs/schema.yaml
INPUT ?=
KERNEL ?= cig_drink

.PHONY: help setup install precommit env lint format test validate clean

help:
	@echo "Targets: setup install precommit env lint format test validate clean"
	@echo "Vars: CONFIG=$(CONFIG) SCHEMA=$(SCHEMA) INPUT=$(INPUT) KERNEL=$(KERNEL)"

setup: install precommit

install:
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PIP) install -e '.[dev]'

precommit:
	pre-commit install || true

env:
	# Why: reproducible notebook kernel name for this project
	$(PYTHON) -m ipykernel install --user --name=$(KERNEL) || true

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check --fix .

test:
	pytest -q

# Validate CSV against schema/default; use INPUT to override auto path selection
validate:
	@if [ -n "$(INPUT)" ]; then \
		$(PYTHON) -m cig_drink.validate --config $(CONFIG) --schema $(SCHEMA) --input "$(INPUT)"; \
	else \
		$(PYTHON) -m cig_drink.validate --config $(CONFIG) --schema $(SCHEMA); \
	fi

clean:
	rm -rf .pytest_cache .ruff_cache build dist **/__pycache__ *.egg-info

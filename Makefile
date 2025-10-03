# ================================
# Makefile (containers-only)
# ================================
SHELL := /bin/bash

# Compose files
COMPOSE       ?= docker compose
DEV_COMPOSE   ?= docker-compose.dev.yml
PROD_COMPOSE  ?= docker-compose.yml

# Images
IMAGE_DEV  ?= cigarette-and-drinking-data:dev
IMAGE_PROD ?= cigarette-and-drinking-data:latest

.PHONY: dev up down logs test build ps clean

## Start Jupyter dev stack (Dockerfile.dev)
dev:
	$(COMPOSE) -f $(DEV_COMPOSE) up --build

## Start production watcher stack (Dockerfile)
up:
	$(COMPOSE) -f $(PROD_COMPOSE) up -d --build

## Stop production watcher stack
down:
	$(COMPOSE) -f $(PROD_COMPOSE) down

## Tail watcher logs
logs:
	$(COMPOSE) -f $(PROD_COMPOSE) logs -f watcher

## Run pytest inside the dev container
test:
	$(COMPOSE) -f $(DEV_COMPOSE) run --rm dev pytest -q

## Build dev + prod images
build:
	docker build -f Dockerfile.dev -t $(IMAGE_DEV) .
	docker build -f Dockerfile -t $(IMAGE_PROD) .

## Show running services (prod compose)
ps:
	$(COMPOSE) -f $(PROD_COMPOSE) ps

## Clean: stop stacks + prune dangling images/build cache (keeps volumes/data)
clean:
	-$(COMPOSE) -f $(PROD_COMPOSE) down
	-$(COMPOSE) -f $(DEV_COMPOSE) down
	docker image prune -f
	docker builder prune -f

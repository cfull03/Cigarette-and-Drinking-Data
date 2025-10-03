# =========================================================
# file: Dockerfile         (EDITED: ensure tini; lean runtime)
# =========================================================
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OS deps + tini (required by ENTRYPOINT)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer-cached deps
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# Project install (immutable in runtime)
COPY pyproject.toml ./pyproject.toml
COPY src ./src
RUN pip install .

# Config + expected runtime dirs
COPY config ./config
RUN mkdir -p /app/data/raw /app/data/interim /app/data/processed /app/reports /app/models

# Default: run ingest; override in compose/CLI as needed
ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["addiction-ingest","--config","config/config.yaml","--schema","config/schema.yaml"]
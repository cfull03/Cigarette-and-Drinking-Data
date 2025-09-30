# file: Dockerfile
# -------- builder: build project wheel(s) --------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Copy metadata first for better caching
COPY pyproject.toml ./
# Copy the rest of the project
COPY . .
# Build wheels for the project (and transitive deps if present)
RUN python -m pip install --upgrade pip wheel setuptools \
 && pip wheel --wheel-dir /wheels .

# -------- runtime: self-contained app --------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# App root inside container
WORKDIR /app

# Install from wheels built in builder stage
COPY --from=builder /wheels /wheels
RUN python -m pip install --upgrade pip \
 && pip install --no-index /wheels/*.whl

# 🔽 Bake config into the image so no bind mounts are required
# (Ensure .dockerignore does NOT exclude config/)
COPY config/ /app/config/

# Create writable directories for artifacts
RUN mkdir -p /app/data/interim /app/data/processed /app/reports /app/models

# Use tini for proper signal handling and reaping
ENTRYPOINT ["/usr/bin/tini","-g","--"]

# Default command: batch ingest (you can override to run the watcher)
CMD ["addiction-ingest","--config","config/config.yaml","--schema","config/schema.yaml"]

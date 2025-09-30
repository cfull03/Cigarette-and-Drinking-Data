# File: Dockerfile
# -------- base --------
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl ca-certificates tini \
  && rm -rf /var/lib/apt/lists/*

# -------- builder (wheels) --------
FROM base AS builder
WORKDIR /src
# Copy metadata first for caching
COPY pyproject.toml ./
COPY requirements.txt ./
# Bring the code
COPY . .
RUN python -m pip install --upgrade pip wheel setuptools \
 && pip wheel --wheel-dir /wheels .

# -------- runtime --------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/app/.local/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends tini ca-certificates \
  && rm -rf /var/lib/apt/lists/*
# Non-root user
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} app && useradd -m -u ${APP_UID} -g ${APP_GID} app
WORKDIR /app

# Install from wheels produced in builder
COPY --from=builder /wheels /wheels
RUN python -m pip install --upgrade pip \
 && pip install --no-index --find-links /wheels addiction-ds

# Copy only runtime assets that must exist inside image (optional; config/data are mounted)
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
USER app
ENTRYPOINT ["/usr/bin/tini","--","entrypoint.sh"]

# Default command can be overridden by compose
CMD ["ad-ingest","--config","config/config.yaml","--schema","config/schema.yaml"]

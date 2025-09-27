# file: Dockerfile
# ---------- Base ----------
FROM python:3.12-slim AS base
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/mpl
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app

# ---------- Deps layer (cacheable) ----------
FROM base AS deps
COPY requirements.txt pyproject.toml ./
RUN python -m pip install -U pip setuptools wheel certifi && \
    pip install --prefer-binary -r requirements.txt && \
    pip install --prefer-binary -e .

# ---------- App image ----------
FROM base AS app
COPY --from=deps /usr/local /usr/local
# Copy source last for better layer caching during dev
COPY src ./src
COPY config ./config
COPY models ./models
COPY reports ./reports
COPY data ./data
COPY tests ./tests
COPY Makefile README.md ./
RUN mkdir -p /app/reports/figures /app/data/interm /app/data/processed && \
    chown -R appuser:appuser /app
USER appuser
WORKDIR /app
# Helpful default
CMD ["bash","-lc","addiction-train --help && echo && addiction-predict --help"]

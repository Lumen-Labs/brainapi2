# syntax=docker/dockerfile:1.4

# ── Stage 1: Console builder ────────────────────────────────
FROM node:22.22.0-bookworm-slim AS console-builder

WORKDIR /console
COPY console/package.json console/package-lock.json ./
RUN npm ci
COPY console/ ./
RUN npm run build

# ── Stage 2: Python builder ─────────────────────────────────
FROM python:3.11.14-slim-bookworm AS builder

ARG INSTALL_LOCAL_ML=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_RETRIES=8 \
    PIP_TIMEOUT=900 \
    PIP_DEFAULT_TIMEOUT=900 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    HF_HOME=/app/.cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==2.1.3

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN if [ "$INSTALL_LOCAL_ML" = "true" ]; then \
        poetry sync --no-root --only main --with local-ml; \
    else \
        poetry sync --no-root --only main; \
    fi \
    && mkdir -p "$TIKTOKEN_CACHE_DIR" \
    && /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && rm -rf /root/.cache /tmp/*

COPY src/ ./src/
COPY plugins/ ./plugins/
COPY scripts/preload_docker_models.py ./scripts/

RUN /app/.venv/bin/python scripts/preload_docker_models.py --spacy-only \
    && if [ "$INSTALL_LOCAL_ML" = "true" ]; then \
        /app/.venv/bin/python scripts/preload_docker_models.py --local-ml-only; \
    fi \
    && rm -rf /root/.cache /tmp/*

# ── Stage 3: runtime ────────────────────────────────────────
FROM python:3.11.14-slim-bookworm

ARG BUILD_DATE
ARG BUILD_SHA=unknown
ARG BUILD_VERSION=

LABEL build_date="${BUILD_DATE}" \
      build_sha="${BUILD_SHA}" \
      build_version="${BUILD_VERSION}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    PIP_NO_CACHE_DIR=1 \
    PIP_RETRIES=8 \
    PIP_TIMEOUT=900 \
    PIP_DEFAULT_TIMEOUT=900 \
    BUILD_SHA=${BUILD_SHA} \
    BUILD_VERSION=${BUILD_VERSION}

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    curl \
    util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip uninstall -y setuptools wheel \
    && groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.cache /app/.cache
COPY --from=builder /app/src /app/src
COPY --from=builder /app/plugins /app/plugins
COPY --from=builder /app/pyproject.toml /app/
COPY --from=console-builder /console/dist /app/console/dist
COPY LICENSE /app/LICENSE
COPY licenses/ /app/licenses/
COPY entrypoint.sh ./

RUN chmod +x /app/entrypoint.sh

RUN mkdir -p /app/custom-plugins /app/ingest-publication-locks \
    && chown -R appuser:appuser /app/plugins /app/custom-plugins /app/ingest-publication-locks
VOLUME ["/app/custom-plugins"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

USER root
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["-m", "uvicorn", "src.services.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--access-log", "--log-level", "info"]

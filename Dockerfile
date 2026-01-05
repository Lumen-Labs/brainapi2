# Multi-stage build for brainapi2
FROM python:3.11-slim AS builder

ARG BUILD_DATE
ARG BUILD_SHA
ARG CACHE_BUST

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN mkdir -p /app/.cache && chown -R appuser:appuser /app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_VENV_PATH=/app/.venv

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.in-project true && \
    poetry lock --no-update && \
    poetry install --no-root --sync && \
    rm -rf $POETRY_CACHE_DIR && \
    .venv/bin/pip uninstall -y torch && \
    .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Production stage
FROM python:3.11-slim AS production

ARG BUILD_DATE
ARG BUILD_SHA
ARG CACHE_BUST

LABEL build_date="${BUILD_DATE}" \
      build_sha="${BUILD_SHA}" \
      cache_bust="${CACHE_BUST}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

COPY --chown=appuser:appuser src/ ./src/

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Default command
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "src.services.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--access-log", "--log-level", "info"]

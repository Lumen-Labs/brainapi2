# syntax=docker/dockerfile:1.4

# ── Stage 1: builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_RETRIES=8 \
    PIP_TIMEOUT=900 \
    PIP_DEFAULT_TIMEOUT=900 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==2.1.3

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry sync --no-root \
    && TORCH_VER=$(/app/.venv/bin/python -c "import torch; print(torch.__version__.split('+')[0])") \
    && TV_VER=$(/app/.venv/bin/python -c "import torchvision; print(torchvision.__version__.split('+')[0])") \
    && /app/.venv/bin/pip install --no-cache-dir --no-deps --force-reinstall \
       "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
       --index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /root/.cache /tmp/*

COPY src/ ./src/

RUN /app/.venv/bin/python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
SentenceTransformer('intfloat/e5-small')" \
    && /app/.venv/bin/python -m spacy download en_core_web_sm \
    && rm -rf /root/.cache /tmp/*

# ── Stage 2: runtime ────────────────────────────────────────
FROM python:3.11-slim

ARG BUILD_DATE
ARG BUILD_SHA

LABEL build_date="${BUILD_DATE}" \
      build_sha="${BUILD_SHA}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache \
    PIP_NO_CACHE_DIR=1 \
    PIP_RETRIES=8 \
    PIP_TIMEOUT=900 \
    PIP_DEFAULT_TIMEOUT=900

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && GOSU_VERSION=1.19 \
    && dpkgArch="$(dpkg --print-architecture | awk -F- '{ print $NF }')" \
    && curl -fsSL -o /usr/local/bin/gosu \
       "https://github.com/tianon/gosu/releases/download/${GOSU_VERSION}/gosu-${dpkgArch}" \
    && chmod +x /usr/local/bin/gosu \
    && groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.cache /app/.cache
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/
COPY entrypoint.sh ./

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/docs || exit 1

USER root
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["-m", "uvicorn", "src.services.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--access-log", "--log-level", "info"]

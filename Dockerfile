# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

ARG BUILD_DATE
ARG BUILD_SHA

LABEL build_date="${BUILD_DATE}" \
      build_sha="${BUILD_SHA}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    TRANSFORMERS_CACHE=/app/.cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    PIP_RETRIES=8 \
    PIP_TIMEOUT=900 \
    PIP_DEFAULT_TIMEOUT=900

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* && \
    pip install poetry==2.1.3 && \
    groupadd -r appuser && useradd -r -g appuser -m appuser && \
    GOSU_VERSION=1.19 && \
    dpkgArch="$(dpkg --print-architecture | awk -F- '{ print $NF }')" && \
    wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/${GOSU_VERSION}/gosu-${dpkgArch}" && \
    chmod +x /usr/local/bin/gosu

WORKDIR /app

COPY --chown=root:root pyproject.toml poetry.lock ./
COPY --chown=root:root src/ ./src/
COPY --chown=root:root entrypoint.sh ./

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=900s --retries=5 \
    CMD curl -f http://localhost:8000/docs || exit 1

USER root
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["-m", "uvicorn", "src.services.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--access-log", "--log-level", "info"]

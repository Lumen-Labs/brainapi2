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
    PIP_RETRIES=5 \
    PIP_TIMEOUT=300

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    pip install poetry==2.1.3 && \
    groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /app

COPY --chown=appuser:appuser pyproject.toml poetry.lock ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser entrypoint.sh ./

RUN chmod +x /app/entrypoint.sh && \
    mkdir -p /app/.cache && \
    poetry config virtualenvs.in-project true && \
    poetry sync --no-root && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["-m", "uvicorn", "src.services.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--access-log", "--log-level", "info"]

#!/bin/bash
set -e

if [ ! -f "/app/.venv/bin/python" ] || ! /app/.venv/bin/python -c "import torch; import sentence_transformers; import spacy" 2>/dev/null; then
    echo "Installing dependencies..."
    mkdir -p /app/.venv /app/.cache
    if [ -d "/app/.venv" ] && [ "$(ls -A /app/.venv 2>/dev/null)" ]; then
        find /app/.venv -mindepth 1 -delete
    fi
    poetry config virtualenvs.in-project true
    poetry sync --no-root
    chown -R appuser:appuser /app/.venv /app/.cache
fi

echo "Checking for sentence-transformers model..."
gosu appuser /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
echo "Ready."

exec gosu appuser /app/.venv/bin/python "$@"


#!/bin/bash
set -e

if [ ! -d "/app/.venv" ] || [ ! -f "/app/.venv/bin/python" ]; then
    echo "Installing dependencies..."
    poetry config virtualenvs.in-project true
    poetry install --no-root --sync
    .venv/bin/pip uninstall -y torch
    .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
fi

echo "Checking for sentence-transformers model..."
/app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
echo "Ready."

exec /app/.venv/bin/python "$@"


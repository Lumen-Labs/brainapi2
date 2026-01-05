#!/bin/bash
set -e

if [ ! -f "/app/.venv/bin/python" ] || ! /app/.venv/bin/python -c "import sentence_transformers" 2>/dev/null; then
    echo "Installing dependencies..."
    poetry config virtualenvs.in-project true
    poetry lock --no-update
    poetry install --no-root --sync
    /app/.venv/bin/pip uninstall -y torch || true
    /app/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
fi

echo "Checking for sentence-transformers model..."
/app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
echo "Ready."

exec /app/.venv/bin/python "$@"


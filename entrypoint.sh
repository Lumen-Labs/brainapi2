#!/bin/bash
set -e

if [ ! -f "/app/.venv/bin/python" ] || ! /app/.venv/bin/python -c "import torch; import sentence_transformers; import spacy" 2>/dev/null; then
    echo "Installing dependencies..."
    poetry config virtualenvs.in-project true
    poetry lock
    poetry install --no-root --sync
fi

echo "Checking for sentence-transformers model..."
/app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
echo "Ready."

exec /app/.venv/bin/python "$@"


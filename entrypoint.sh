#!/bin/bash
set -e

if [ -d /app/.cache ]; then
    chown -R appuser:appuser /app/.cache 2>/dev/null || true
fi

exec gosu appuser /app/.venv/bin/python "$@"

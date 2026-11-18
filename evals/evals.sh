#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
PYTHON="${EVAL_PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${ROOT}/.venv/bin/python3"
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing ${ROOT}/.venv. Create it with:"
  echo "  cd evals && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m evals "$@"

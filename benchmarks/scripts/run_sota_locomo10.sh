#!/usr/bin/env bash
# Full LoCoMo10 SOTA leaderboard run (requires live BrainAPI + DeepSeek keys).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export BENCH_PROFILE="${BENCH_PROFILE:-sota}"
export BENCH_SC_SAMPLES="${BENCH_SC_SAMPLES:-5}"
export BENCH_GAP_FILL="${BENCH_GAP_FILL:-1}"
# Preferred answerer/judge: deepseek-v4-flash (maintainer: stronger than gpt-4o here).
export BENCH_SOTA_ANSWER_MODEL="${BENCH_SOTA_ANSWER_MODEL:-deepseek-v4-flash}"
export BENCH_SOTA_JUDGE_MODEL="${BENCH_SOTA_JUDGE_MODEL:-deepseek-v4-flash}"

RUN_INGEST="${RUN_INGEST:-sota-locomo10-ingest}"
RUN_EVAL="${RUN_EVAL:-sota-locomo10-eval}"

echo "== prompt-audit =="
../.venv/bin/python -m locomo prompt-audit

echo "== ingest all samples =="
../.venv/bin/python -m locomo ingest --run "$RUN_INGEST" --concurrency 2 --granularity session

echo "== evaluate (non-adversarial) =="
../.venv/bin/python -m locomo evaluate \
  --run "$RUN_EVAL" \
  --max-facts 50 \
  --max-passages 16 \
  --historical-limit 16 \
  --use-ppr \
  --no-fact-filter \
  --concurrency 2

echo "Done. See benchmarks/runs/$RUN_EVAL/report.md"

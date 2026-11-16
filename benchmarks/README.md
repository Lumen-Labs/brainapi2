# Benchmarks for BrainAPI

Standalone HTTP-only harnesses (no `src/` imports):

| Suite | Package | Wrapper |
| --- | --- | --- |
| [LoCoMo](https://github.com/snap-research/LoCoMo) | `locomo/` | `./locomo.sh` |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | `longmemeval/` | `./longmemeval.sh` |
| [BEAM](https://github.com/mohammadtavakoli78/BEAM) | `beam/` | `./beam.sh` |
| RecSys (train-free `/retrieve/recommend`) | `recsys/` | `./recsys.sh` |
| Search | `search/` | `./search.sh` |

Shared setup: `requirements.txt`, `.env` (`BRAINPAT_TOKEN`, LLM keys). Results ledger: [`REPORTS.json`](REPORTS.json). Agent notes: [`AGENTS.md`](AGENTS.md).

---

# LoCoMo benchmark for BrainAPI

Standalone harness that:

1. Downloads the [LoCoMo](https://github.com/snap-research/LoCoMo) dataset
2. Ingests conversations into BrainAPI over HTTP (`POST /ingest/`)
3. Answers annotated QA using `POST /retrieve/context` + an LLM answerer
4. Scores with an LLM judge plus F1 / BLEU-1

This package does **not** import anything from `src/`. It only talks to a running BrainAPI instance.

## Prerequisites

- BrainAPI running locally (`make start-all` or `brainapi start`)
- `BRAIN_CREATION_ALLOWED="true"` in the server env (needed so brains like `locomoconv26` are auto-created)
- A PAT token (`BRAINPAT_TOKEN`)
- A DeepSeek API key for answering and judging (or any OpenAI-compatible provider)

Brain IDs must be alphanumeric. This harness maps `conv-26` → `locomoconv26`.

## Setup

```bash
cd benchmarks
python3 -m venv .venv
source .venv/bin/activate
# If `python` is aliased to a system binary (common on macOS), prefer the wrapper:
#   ./locomo.sh ...
# or call the venv interpreter directly:
#   .venv/bin/python -m locomo ...
pip install -r requirements.txt
cp .env.example .env
# edit .env: BRAINPAT_TOKEN, DEEPSEEK_API_KEY
```

## Quickstart

```bash
# 1. Download dataset
./locomo.sh download
./locomo.sh dataset-stats

# 2. Smoke-test ingest + retrieve against your API
./locomo.sh smoke --brain locomosmoke

# 3. Ingest one conversation (session-level; recommended start)
./locomo.sh ingest --sample conv-26 --run locomo-conv26

# 4. Evaluate a small slice first
./locomo.sh evaluate --sample conv-26 --run locomo-conv26 --limit 20

# 5. Full non-adversarial QA for that conversation
./locomo.sh evaluate --sample conv-26 --run locomo-conv26

# 6. Rebuild the report anytime (no API calls)
./locomo.sh report --run locomo-conv26
```

Equivalent: `.venv/bin/python -m locomo ...` (avoid bare `python` if you have a shell alias).

Reports land in `runs/<run_id>/`:

- `ingest.jsonl` — per-session ingest outcomes
- `answers.jsonl` — per-QA predictions and scores
- `manifest.json` — run config
- `report.md` / `report.json` — aggregated metrics

## Results ledger

[`REPORTS.json`](REPORTS.json) is the shared BrainAPI benchmark ledger (multi-suite). LoCoMo writes the `locomo` entry; successful `evaluate` / `report` upserts automatically. See [`AGENTS.md`](AGENTS.md).

## Commands

| Command | Purpose |
|---|---|
| `download` | Fetch `data/locomo10.json` |
| `dataset-stats` | Print sessions / turns / QA per sample |
| `smoke` | One-shot ingest + retrieve health check |
| `ingest` | Push sessions (or turns) into BrainAPI |
| `answer-once` | Retrieve + answer a single question |
| `selftest-metrics` | Local metric, channel-split and record-construction checks |
| `prompt-audit` | Fail if the answer prompt shares an n-gram with any gold answer |
| `evaluate` | Retrieve → answer → judge for QA items |
| `report` | Aggregate `answers.jsonl` into markdown/JSON |
| `compare` | Paired exact-McNemar comparison between two run arms |

### Useful flags

**ingest**

- `--sample conv-26` (repeatable; default = all 10)
- `--granularity session|turn` (default `session`)
- `--concurrency 2`
- `--limit-sessions N`
- `--dry-run`
- `--no-resume`
- `--run <id>` (resume / continue a named run)

**evaluate**

- `--sample conv-26`
- `--limit 20`
- `--categories 1,2,4`
- `--include-adversarial` (category 5 is skipped by default)
- `--concurrency 2`
- `--run <id>`

**compare**

- `--baseline <run>` / `--candidate <run>`, both repeatable — repeat to pool several runs of the same configuration into one arm
- `--include-adversarial`
- `--json`

## Scoring

- **Headline number**: LLM-judge accuracy excluding category 5 (adversarial), matching common LoCoMo memory-system reporting.
- Also reported: overall accuracy, per-category breakdown, mean F1, mean BLEU-1, retrieval latency p50/p95, total LLM tokens.
- Categories: `1` multi-hop, `2` temporal, `3` open-domain, `4` single-hop, `5` adversarial.
- Gold for adversarial questions is read from `adversarial_answer` when `answer` is absent. F1 and BLEU-1 are `null` (unscorable) when gold is empty rather than a perfect 1.0.
- Evidence-session recall is reported for the **graph** and **passage** channels separately as well as combined, so the event graph's contribution is visible on its own.

### Comparing two runs

Judge accuracy is noisy: 11–12% of questions flip between identical-config runs at
`temperature=0`. **Passage** retrieval metrics are deterministic for a fixed brain
and config. **Graph** session sets are a measurement only after identical-config
agreement clears the **≥95%** gate (see `compare` graph-session stability output).
Do not A/B graph EvR below that gate.

```bash
./locomo.sh compare --baseline locomo-conv26-push75-b --candidate locomo-conv26-push75-c
./locomo.sh compare \
  --baseline locomo-conv26-push75-a --baseline locomo-conv26-push75-b \
  --candidate locomo-conv26-push75-c --candidate locomo-conv26-push75-d

# Checkpoint A — identical-config graph stability (gate ≥95% session-set agreement)
./locomo.sh evaluate --sample conv-26 --brain locomoconv26clean \
  --run graph-stable-a --historical-limit 16 --max-passages 16 \
  --max-facts 50 --use-ppr --no-sufficiency-retry
./locomo.sh evaluate --sample conv-26 --brain locomoconv26clean \
  --run graph-stable-b --historical-limit 16 --max-passages 16 \
  --max-facts 50 --use-ppr --no-sufficiency-retry
./locomo.sh compare --baseline graph-stable-a --candidate graph-stable-b
```

This reports the exact McNemar p-value with the flip table (how many questions moved
to correct and how many to wrong), overall and per category. Two independent Wilson
intervals are not a substitute: they discard the pairing and most of the power.
When both arms are single runs, `compare` also prints graph-session set agreement
vs the 95% gate.

### Models

Defaults use **DeepSeek** for the answerer (override in `.env`):

- Base URL: `https://api.deepseek.com` (`BENCH_LLM_BASE_URL`)
- Answerer: `deepseek-v4-flash` (`BENCH_ANSWER_MODEL`)
- API key: `DEEPSEEK_API_KEY` (or `OPENAI_API_KEY` for other providers)

The judge is configured independently and should be a **different model family** from
the answerer, otherwise judge accuracy carries self-preference bias. `evaluate` prints a
warning and records `judge_shares_answer_family` in the manifest when the families match.

- `BENCH_JUDGE_MODEL`, `BENCH_JUDGE_BASE_URL`, `BENCH_JUDGE_API_KEY` for any OpenAI-compatible provider
- `BENCH_JUDGE_AZURE_ENDPOINT` / `BENCH_JUDGE_AZURE_KEY` / `BENCH_JUDGE_AZURE_API_VERSION` for Azure OpenAI; the repo-root `AZURE_LARGE_LLM_*` variables are picked up automatically when they are exported and no judge model is set explicitly
- With no judge configuration at all, the judge falls back to the answerer's provider and model, and the run is marked as sharing a family

Other OpenAI-compatible providers work the same way — set the base URL, model ids, and the key.

## Reproducibility

Every `ingest` and `evaluate` manifest records the git SHA and dirty flag, SHA-256 of the
answer and judge prompts, SHA-256 of the dataset file, and the resolved answerer/judge
model ids, families and providers. A prompt edit is a measurement-instrument change: it
changes `answer_prompt_sha256`, and any comparison across that boundary is invalid.

`prompt-audit` guards against tuning the answer prompt on gold answers:

```bash
./locomo.sh prompt-audit           # fails if any 3-gram of ANSWER_SYSTEM appears in a gold answer
```

`report` de-duplicates answers by `(sample_id, qa_index)`, counts errored rows, and marks
a run that failed wholesale as `status: failed` with a banner, so a broken run can no
longer render as a clean 0.0%.

## Cost and time expectations

LoCoMo conversations are long (roughly 19–32 sessions and 370–690 turns each). With `PIPELINE_MODE=accurate`, ingesting even one conversation can take a long time and burn substantial LLM tokens inside BrainAPI.

Recommendations:

1. Start with `conv-26` only.
2. Prefer `--granularity session` (default).
3. Use `--limit-sessions` / `--limit` while iterating.
4. Optionally compare a `PIPELINE_MODE=lightweight` ingest run for cost/quality tradeoffs.
5. Do **not** ingest the `qa` annotations — that leaks the test set.

Ingest and evaluate are resumable: re-running the same `--run` skips completed units / QA rows.

## Adding another benchmark

Create a sibling package under `benchmarks/`:

```text
benchmarks/
  locomo/          # LoCoMo package
  longmemeval/     # LongMemEval package
  beam/            # BEAM package
  recsys/          # RecSys / recommend package
  yourbench/       # new package with its own cli / client / metrics
  requirements.txt # shared deps, or a per-bench requirements file
```

Reuse patterns from `locomo/` / `longmemeval/` / `recsys/`:

- HTTP-only `BrainAPIClient`
- JSONL resume files under `runs/`
- Separate ingest → evaluate → report stages
- Upsert into `REPORTS.json` under `benchmarks.<suite_id>`

Keep each benchmark self-contained and free of imports from BrainAPI's `src/`.

---

# LongMemEval benchmark for BrainAPI

Sibling harness for [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (cleaned HF release).

1. Downloads cleaned LongMemEval_S (default), oracle, or M
2. Ingests each question's haystack into its own brain (`lme…`)
3. Answers via `POST /retrieve/context` + LLM (includes `question_date`)
4. Scores with official LongMemEval yes/no judge prompts by `question_type`

Protocol: [`docs/research/09-longmemeval-protocol.md`](../docs/research/09-longmemeval-protocol.md).

## Quickstart

```bash
./longmemeval.sh download --variant s
./longmemeval.sh dataset-stats --variant s
./longmemeval.sh smoke --limit 1
./longmemeval.sh ingest --run lme-s-smoke --limit 3
./longmemeval.sh evaluate --run lme-s-smoke --limit 3
./longmemeval.sh report --run lme-s-smoke
./longmemeval.sh selftest-metrics
./longmemeval.sh prompt-audit
```

### Useful flags

- `--variant s|oracle|m`
- `--question-id <id>` (repeatable)
- `--limit N` / `--limit-sessions N`
- `--concurrency N`
- `--dry-run` / `--no-resume`

### Scoring

- **Headline**: LLM yes/no accuracy over all questions (including abstention).
- Also: per-`question_type` accuracy, abstention accuracy, session recall vs `answer_session_ids` (abstention skipped), retrieval latency, LLM tokens.
- Successful runs with ≥50 scored questions upsert `benchmarks.longmemeval` in `REPORTS.json`.

Do **not** ingest gold answers or `has_answer` turn labels.

---

# BEAM benchmark for BrainAPI

Sibling harness for [BEAM](https://github.com/mohammadtavakoli78/BEAM) (ICLR 2026). Evaluates BrainAPI as long-term memory on multi-scale chats (`100K` / `500K` / `1M` / `10M`).

1. Downloads HuggingFace `Mohammadta/BEAM` (and `Mohammadta/BEAM-10M` for `--size 10M`) and normalizes under `data/beam/`
2. Ingests each batch turn into a brain (`beam100k1`, `beam10m1`, …)
3. Answers probing questions via `POST /retrieve/context` + LLM
4. Scores with BEAM rubric LLM-judge (10 abilities); headline = mean of ability means

Protocol: [`docs/research/10-beam-protocol.md`](../docs/research/10-beam-protocol.md). 10M rows concatenate interlocking plans chronologically into one chat (same `bN_tM` / `session_N` path as 1M).

## Quickstart

```bash
./beam.sh download --size 100K
./beam.sh dataset-stats --size 100K
./beam.sh smoke --size 100K --sample 1
./beam.sh ingest --size 100K --sample 1 --run beam-100k-1 --limit-turns 10
./beam.sh evaluate --size 100K --sample 1 --run beam-100k-1 --limit 5
./beam.sh report --run beam-100k-1
./beam.sh selftest
```

10M (after `benchmarks/.env` + live API; do not start long ingest casually):

```bash
./beam.sh download --size 10M
./beam.sh dataset-stats --size 10M   # read n_turns / ingest_target first
./beam.sh smoke --size 10M --sample 1 --limit-turns 2
# durable campaign: scripts/boot_beam_10m_screen.sh + beam_10m_keepalive.sh
```

### Useful flags

- `--size 100K|500K|1M|10M`
- `--sample <id>` (repeatable; id or `size/id`)
- `--limit-turns N` / `--limit N` / `--abilities a,b,…`
- `--concurrency N` (ingest default 2; BEAM 1M/10M prefer 2–3, ≤4; keep ≤ `CELERY_WORKER_CONCURRENCY`)
- `--dry-run` / `--no-resume` (resume skips completed + permanent embed-8192 fails)

### Scoring

- **Headline**: mean of 10 ability means (`headline_score`).
- `event_ordering` uses Kendall `tau_norm`; others use mean rubric `llm_judge_score`.
- Judge variant: `beam-rubric-v1-question-aware` (question-aware, float scores).
- Successful runs with ≥20 scored questions upsert `benchmarks.beam` in `REPORTS.json`.

Do **not** ingest probing questions, rubrics, or ideal answers.

**Cost warning:** even one 100K chat is ~100+ turns; start with `--limit-turns` / `--limit`.
**1M wall time:** ~40h at concurrency 1; ~20–25h at 2 with healthy API (see `docs/research/10-beam-protocol.md`).

---

# RecSys benchmark for BrainAPI

Held-out next-item HitRate/Recall@K against **train-free** `GET /retrieve/recommend` (default) or optional plugin LightGCN. Uses structured ingest into brain `demorecsys` only — never LoCoMo/BEAM brains.

1. Optional MovieLens download (`ml-100k`) or toy JSONL fixtures under `data/`
2. Maps interactions → `POST /ingest/structured` (`mode=deterministic`)
3. Ranks with `GET /retrieve/recommend` (`--backend graph`) or LightGCN (`--backend lightgcn`)
4. Scores HitRate@K / Recall@K (+ diversity); upserts only `benchmarks.recsys`

Protocol: [`docs/research/16-recsys-eval-protocol.md`](../docs/research/16-recsys-eval-protocol.md).

## Quickstart

```bash
./recsys.sh smoke --backend graph
./recsys.sh download --name ml-100k
./recsys.sh evaluate --backend graph --dataset data/movielens_100k.jsonl --max-users 50
# attributed fixture (color/material):
./recsys.sh evaluate --backend graph --dataset data/recsys_toy_attrs.jsonl --min-interactions 2
# optional offline CF:
./recsys.sh evaluate --backend lightgcn --dataset data/recsys_toy.jsonl --epochs 20
```

### Scoring

- **Headline**: `hit_rate@K` / `recall@K` (K typically 10/20).
- Also: intra-list diversity, type coverage; report `model=graph-recommend` or `lightgcn`.
- Successful scored runs upsert `benchmarks.recsys` only — do not touch other suite keys.

---

# Search benchmark for BrainAPI

Standalone HTTP-only harness for `POST /retrieve/search` (hybrid BM25 + dense). It does **not** import `src/` and does **not** call `/retrieve/context`.

Protocol: [`docs/research/18-search-eval-protocol.md`](../docs/research/18-search-eval-protocol.md).

## Prerequisites

- BrainAPI running with `SEARCH_ENABLED=true`, `DATA_DB=postgresql`, `BRAIN_CREATION_ALLOWED=true`
- `BRAINPAT_TOKEN` in `benchmarks/.env`
- Brain id starting with `searchbench` (default `searchbenchsmoke`)

A 404 on `/retrieve/search` means search is disabled — fix the server env; it is not a ranking miss.

## Quickstart

```bash
cd benchmarks
./search.sh dataset-stats
./search.sh smoke
./search.sh evaluate --fusion rrf
./search.sh evaluate --rerank plugin:cross-encoder
./search.sh report --run <run_id>
```

Toy corpus: [`data/search_toy.jsonl`](data/search_toy.jsonl) (keyword + paraphrase queries; gold via `DOCID` markers). Search ingest defaults to **chunk + embed only** (`skip_enrichment`); pass `--enrich` for Scout/Architect.

Product-search corpora (LLM-free ingest; dedicated `searchbench*` brains, not `demorecsys`):

```bash
./search.sh download --name esci
./search.sh download --name wands
./search.sh dataset-stats --dataset data/search_esci.jsonl
./search.sh evaluate --dataset data/search_esci.jsonl --brain searchbenchesci --run search-esci
./search.sh --brain searchbenchwands evaluate --dataset data/search_wands.jsonl --run search-wands-passages-k50 --channels passages --k 50 --ks 5,10,20,50
./search.sh --brain searchbenchwands evaluate --dataset data/search_wands.jsonl --run search-wands-passages-k50 --channels passages --k 50 --ks 5,10,20,50 --skip-ingest
./search.sh --brain searchbenchwandsgraph evaluate --dataset data/search_wands.jsonl --run search-wandsgraph-communities-k50 --channels communities --k 50 --ks 5,10,20,50 --ingest-graph
```

`download` writes a **subset** under `data/` (gitignored): Amazon ESCI / Shopping Queries (KDD Cup 2022, US Task 1 test) and WANDS (Wayfair). Defaults: 80 queries, 2000 products, 40 candidates per query. Frozen WANDS quality slice is `data/search_wands.jsonl` (**66** queries / **2000** docs); `--force` cannot clobber it. Raw caches live in `data/esci/` and `data/wands/`. nDCG@10 is graded (ESCI E=1 / S=0.1 / C=0.01 / I=0; WANDS Exact=1 / Partial=0.5 / Irrelevant=0). WANDS first-stage control: passages, k=50, `rerank=none`, brain `searchbenchwands`, then `--skip-ingest`. Opt-in catalog graph: new brain `searchbenchwandsgraph` only; architecture demo, do not mix with ESCI nDCG or frozen WANDS 0.823.

Optional plugins (core hybrid works if they are absent; unknown `plugin:<name>` is 400):

- `plugins/search-rerank` — `--rerank plugin:cross-encoder` (K≤10)
- `plugins/search-splade` — `--channels plugin:splade` after `POST /search-splade/index`
- `plugins/search-colbert` — `--channels plugin:colbert` after `POST /search-colbert/index`

Metrics: Recall@{5,10,20}, nDCG@10, MRR; p50/p95 of retrieve **excluding** embed RTT. Successful `evaluate` upserts `benchmarks.search` in [`REPORTS.json`](REPORTS.json). Smoke does not publish scores.

Never ingest into or wipe `locomoconv*`, `beam*`, or `demorecsys`.

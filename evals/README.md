# Use-case evaluation harness

Compares **BrainAPI** (`POST /ingest/` + `POST /retrieve/context`) with a **well-tuned hybrid search baseline** (BM25 + dense vectors + RRF or weighted fusion + reranker) on four product use cases. Both systems share the same answerer and the same prompt budget.

This suite is intentionally **not** flattering. Local/small-scale numbers are labeled as such. The report will not claim BrainAPI beats hybrid search unless Wilson 95% intervals on the same scored items do not overlap.

Academic memory/search leaderboards stay in [`benchmarks/`](../benchmarks/). This package does not import `src/`.

## Use cases

| Id | What it tests | BrainAPI on current main |
| --- | --- | --- |
| `versioned_rules` | As-of clause text and changelogs over dated regulation versions | Adapter is **pluggable**; no `as_of` API yet. Questions still run as written. |
| `policy_conflicts` | Precision/recall of contradicting statement pairs | Adapter is **pluggable** (`EVAL_CONFLICT_PATH` / `temporal_conflicts` sidecar). Pair detection is pending as a first-class product feature. |
| `expert_knowhow` | Multi-hop procedural QA over runbooks | **Runnable on current main.** |
| `decision_precedent` | Optional: prior incident decisions | Same path as know-how. |

Each dataset has `smoke` / `dev` / `test` splits and a README covering provenance and licensing.

## Hybrid baseline (must be strong, not naive RAG)

Independent of the product search stack:

1. Chunk documents (size/overlap tuned on **dev**).
2. Okapi BM25.
3. Dense vectors: hashed n-grams locally, or OpenAI embeddings when keys are set.
4. Fuse with **RRF** or **min-max weighted** fusion (grid-searched on dev).
5. Rerank the fused head: lexical coverage by default; `sentence-transformers` cross-encoder when `EVAL_RERANKER=cross-encoder`.
6. **As-of:** filter chunks by `valid_from` / `valid_to` before retrieval (this is why the baseline is strong on versioned rules).
7. **Conflicts:** retrieve near-duplicate statements from *other* docs, score numeric/negation clashes, down-rank plan-scoped or sunset articles. Threshold can be tuned on dev.

Tune grid (full): chunk size `{500,800}`, overlap `{80,120}`, fusion `{rrf,weighted}`, `rrf_k` `{10,60}`, dense weight `{0.35,0.65}`, retrieve_k `{20,40}`, rerank depth `{8,16}`. Smoke uses a tiny grid. Trials are written into `report.json`.

Naive RAG (no date filter, no rerank, no fusion) is **not** the baseline.

## Shared answerer / prompt budget

| Knob | Value |
| --- | --- |
| max passages | 8 |
| max context chars | 6000 |
| max answer tokens | 256 |
| temperature | 0 |
| system prompt | `evals/prompts.py` `ANSWER_SYSTEM` (hashed into the report) |

`--answerer extractive` (CI / local smoke) uses span/must-include extraction from retrieved chunks — no LLM. `--answerer llm` uses the configured model for **both** systems.

Scoring: exact match where the gold is short; otherwise `must_include` phrases. Optional LLM judge (`--judge`) with a fixed rubric. Cohen's kappa vs the hand-labeled sample in `datasets/judge_agreement.jsonl` is always reported against exact-match, and against the LLM judge when `--judge` is on.

## Setup

```bash
cd evals
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# edit .env — never commit it
./evals.sh selftest
./evals.sh smoke
```

`./evals.sh smoke` runs the tiny `smoke` split of every use case with the hybrid baseline. If `BRAINPAT_TOKEN` (or `EVAL_API_KEY`) is unset, the BrainAPI arm is skipped and the report stays `local/small-scale`. If the docker stack is up, the same command will ingest into `evaluc*` brains (skip-enrichment) and query `/retrieve/context`.

## Commands

```bash
./evals.sh list-datasets
./evals.sh selftest
./evals.sh smoke                         # local/small-scale, hybrid (+ BrainAPI if token present)
./evals.sh evaluate --use-case expert_knowhow --split test --systems hybrid --run ek-local
./evals.sh evaluate --use-case versioned_rules,policy_conflicts,expert_knowhow \
    --split test --systems hybrid,brainapi --answerer llm --run staging-full
./evals.sh report --run local-smoke
./evals.sh download-far                  # public-domain FAR HTML; does not invent gold labels
```

From repo root:

```bash
PYTHONPATH=. python3 -m evals smoke
PYTHONPATH=. python3 -m pytest -q evals/tests
```

## Environment variables

Do **not** hardcode or commit secrets. Staging keys are expected to already exist in the operator environment.

| Variable | Required | Purpose |
| --- | --- | --- |
| `BRAINAPI_URL` / `EVAL_BRAINAPI_URL` | for BrainAPI arm | Default `http://localhost:8000` (docker stack). Point at staging for real runs. |
| `BRAINPAT_TOKEN` / `EVAL_API_KEY` | for BrainAPI arm | PAT. Server also needs `BRAIN_CREATION_ALLOWED=true` so `evaluc*` brains can be created. |
| `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` | for `--answerer llm` / `--judge` | Shared answerer + judge. |
| `EVAL_LLM_BASE_URL` | optional | Default `https://api.deepseek.com`. |
| `EVAL_ANSWER_MODEL` | optional | Default `deepseek-v4-flash`. |
| `EVAL_JUDGE_MODEL` / `EVAL_JUDGE_API_KEY` / `EVAL_JUDGE_BASE_URL` | optional | Prefer a **different family** from the answerer. |
| `EVAL_JUDGE_AZURE_ENDPOINT` / `EVAL_JUDGE_AZURE_KEY` | optional | Azure judge; falls back to `AZURE_LARGE_LLM_*`. |
| `EVAL_EMBEDDING_PROVIDER` | optional | `hash` (default, local) or `openai`. |
| `EVAL_EMBEDDING_API_KEY` / `EVAL_EMBEDDING_BASE_URL` / `EVAL_EMBEDDING_MODEL` | staging-quality hybrid | Real dense embeddings. |
| `EVAL_RERANKER` / `EVAL_CROSS_ENCODER_MODEL` | optional | `lexical` (default) or `cross-encoder` (needs `sentence-transformers`). |
| `EVAL_AS_OF_FIELD` | optional | If OpenAPI `GetContextRequestBody` grows an as-of field, set its name (e.g. `as_of`). |
| `EVAL_CONFLICT_PATH` | optional | Future pair-conflict endpoint, e.g. `/retrieve/conflicts`. |
| `EVAL_CONFLICT_PLUGIN` | optional | Default `temporal_conflicts` (sidecar on `/retrieve/context`, **not** statement-pair gold). |
| `EVAL_RUNS_DIR` / `EVAL_SCALE` / `EVAL_BRAIN_PREFIX` | optional | Defaults: `evals/runs`, `local/small-scale`, `evaluc`. |

The harness **never** writes locomo/beam/searchbench/demorecsys brains.

## Outputs

Each run directory (`evals/runs/<run_id>/`):

- `report.json` — machine-readable results (schema_version, scale, env redacted, per-system metrics, tuning trials, capabilities).
- `report.md` — human report with the no-claim policy, Wilson intervals, p50/p95, ingest time, tokens/calls.
- `answers.jsonl` — per-question predictions.
- `<use_case>.jsonl` — same, split by use case.

Metrics: accuracy (or conflict-pair P/R/F1), Wilson 95% CI, p50/p95 end-to-end and retrieve latency, ingest time, token and HTTP/LLM call counts.

## Local vs staging

Anything produced without staging URL + real embedding/LLM keys is **`scale: local/small-scale`**. Do not paste those numbers into external claims. The CI job only runs unit tests plus a hybrid-only smoke against the in-repo fixtures (no docker, no secrets).

## Product constraints honoured

- No product behavior or feature-flag default changes.
- No deployment or CI credential changes.
- As-of and conflict BrainAPI support is probed and plugged in when present; until then those arms are labeled capability-pending, while the hybrid baseline still produces numbers.

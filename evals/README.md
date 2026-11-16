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

1. Chunk documents (size/overlap tuned on **dev** only).
2. Okapi BM25.
3. Dense vectors: **`sentence-transformers/all-MiniLM-L6-v2` by default** (no API key). OpenAI/Azure embeddings when `EVAL_EMBEDDING_PROVIDER=openai` and a key is set. **Hashed n-grams are CI-smoke only.**
4. Fuse with **RRF** or **min-max weighted** fusion (grid-searched on dev).
5. Rerank the fused head with **`cross-encoder/ms-marco-MiniLM-L-6-v2`** by default. Lexical rerank is smoke-only.
6. **As-of:** filter chunks by `valid_from` / `valid_to` before retrieval (this is why the baseline is strong on versioned rules).
7. **Conflicts:** retrieve near-duplicate statements from *other* docs, score numeric/negation clashes, down-rank plan-scoped or sunset articles. Threshold can be tuned on dev.

Install the strong stack with `pip install -r evals/requirements-strong.txt`. Reports set `baseline_tier` to `strong` or `smoke_hash`. Do not quote `smoke_hash` numbers as the well-tuned bar.

Tune grid (full): chunk size `{500,800}`, overlap `{80,120}`, fusion `{rrf,weighted}`, `rrf_k` `{10,60}`, dense weight `{0.35,0.65}`, retrieve_k `{20,40}`, rerank depth `{8,16}`. Smoke uses a tiny grid and hashed embeddings. Trials are written into `report.json`.

Naive RAG (no date filter, no rerank, no fusion) is **not** the baseline.

## Shared answerer / prompt budget

| Knob | Value |
| --- | --- |
| max passages | 8 |
| max context chars | 6000 |
| max answer tokens | 256 |
| temperature | 0 |
| system prompt | `evals/prompts.py` `ANSWER_SYSTEM` (hashed into the report) |

`--answerer extractive` (CI / local smoke) uses span/must-include extraction from retrieved chunks — no LLM. `--answerer llm` and `--judge` use **Claude Haiku 5.5 only** (`claude-haiku-5-5`) for both systems and the judge. The harness fails at load if a different chat model or DeepSeek/OpenAI/Azure chat endpoint is configured. Embeddings and the cross-encoder reranker are not LLMs.

Scoring: exact match where the gold is short; otherwise `must_include` phrases. Optional LLM judge (`--judge`) with a fixed rubric. Cohen's kappa vs the hand-labeled sample in `datasets/judge_agreement.jsonl` is always reported against exact-match, and against the LLM judge when `--judge` is on.

## Setup

```bash
cd evals
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # CI / smoke (no models)
.venv/bin/pip install -r requirements-strong.txt   # test/dev hybrid (MiniLM + cross-encoder)
cp .env.example .env
# edit .env — never commit it
./evals.sh selftest
./evals.sh smoke
```

`./evals.sh smoke` runs the tiny `smoke` split of every use case with **hashed embeddings + lexical rerank** (no model download, no secrets). If `BRAINPAT_TOKEN` (or `EVAL_API_KEY`) is unset, the BrainAPI arm is skipped and the report stays `local/small-scale`. If the docker stack is up, the same command will ingest into `evaluc*` brains with **skip-enrichment** (no BrainAPI LLM) and query `/retrieve/context`. Full ingest enrichment is `--enrich` only, and the target stack's chat LLM must be Haiku 5.5.

Test splits are ≥150 scored items for versioned rules, policy conflicts, and expert know-how. Gold production is documented in each dataset `GOLD.md`. Regenerate from public sources with `PYTHONPATH=. python3 -m evals.datasets.build_public_corpora`.

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
| `ANTHROPIC_API_KEY` / `EVAL_LLM_API_KEY` | for `--answerer llm` / `--judge` | Shared Haiku 5.5 client. Not `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`. |
| `EVAL_LLM_MODEL` | optional | Default `claude-haiku-5-5`. Any other chat model is rejected. |
| `EVAL_LLM_BASE_URL` | optional | Default `https://api.anthropic.com/v1`. DeepSeek/OpenAI/Azure chat hosts are rejected. |
| `EVAL_EMBEDDING_PROVIDER` | optional | Default `sentence-transformers`. `openai`/`azure` with a key, or `hash` for CI smoke. |
| `EVAL_EMBEDDING_API_KEY` / `EVAL_EMBEDDING_BASE_URL` / `EVAL_EMBEDDING_MODEL` | optional | OpenAI embeddings, or override the MiniLM model name. |
| `EVAL_RERANKER` / `EVAL_CROSS_ENCODER_MODEL` | optional | Default `cross-encoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`). `lexical` is smoke-only. |
| `EVAL_AS_OF_FIELD` | optional | If OpenAPI `GetContextRequestBody` grows an as-of field, set its name (e.g. `as_of`). |
| `EVAL_CONFLICT_PATH` | optional | Future pair-conflict endpoint, e.g. `/retrieve/conflicts`. |
| `EVAL_CONFLICT_PLUGIN` | optional | Default `temporal_conflicts` (sidecar on `/retrieve/context`, **not** statement-pair gold). |
| `EVAL_RUNS_DIR` / `EVAL_SCALE` / `EVAL_BRAIN_PREFIX` | optional | Defaults: `evals/runs`, `local/small-scale`, `evaluc`. |

The harness **never** writes locomo/beam/searchbench/demorecsys brains.

## Outputs

Each run directory (`evals/runs/<run_id>/`):

- `report.json` — machine-readable results (schema_version, `llm_model`, scale, env redacted, per-system metrics, tuning trials, capabilities). `llm_model` is always `claude-haiku-5-5`.
- `report.md` — human report with the no-claim policy, Wilson intervals, p50/p95, ingest time, tokens/calls.
- `answers.jsonl` — per-question predictions.
- `<use_case>.jsonl` — same, split by use case.

Metrics: accuracy (or conflict-pair P/R/F1), Wilson 95% CI, p50/p95 end-to-end and retrieve latency, ingest time, token and HTTP/LLM call counts.

## Local vs staging

Anything produced without staging URL + LLM keys is labeled **`scale: local/small-scale`** unless you set `EVAL_SCALE`. Do not paste smoke-hash numbers into external claims. The well-tuned hybrid bar is `baseline_tier: strong` (MiniLM embeddings + cross-encoder, tuned on dev). The CI job only installs `evals/requirements.txt` and runs unit tests plus a hybrid-only **smoke** split (no docker, no secrets, no model download).

## Product constraints honoured

- No product behavior or feature-flag default changes.
- No deployment or CI credential changes.
- As-of and conflict BrainAPI support is probed and plugged in when present; until then those arms are labeled capability-pending, while the hybrid baseline still produces numbers.

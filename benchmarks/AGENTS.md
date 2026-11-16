# Benchmarks — agent guidance

## Suites

| Suite | Package | Wrapper | Ledger key |
| --- | --- | --- | --- |
| **LoCoMo** | `benchmarks/locomo/` | `./locomo.sh` | `benchmarks.locomo` |
| **LongMemEval** | `benchmarks/longmemeval/` | `./longmemeval.sh` | `benchmarks.longmemeval` |
| **BEAM** | `benchmarks/beam/` | `./beam.sh` | `benchmarks.beam` |
| **RecSys** | `benchmarks/recsys/` | `./recsys.sh` | `benchmarks.recsys` |
| **Search** | `benchmarks/search/` | `./search.sh` | `benchmarks.search` |

All harnesses talk to a running BrainAPI over HTTP only (no `src/` imports, except Search `mapping.py` re-exporting the product catalog mapper). RecSys uses `POST /ingest/structured` (KB write) + **train-free** `GET /retrieve/recommend` on brain **`demorecsys` only** — never LoCoMo/BEAM brains. Optional: `plugins/features-rec` for attribute prefs; `plugins/recsys-gnn` LightGCN via `--backend lightgcn`. Search uses `POST /ingest/` + `POST /retrieve/search` on **`searchbench*` only** — never LoCoMo/BEAM/recsys brains, never `/retrieve/context` as the metric path.

### LoCoMo

Typical flow: `download` → `ingest` → `evaluate` → `report` via `./locomo.sh` or `.venv/bin/python -m locomo`.

### LongMemEval

Typical flow: `download` → `ingest` → `evaluate` → `report` via `./longmemeval.sh` or `.venv/bin/python -m longmemeval`. Protocol: `docs/research/09-longmemeval-protocol.md`.

### RecSys

Held-out next-item HitRate/Recall@K via **train-free graph recommend** (default) or optional plugin LightGCN. Flow: `download` (optional MovieLens) → structured ingest → `GET /retrieve/recommend` via `./recsys.sh --backend graph`. Brain: `demorecsys`.

```bash
./recsys.sh smoke --backend graph
./recsys.sh download --name ml-100k
./recsys.sh evaluate --backend graph --dataset data/movielens_100k.jsonl --max-users 50
# attributed fixture (color/material):
./recsys.sh evaluate --backend graph --dataset data/recsys_toy_attrs.jsonl --min-interactions 2
# optional offline CF:
./recsys.sh evaluate --backend lightgcn --dataset data/recsys_toy.jsonl --epochs 20
```

Protocol: [`docs/research/16-recsys-eval-protocol.md`](../docs/research/16-recsys-eval-protocol.md). Upserts only `benchmarks.recsys` — never wipe or write to `beam1m1clean` / `locomoconv26*`.

### Search

Hybrid BM25 + dense ranking via `POST /retrieve/search`, plus optional graph channels (`entities` / `events` / `communities`, `expand=neighbors`). Flow: ingest docs (chunk+embed; LLM enrichment skipped by default) → optional structured catalog triples (`--ingest-graph`) → map `DOCID` markers to chunk ids → search with `profile_stages`. Gold matches chunk ids or `hit.id == doc_id`. Brain: `searchbenchsmoke` (any id must start with `searchbench`). Server must have `SEARCH_ENABLED=true`, `DATA_DB=postgresql`, `BRAIN_CREATION_ALLOWED=true`.

```bash
./search.sh smoke
./search.sh evaluate --fusion rrf
./search.sh download --name esci
./search.sh download --name wands
./search.sh evaluate --dataset data/search_esci.jsonl --brain searchbenchesci --run search-esci
./search.sh --brain searchbenchwands evaluate --dataset data/search_wands.jsonl --run search-wands-passages-k50 --channels passages --k 50 --ks 5,10,20,50
./search.sh --brain searchbenchwands evaluate --dataset data/search_wands.jsonl --run search-wands-passages-k50 --channels passages --k 50 --ks 5,10,20,50 --skip-ingest
./search.sh --brain searchbenchwandsgraph evaluate --dataset data/search_wands.jsonl --run search-wandsgraph-communities-k50 --channels communities --k 50 --ks 5,10,20,50 --ingest-graph
./search.sh evaluate --dataset data/search_esci.jsonl --brain searchbenchesci --ingest-graph --channels passages,entities,communities
./search.sh evaluate --rerank plugin:cross-encoder
./search.sh report --run <run_id>
```

Catalog downloads are a **subset** of Amazon ESCI (KDD Cup 2022, US Task 1 / small_version test) and WANDS (Wayfair). Default caps: 80 queries, 2000 products, 40 candidates/query. Frozen WANDS quality slice: `data/search_wands.jsonl` (66 queries / 2000 docs); download refuses overwrite. Gold is graded (ESCI E/S/C/I; WANDS Exact/Partial/Irrelevant). WANDS control is passages k=50 `rerank=none` on `searchbenchwands`, then `--skip-ingest`. Opt-in catalog graph is brain `searchbenchwandsgraph` only (`--ingest-graph`, `channels=communities`); ledger rows are an **architecture demo**, not a claim vs 0.823 / ESCI 0.500. `--ingest-graph` is refused on frozen `searchbenchwands` / `searchbenchesci74` / `searchbenchescies` / `searchbenchesciltr2`. Pass `--enrich` only if you want Scout/Architect on ingest. Use dedicated `searchbench*` brains — never `demorecsys`. Graph mapping is product `src/core/search/catalog_graph.py` (generic HAS events + CLASS/TYPE/ATTR hubs). Do not score recs HitRate as search.

Metrics: Recall@{5,10,20}, nDCG@10, MRR; p50/p95 retrieve **excluding** embed RTT. No LLM judge. Protocol: [`docs/research/18-search-eval-protocol.md`](../docs/research/18-search-eval-protocol.md). Upserts only `benchmarks.search` — never wipe or write to `locomoconv*`, `beam*`, `demorecsys`.

### BEAM

Typical flow: `download` → `ingest` → `evaluate` → `report` via `./beam.sh` or `.venv/bin/python -m beam`.

- Dataset: HuggingFace `Mohammadta/BEAM` (`100K` / `500K` / `1M`) and `Mohammadta/BEAM-10M` (`10M`). Harness supports all four sizes in `CHAT_SIZES`.
- 10M normalize: concatenate interlocking plans chronologically into one chat (`bN_tM` / `session_N`); see [`docs/research/10-beam-protocol.md`](../docs/research/10-beam-protocol.md) § BEAM-10M. Brain `beam10m1` / `--brain beam10m1clean`; never wipe `beam1m1clean`.
- Download sample 1 once `.env` exists: `./beam.sh download --size 10M` then `./beam.sh dataset-stats --size 10M` (read `n_turns` before any ETA). Ingest/eval still need live API + explicit go (`scripts/run_beam_10m_1_ingest_eval.sh`).
- Ingest turns only — never probing questions, rubrics, or ideal answers.
- Scoring: rubric LLM-judge (`judge_prompt_variant: beam-rubric-v1-question-aware`); headline = mean of 10 ability means; `event_ordering` uses `tau_norm`.
- Brain IDs: `beam{size}{convid}` → e.g. `beam100k1`, `beam10m1`.
- Protocol: [`docs/research/10-beam-protocol.md`](../docs/research/10-beam-protocol.md).
- Ingest parallelism: prefer `--concurrency 2` (≤4); resume skips permanent embed-8192 fails. See protocol “Parallel ingest” + `scripts/run_beam_1m_1_ingest_eval.sh` / `scripts/run_beam_10m_1_ingest_eval.sh` (`BEAM_INGEST_CONCURRENCY`). Do not wipe live eval brains.

## Results ledger (`REPORTS.json`)

`benchmarks/REPORTS.json` is the public BrainAPI results ledger: top published scores for **all** benchmark suites under `benchmarks.<suite>`.

- LoCoMo harness upserts `benchmarks.locomo` via `write_report` → `update_reports_json` when report `status` is `ok`.
- BEAM harness upserts `benchmarks.beam` the same way (field: `headline_score`, continuous `[0,1]`).
- RecSys harness upserts `benchmarks.recsys` only (`hit_rate@K` / `recall@K`); it must not touch other suite keys.
- Search harness upserts `benchmarks.search` only (`ndcg@10` / `recall@10` / `p50_retrieve_ms`); it must not touch other suite keys.
- After any completed evaluate/report that produces a successful scored run, ensure the suite entry in `REPORTS.json` reflects it.
- If you write or patch `runs/<id>/report.json` manually, update `REPORTS.json` yourself (same schema; upsert by `run_id` under the suite).
- When adding a new suite, add a `benchmarks.<suite_id>` key (`name` + `leaderboard`) and upsert the same way.
- Skip failed/junk runs (`status: failed`, 0% judge, nonsense scores) — they do not belong on the leaderboard.
- Research caveats and checkpoint narrative live in `runs/CHECKPOINT_NOTES.md` (local), not in `REPORTS.json`.

## Tracks (LoCoMo)

| Track | Profile | Intent |
| --- | --- | --- |
| **SOTA** | `BENCH_PROFILE=sota` | Beat HyperMem-class LoCoMo (≥93% on full-10). Prefer **deepseek-v4-flash** answerer/judge. SC + gap-fill live in the harness, not product latency path. |
| **Product** | `BENCH_PROFILE=product` (default) | ADR-006 sub-second `/retrieve/context` path — greedy, no online LLM loops. |

Protocol and competitor bar: `docs/research/08-sota-locomo-protocol.md`. Historical checkpoint narrative: `runs/CHECKPOINT_NOTES.md`.

## Live stack

TUI / `brainapi start` may run from `~/.brainapi/source`, not this workspace. Sync or restart from the intended checkout before measuring, or scores will reflect stale code.

## Conventions

- Brain IDs are alphanumeric (`conv-26` → `locomoconv26`; BEAM `100K/1` → `beam100k1`). Prefer clean brains for product/SOTA claims after Phase 0.
- Compare LoCoMo judge accuracy with `./locomo.sh compare` (paired McNemar). Do not A/B graph EvR until identical-config session-set agreement clears the ≥95% gate.
- Do not ingest QA annotations / probing questions. Keep answer prompts prompt-audit clean where the suite provides it (`./locomo.sh prompt-audit`).

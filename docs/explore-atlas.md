# Progressive Explore API

The authenticated `/retrieve/explore/v1` API serves bounded views of a large brain. It currently supports `GRAPH_DB=networkx` (the PostgreSQL graph driver). Other drivers return 501 so the dashboard can retain bounded exploration.

- `GET /overview`: top-level relationship communities, aggregate links, snapshot identity, coverage, and build status.
- `GET /communities/{community_id}?snapshot=…&cursor=…`: child communities, or a page of 300 member entities and 300 relationships. Relationship endpoints accompany each page.
- `GET /neighbors/{entity_id}?cursor=…`: 300 actual incident relationships per page with their endpoints and the focal entity.
- `types` and `connectionTypes` are comma-separated filters. Entity-type filters apply to both relationship endpoints. Counts are evaluated before pagination.

All routes use the existing BrainPAT and brain identity middleware. The dashboard additionally applies its existing Explore permission checks, signed non-billable usage context, and rate limiter. The requested brain's own PostgreSQL database contains all overview state; no cross-brain graph is queried.

## Snapshot lifecycle

`explore_snapshots`, `explore_memberships`, and `explore_state` are created idempotently on first use. No existing graph data is rewritten. The API claims a rebuild using an atomic expiring database lease and enqueues `src.workers.tasks.explore.build_atlas` on the existing `consolidate_graph` queue. Deploy/restart workers with the new task module before enabling the frontend.

A snapshot older than five minutes refreshes on the next overview read. The previous snapshot remains usable during rebuilding. The job reads a consistent, lightweight node/edge projection, runs seeded hierarchical Louvain, and publishes its metadata and memberships in one transaction. Parallel relationships contribute weight; self-loops remain factual relationships but do not influence community projection. Isolated entities remain reachable. Representative names and dominant types supply labels without an LLM.

Build leases last 15 minutes; worker limits are 13/14 minutes. Failed builds release their lease and retain the previous snapshot. Old snapshots are retained for at least one hour, then removed during publication. Expired snapshot navigation returns 409; the dashboard offers a return to the overview. Neighborhoods and entity details remain live. Cursors bind snapshot, scope, and filters and use UUID keysets to avoid offset drift when live records change. Newly inserted records earlier than the cursor appear on refresh.

Overview links are limited to the strongest 4,000 aggregate connections, with an explicit truncation flag. Their counts describe grouped relationships, not factual entity-to-entity assertions. Member/neighbor API pages are bounded independently of the dashboard's 2,000-node / 4,000-edge display budget.

## Verification

```sh
.venv/bin/python -m pytest tests/test_explore_atlas.py -q
ATLAS_TEST_POSTGRES=1 .venv/bin/python -m pytest tests/test_explore_atlas_postgres.py -q
PYTHONPATH=. .venv/bin/python scripts/benchmark_explore_atlas.py
```

The PostgreSQL integration suite creates and removes its own randomly named `atlas_test_*` database via a local Unix socket and the current OS database role; it never uses application credentials.

Measured on the development machine on 2026-11-05: a deterministic 50,000-node / 200,000-edge fixture completed community computation in **38.1 seconds**, producing 74 root groups and complete membership coverage. This is background computation time, not an API latency measurement. Cached overview latency and production queue contention still require validation in the deployed environment.

Operational events are `explore_overview_built` (duration and counts), `explore_overview_build_failed`, and `explore_overview_enqueue_failed`. No entity content is added to these events.

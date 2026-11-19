# Changelog

## 2.21.0 — 2026-11-19

Knowledge-graph memory: expert verification, cross-source conflict detection, fail-closed as-of retrieve, validity changelogs, staged-ingest rollback, and Cypher hardening. Defaults are unchanged: raw Cypher is read-only, `CONFLICT_DETECT_ON_INGEST` and `CONFLICT_DETECT_LLM` are off, as-of lookups fail closed, and ingest staging is not required for a normal import.

Deferred: [#121](https://github.com/Lumen-Labs/brainapi2/pull/121) (agent-first MCP remember/recall/correct/check).

### Verification

- Facts carry `unverified` / `verified` / `rejected` with history. REST: `GET/POST /model/facts…`. MCP: `verify_fact`, `reject_fact`, `reset_fact_verification`, `list_facts_by_verification`, `get_fact_verification`. ([#113](https://github.com/Lumen-Labs/brainapi2/pull/113), [#125](https://github.com/Lumen-Labs/brainapi2/pull/125), [#126](https://github.com/Lumen-Labs/brainapi2/pull/126))
- Writes require the `verification` token scope (not granted by default). `method=verified_side` conflict resolve uses the same scope. ([#125](https://github.com/Lumen-Labs/brainapi2/pull/125), [#126](https://github.com/Lumen-Labs/brainapi2/pull/126))
- `verified_side` snapshots verification properties before writes. On reject failure it restores exact prior state (`verified_by`, `verified_at`, `verification_source`, reason) with a single history entry tagged `rollback`, and keeps the conflict open. Pre-read miss or error aborts before any write. ([#133](https://github.com/Lumen-Labs/brainapi2/pull/133), [#135](https://github.com/Lumen-Labs/brainapi2/pull/135))

### Conflicts

- Cross-source conflict scan, list, report, resolve, dismiss. REST under `/retrieve/conflicts`. MCP: `list_conflicts`, `get_conflict`, `resolve_conflict`, `dismiss_conflict`, `conflict_report`. ([#114](https://github.com/Lumen-Labs/brainapi2/pull/114), [#126](https://github.com/Lumen-Labs/brainapi2/pull/126))
- Ingest-time scan is incremental (touched subjects), pages past the old 10k cap, and evals scan-then-list with versioned timestamps. ([#128](https://github.com/Lumen-Labs/brainapi2/pull/128), [#132](https://github.com/Lumen-Labs/brainapi2/pull/132))
- Optional ingest/LLM detection stays **off**: `CONFLICT_DETECT_ON_INGEST=false`, `CONFLICT_DETECT_LLM=false`.

### As-of retrieve and changelogs

- Retrieve/search accept `as_of`. Undated facts are omitted (fail closed) and counted in `validity_unknown_count`; `include_validity_unknown=true` returns them separately. ([#112](https://github.com/Lumen-Labs/brainapi2/pull/112), [#124](https://github.com/Lumen-Labs/brainapi2/pull/124), [#129](https://github.com/Lumen-Labs/brainapi2/pull/129))
- Changelogs: `GET /retrieve/changelogs`, `GET /retrieve/changelogs/{id}`, `GET /retrieve/changelogs/types`. MCP: `list_changelogs`, `get_changelog`. Window filters run in Cypher; Neo4j creates per-type range indexes on validity timestamps. Index setup is cached per database (TTL or after writes that add relationship types; failed types back off and log once). ([#129](https://github.com/Lumen-Labs/brainapi2/pull/129), [#136](https://github.com/Lumen-Labs/brainapi2/pull/136))

### History and sources

- History-mode triggers keep original revisions; ranking cost is capped. ([#110](https://github.com/Lumen-Labs/brainapi2/pull/110), [#117](https://github.com/Lumen-Labs/brainapi2/pull/117), [#127](https://github.com/Lumen-Labs/brainapi2/pull/127))
- Retrieve prefers the newest copy of an app source; `list_source_versions` lists versions. ([#106](https://github.com/Lumen-Labs/brainapi2/pull/106))
- Confirmed `POST /tasks/{task_id}/purge` removes one import’s owned graph data. ([#119](https://github.com/Lumen-Labs/brainapi2/pull/119))

### Staged ingest and rollback

- Settlement failures are bounded; leftover cleanup is retried. Live staged units roll back when publish settlement is exhausted. ([#115](https://github.com/Lumen-Labs/brainapi2/pull/115), [#118](https://github.com/Lumen-Labs/brainapi2/pull/118))
- Graph rollback distinguishes created vs reused items and keys `rollback_done` as `{operation_id}:{kind}:{uuid}` so a later retouch cannot skip deleting the original create. ([#131](https://github.com/Lumen-Labs/brainapi2/pull/131), [#134](https://github.com/Lumen-Labs/brainapi2/pull/134))

### Graph store and Cypher

- Identifier interpolation is allowlisted; raw Cypher stays read-only unless `GRAPH_RAW_WRITE_ENABLED=true`. ([#108](https://github.com/Lumen-Labs/brainapi2/pull/108), [#120](https://github.com/Lumen-Labs/brainapi2/pull/120))
- `remove_nodes` decodes driver records via `Record.get`; extra invalid labels no longer drop the node. Multilingual ingest labels, LOAD CSV/APOC rejection, and unsafe purge fallbacks. ([#123](https://github.com/Lumen-Labs/brainapi2/pull/123), [#130](https://github.com/Lumen-Labs/brainapi2/pull/130))

### Evals and MCP

- Use-case eval harness vs a hybrid baseline; BrainAPI eval conflict IDs, as-of `include_history`, and passage attribution. ([#107](https://github.com/Lumen-Labs/brainapi2/pull/107), [#116](https://github.com/Lumen-Labs/brainapi2/pull/116))
- MCP version reads and multilingual history. ([#109](https://github.com/Lumen-Labs/brainapi2/pull/109))

### Defaults and flags (unchanged unless you opt in)

| Flag / setting | Default |
| --- | --- |
| Raw Cypher | read-only |
| `GRAPH_RAW_WRITE_ENABLED` | `false` |
| `CONFLICT_DETECT_ON_INGEST` | `false` |
| `CONFLICT_DETECT_LLM` | `false` |
| As-of missing valid-time start | fail closed (omit; count as unknown) |
| `include_validity_unknown` | `false` |
| Token scope `verification` | not granted by default |
| `VERIFIED_FACT_RANK_BOOST` | `0.15` |

### Known issues

- Neo4j validity index count grows as 7 × the number of relationship types.
- Changelog `MATCH (n)-[r]->(m)` with OR/regex timestamp filters likely still scans relationships despite the new indexes. Needs `EXPLAIN`/`PROFILE` on a real brain; follow-up is stamp normalization at write time or typed `UNION` queries.
- `set_fact_verification` commits then re-reads; a raising re-read can leave a rejected loser in place while reporting failure without `rollback_failed`.
- Verified-side rollback is an unconditional SET by UUID and can overwrite a concurrent verification that landed after the snapshot.
- If a restore callback raises mid-loop, later facts (including the winner) may not be restored.
- REST/MCP conflict-resolve errors can include pre-read exception text; logs should stay detailed, callers should get a fixed message.
- Cold-start index setup is not serialized per database (`CREATE RANGE INDEX IF NOT EXISTS` stays correct; overlapping first callers repeat work). A raw Cypher write during type listing can wait until the next TTL for a new type’s indexes.
- Neo4j Community Edition still shares the default `neo4j` database when `CREATE DATABASE` is unsupported.

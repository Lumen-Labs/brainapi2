# Changelog

## Unreleased

Spreadsheet experiment QA fixes. Gate is still `SPREADSHEET_EXPERIMENT_BRAINS` (empty = off). No version bump.

- Date filters compare timestamps, not text. A date-only equals matches the whole day; an inclusive end date includes that day.
- Per-brain Postgres pool waits up to `POSTGRES_POOL_TIMEOUT` (default 5s) instead of failing immediately; timeout is 503 with `Retry-After`.
- `DELETE /retrieve/spreadsheets/{id}` removes a workbook. Purge/rollback of the import task deletes workbook rows too. Identical uploads return the existing workbook id (content hash). Caps: `SPREADSHEET_MAX_WORKBOOKS_PER_BRAIN` (500), `SPREADSHEET_MAX_CELLS_PER_BRAIN`, `SPREADSHEET_MAX_BYTES_PER_BRAIN`, checked atomically with the insert.
- Imports credit existing ingest-group metering as `source_chars` / `source_tokens` (cell text) plus `counts.cells` / `counts.sheets`.
- Gated `.xlsx`/`.csv`/`.xls` that are empty, encrypted, corrupt, or malformed return 4xx instead of 202 queued.
- CSV infers dates, currency, and percentages like xlsx. Percentages stay fractions with `display=percent`. Merged cells forward-fill. Listed `rows` is data rows (header excluded). Range queries skip the header row. Workbook list is paginated. `group_by` aggregation is supported.

## 2.21.2 — 2026-11-22

Current `main` since 2.21.1. The 2.21.0 and 2.21.1 tags did not publish a container image because the tag workflow ran before the required smoke checks had succeeded. This release is the first stable tag that can publish that image, including the Cypher parameterization from 2.21.0.

### Ingest

- Plain-text file ingest skips docparser and closes OCR once every page has arrived. ([#144](https://github.com/Lumen-Labs/brainapi2/pull/144))
- OCR callbacks, Redis 503 handling, and text ingest limits are hardened. ([#146](https://github.com/Lumen-Labs/brainapi2/pull/146))
- Unstaged OCR pages are dispatched at the deadline and stored atomically. ([#148](https://github.com/Lumen-Labs/brainapi2/pull/148))

### Retrieve

- `POST` and `GET /retrieve/context` are faster for small QA brains. ([#150](https://github.com/Lumen-Labs/brainapi2/pull/150))

### Tracing

- LangSmith tracing is removed, and tracing environment variables are forced off. ([#145](https://github.com/Lumen-Labs/brainapi2/pull/145))

## 2.21.1 — 2026-11-21

Spreadsheet import/query (off by default), Explore overview-link snapshots, and hosted MCP OAuth sign-in. These were live on engine hosts only; this release puts them in the image so hosts can drop bind mounts and custom layers.

### Spreadsheets

- `.xlsx` / `.csv` uploads on `POST /ingest/file` import as structured workbooks when the brain is gated on, instead of the text ingest pipeline. Direct upload: `POST /ingest/spreadsheets`. List/query: `GET /retrieve/spreadsheets`, `POST /retrieve/spreadsheets/query`.
- Gate: `SPREADSHEET_EXPERIMENT_BRAINS` (comma-separated brain keys, `*` for all). Empty or unset disables every brain; disabled brains fall through to normal ingest and the retrieve routes return 404.
- Adds `openpyxl` and `et-xmlfile`. PostgreSQL brains only.

### Explore

- Community snapshots store capped `overviewLinks` (4000) plus `overviewLinksTruncated`. Unfiltered overview reads those snapshot counts instead of aggregating the live graph.

### MCP OAuth

- Hosted sign-in: consent is a one-time `request_id` (OAuth params stay in Redis), BrainPATs in Redis are Fernet-encrypted from `BRAINPAT_TOKEN`, origin/CSP headers, 303 redirect, and brain-scoped PAT only (system PAT cannot grant OAuth). Consent `form-action` includes the pending callback origin so non-hosted clients are not blocked after POST. That origin is built from hostname characters only (`A-Za-z0-9.-` plus an optional port). Bracketed IPv6 literal callbacks (for example `http://[::1]:8000/callback`) are not added to `form-action`, so the post-consent redirect may be blocked by CSP.
- `MCP_OAUTH_CHATGPT_ONLY=true` or a set `MCP_HOSTED_OAUTH_ISSUER_URL` restricts client registration and authorize redirects to the hosted connector callback. `#142` verification scopes, `pat_scope_resolver`, conflict/changelog tools, and `McpToolError` are unchanged.
- Refresh-token TTL still defaults to unset (no expiry) unless `MCP_OAUTH_REFRESH_TOKEN_TTL` is set. Fernet decryption failures (plaintext, corrupt, or wrong key) delete that Redis key and fail the access token so `/mcp` returns 401 and the client re-auths immediately. On startup, leftover plaintext PAT values are scanned and deleted (Fernet tokens start with `gAAAAA`; values are not logged). There is no plaintext decrypt fallback.

### Defaults and flags

| Flag / setting | Default |
| --- | --- |
| `SPREADSHEET_EXPERIMENT_BRAINS` | empty (off) |
| `MCP_OAUTH_CHATGPT_ONLY` | unset / not `true` |
| `MCP_HOSTED_OAUTH_ISSUER_URL` | empty |

### Deploy

After shipping this image, remove the two MCP bind mounts that overrode `src/services/mcp/main.py` and `src/services/mcp/oauth_provider.py`. Spreadsheet and Explore no longer need custom image layers.

**Hard requirement:** `BRAINPAT_TOKEN` and the MCP issuer (`MCP_HOSTED_OAUTH_ISSUER_URL` if set, otherwise `MCP_OAUTH_ISSUER_URL`) must stay exactly the same across this redeploy. Changing either one makes every stored PAT undecryptable. The Fernet key is HMAC-SHA256(`BRAINPAT_TOKEN`, `brainapi-mcp-oauth-pat:{issuer}`); keep that derivation as-is.

Hosts that already encrypt MCP PATs with the same `BRAINPAT_TOKEN` and issuer keep working. Leftover plaintext PATs from 2.21.0 / `#142` (or any unencrypted image) are deleted on startup and on first decrypt failure; those clients get 401 on `/mcp` and must re-authorize once. Idle refresh keys that never expire are included in that cleanup.

## 2.21.0 — 2026-11-21

Knowledge-graph memory: expert verification, cross-source conflict detection, fail-closed as-of retrieve, validity changelogs, staged-ingest rollback, and Cypher hardening. Defaults are unchanged: raw Cypher is read-only, `CONFLICT_DETECT_ON_INGEST` and `CONFLICT_DETECT_LLM` are off, as-of lookups fail closed, and ingest staging is not required for a normal import.

Deferred: [#121](https://github.com/Lumen-Labs/brainapi2/pull/121) (agent-first MCP remember/recall/correct/check), [#104](https://github.com/Lumen-Labs/brainapi2/pull/104) (Celery Redis reconnect), [#105](https://github.com/Lumen-Labs/brainapi2/pull/105) (hosted ChatGPT MCP OAuth).

### Verification

- Facts carry `unverified` / `verified` / `rejected` with history. REST: `GET/POST /model/facts…`. MCP: `verify_fact`, `reject_fact`, `reset_fact_verification`, `list_facts_by_verification`, `get_fact_verification`. ([#113](https://github.com/Lumen-Labs/brainapi2/pull/113), [#125](https://github.com/Lumen-Labs/brainapi2/pull/125), [#126](https://github.com/Lumen-Labs/brainapi2/pull/126))
- Writes require the `verification` token scope (not granted by default). `method=verified_side` conflict resolve uses the same scope. ([#125](https://github.com/Lumen-Labs/brainapi2/pull/125), [#126](https://github.com/Lumen-Labs/brainapi2/pull/126))
- `verified_side` snapshots verification properties before writes. On reject failure it restores exact prior state (`verified_by`, `verified_at`, `verification_source`, reason) with a single history entry tagged `rollback`, and keeps the conflict open. Pre-read miss or error aborts before any write. ([#133](https://github.com/Lumen-Labs/brainapi2/pull/133), [#135](https://github.com/Lumen-Labs/brainapi2/pull/135))

### Conflicts

- Cross-source conflict scan, list, report, resolve, dismiss. REST under `/retrieve/conflicts`. MCP: `list_conflicts`, `get_conflict`, `resolve_conflict`, `dismiss_conflict`, `conflict_report`. ([#114](https://github.com/Lumen-Labs/brainapi2/pull/114), [#126](https://github.com/Lumen-Labs/brainapi2/pull/126))
- Ingest-time scan is incremental (touched subjects), pages past the old 10k cap, and evals scan-then-list with versioned timestamps. ([#128](https://github.com/Lumen-Labs/brainapi2/pull/128), [#132](https://github.com/Lumen-Labs/brainapi2/pull/132))
- Stale auto-resolve requires a concrete reason (side removed/superseded, validity windows no longer overlap, values compatible, or an explicit LLM `contradicts=false`). LLM-detected conflicts stay open when the LLM is off, the pair budget is exhausted, or that pair got no verdict. A stale-closed conflict reopens if the same contradiction is found again. Compact ISO clock times compare as instants. `upsert_conflict` does not overwrite a concurrent operator resolution. Ingest scans include prior published units in the same run and skip detect when a prior staged output cannot be read. ([#138](https://github.com/Lumen-Labs/brainapi2/pull/138), [#139](https://github.com/Lumen-Labs/brainapi2/pull/139))
- Full scans honor `CONFLICT_DETECT_MAX_RELATIONSHIPS` (default `10000`; `0` is unlimited). Capped scans page by whole `(subject_uuid, normalize_predicate(predicate))` groups, persist a per-brain keyset cursor `{after_subject, after_predicate}` in Mongo `conflict_scan_state` / Postgres `data_conflict_scan_state`, and wrap around so later scans continue past the previous window. Hitting the cap sets `truncated` and stale-closes only the groups that scan loaded. Neo4j `pred_key` matches `normalize_predicate` for relationship types that are not `clean_label` (hyphens, spaces, dots, non-ASCII), so those groups stay together; cursor pages keep `n.uuid >= after_subject_uuid`. ([#138](https://github.com/Lumen-Labs/brainapi2/pull/138), [#139](https://github.com/Lumen-Labs/brainapi2/pull/139), [#140](https://github.com/Lumen-Labs/brainapi2/pull/140))
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

### Security

- Raise `mcp` to `>=1.30.0,<2.0.0` and refresh both lockfiles (Trivy HIGH CVE-2026-115563). ([#140](https://github.com/Lumen-Labs/brainapi2/pull/140))

### Defaults and flags (unchanged unless you opt in)

| Flag / setting | Default |
| --- | --- |
| Raw Cypher | read-only |
| `GRAPH_RAW_WRITE_ENABLED` | `false` |
| `CONFLICT_DETECT_ON_INGEST` | `false` |
| `CONFLICT_DETECT_LLM` | `false` |
| `CONFLICT_DETECT_MAX_RELATIONSHIPS` | `10000` (`0` = unlimited) |
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
- A missing, null, or non-boolean LLM `contradicts` field is treated as `false` and can stale-close as `llm_compatible`.
- Compact timed stamps (`2024-01-01T090000`) collapse to date-only in `normalize_date_string`, so same-day events that differ only by time can share an identity UUID. Conflict detection uses `parse_date_string` and is not affected.
- An in-flight LLM verdict can stale-close a conflict after a later scan has already refreshed it with new fact values (`if_status=open` only).
- LLM batch indexes are not bounded to the current batch; a negative index can overwrite an earlier batch's verdict.
- `Config.conflict_detect_max_relationships` uses `int(...)` with no fallback; a blank or non-numeric env value raises at import. Runtime `conflict_detect_max_relationships()` already falls back.
- Compact clocks with a space separator (`20240101 090000`) compare as dates, not instants. `_instant_key` also drops microseconds, so fractional-second instants compare equal.
- Compact `as_of` values like `2024-01-01T930` can parse as 09:03 instead of being rejected.
- A comparison group larger than `CONFLICT_DETECT_MAX_RELATIONSHIPS` is still loaded whole so groups are never split.
- Truncated stale-close lists all open conflicts, then filters to scanned groups in Python.
- Skip/limit-only `list_validity_relationships` backends fall back to their native first page when no cursor is set. Production Neo4j, NetworkX, and search-fallback scans already order by comparison-group key.
- Neo4j conflict-scan count queries still run on each page (full-brain total, not the current window).
- Unknown-side lookups during stale-close are unbounded.
- The `search_relationships` fallback loads the whole graph before sorting and applying the cap.

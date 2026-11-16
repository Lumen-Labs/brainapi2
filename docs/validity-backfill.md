# Validity backfill

Idempotent job that fills missing `valid_from` / `valid_to` on existing brains so as-of retrieve and since/until changelogs can use canonical ISO-8601 UTC stamps. Readers already accept legacy `inserted_at` / `superseded_at` / `valid_at` / `invalid_at` (including `DD/MM/YYYY`), so a brain keeps answering before this runs.

## What it writes

| Store | Missing field | Source |
| --- | --- | --- |
| text chunks | `metadata.valid_from` | `inserted_at` |
| text chunks | `metadata.valid_to` | `metadata.superseded_at` |
| relationships | `valid_from` | `valid_at` or `source_timestamp` |
| relationships | `valid_to` | `invalid_at` |

It does **not**:

- overwrite an existing `valid_from`, `valid_to`, or `superseded_by`
- close a live copy that has no `superseded_at` / `invalid_at`
- invent an end stamp from `deprecated` alone
- call an LLM

A second run on the same brain writes nothing.

## How to run

Dry-run first (counts only):

```bash
poetry run python scripts/backfill_validity.py --brain-id YOUR_BRAIN --dry-run
```

Then apply:

```bash
poetry run python scripts/backfill_validity.py --brain-id YOUR_BRAIN
```

Uses the process `DATA_DB` / `GRAPH_DB` and the same connection environment as the API (`POSTGRES_*` or `MONGO_CONNECTION_STRING`, `NEO4J_*` when `GRAPH_DB=neo4j`). Repeat the apply command; the JSON `chunks` and `relationships` counts must be `0`.

Community Neo4j only has the `neo4j` database — pass `--brain-id neo4j` there.

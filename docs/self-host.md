# Self-host BrainAPI

Run this engine in your own cloud with the shipped Compose files. Use **light** for a smaller Postgres/pgvector stack, or **heavy** for Neo4j + MongoDB + Milvus. Only nginx publishes host ports (`80` / `443`).

BrainAPI is licensed under the [Business Source License 1.1](../LICENSE). Evaluation and non-production self-hosting are free. Running these production profiles for a live product or service requires a commercial license ([hello@lumen-labs.ai](mailto:hello@lumen-labs.ai)). See the [LICENSE](../LICENSE) file.

This guide covers a versioned-rules test (source versions + webhook/poll source sync), a help-center audit (scheduled re-crawl), and a paid production pilot. Spreadsheet import stays off unless you set `SPREADSHEET_EXPERIMENT_BRAINS` per host.

Do not put secrets in git. Copy an env template outside the repository and replace every secret there.

## Light or heavy

| | Light | Heavy |
| --- | --- | --- |
| Compose | `deploy/docker-compose.light.yaml` | `deploy/docker-compose.heavy.yaml` |
| Env template | `deploy/env.light.example` | `deploy/env.heavy.example` |
| Data / graph / vectors | PostgreSQL + pgvector + NetworkX | MongoDB + Neo4j + Milvus |
| Suggested footprint | smaller VM | larger VM |

Both profiles run `brainapi` (API + Console), `brainapi-worker`, `brainapi-mcp`, and a single `brainapi-beat`. Help-center re-crawl is triggered from a worker timer thread; source-sync poll/catch-up is scheduled by beat. Search can be enabled on either profile: BM25 requires Postgres (`DATA_DB=postgresql`); heavy/Mongo can run dense-only search with `SEARCH_USE_BM25=false`. Startup rejects BM25 on Mongo, not `SEARCH_ENABLED` itself.

## Bring-up

```bash
sudo install -d -m 755 /etc/brainapi
sudo install -m 600 -o "$USER" deploy/env.light.example /etc/brainapi/light.env
# Set BRAINAPI_ENV_FILE=/etc/brainapi/light.env (absolute) inside that file.
# Set BRAINAPI_IMAGE to the image you will run (template default is the current release tag).

docker compose --env-file /etc/brainapi/light.env \
  -f deploy/docker-compose.light.yaml config -q
docker compose --env-file /etc/brainapi/light.env \
  -f deploy/docker-compose.light.yaml up -d --wait
```

The light compose file sets `name: brainapi`. Heavy sets `name: brainapi-heavy`. Pass `--project-name` only when you need a different name (CI). Do not mix profiles under the same project.

Heavy is the same with `deploy/env.heavy.example` and `deploy/docker-compose.heavy.yaml`. The heavy profile uses Neo4j Enterprise because each brain is a Neo4j database. Set `NEO4J_ACCEPT_LICENSE_AGREEMENT` to `yes` or `eval` only if you hold a Neo4j licence; leaving it unset refuses start.

To evaluate without chat-model keys, add the CI stub (embeddings only; extraction, Chat, and file ingest still need a chat LLM):

```bash
docker compose --env-file /etc/brainapi/light.env \
  -f deploy/docker-compose.light.yaml -f deploy/docker-compose.ci.yaml up -d --wait
```

`GET /system/brains-list` requires the system PAT. Brain keys are omitted unless you pass `?include_pats=true` (the embedded Console and MaaS adapters that need keys must send the system token and that flag).

When pairing with the MaaS dashboard, set `BRAINPAT_SCOPES="brainapi verification"` (the env templates already do) so the system PAT can verify/reject facts. `POST /system/brains` accepts optional `pat_scopes` (`brainapi` and/or `verification`); the dashboard sends both when creating brains. Existing brains with empty `pat_scopes` stay `brainapi`-only. Set the same `USAGE_CONTEXT_SECRET` on the dashboard, this engine, and any proxy that signs `X-Usage-Context`. If they differ, dashboard Verify calls 403, or writes are attributed to the system principal.

Required Compose interpolation names (no values here): `BRAINAPI_IMAGE`, `BRAINAPI_ENV_FILE`, `REDIS_PASSWORD`, plus profile backends (`POSTGRES_PASSWORD` on light; `NEO4J_PASSWORD`, `NEO4J_ACCEPT_LICENSE_AGREEMENT`, `MONGO_PASSWORD`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MILVUS_TOKEN`, `MILVUS_ROOT_PASSWORD` on heavy).

Required application names from the env templates include: `BRAINPAT_TOKEN`, `BRAINPAT_SCOPES`, `USAGE_CONTEXT_SECRET`, `CELERY_WORKER_CONCURRENCY`, `CELERY_BACKEND`, `REDIS_HOST`, `REDIS_PORT`, `PLUGIN_FAILURE_POLICY`. Light also needs `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USERNAME`, `POSTGRES_SYSTEM_DATABASE`, `DATA_DB`, `VECTOR_DB`, `GRAPH_DB`. Heavy also needs `NEO4J_HOST`, `NEO4J_PORT`, `NEO4J_USERNAME`, `MONGO_HOST`, `MONGO_PORT`, `MONGO_USERNAME`, `MONGO_SYSTEM_DATABASE`, `MILVUS_HOST`, `MILVUS_PORT`.

Model and embedding credentials stay in the operator env file. Do not commit them.

## Plugins

Image plugins (including `source_sync`) load from `PLUGINS_DIR` (`/app/plugins` in Compose). Custom installs use `PLUGINS_EXTRA_DIR` (`/app/custom-plugins`, the `brainapi-plugins` volume). `BRAINAPI_PLUGINS` installs named plugins into the extra directory at startup. Later extra plugins with the same name override the image copy.

## Source sync (versioned rules)

The `source_sync` plugin is baked into the image. Create a source with `POST /source-sync/sources` (`source_type=zendesk_guide`, `sync_mode=webhook` or `poll`).

Webhook mode: set `webhook_secret_env` to the name of the env var that holds the HMAC secret. Send `X-Source-Sync-Signature` and `X-Source-Sync-Timestamp` (or the Zendesk signature headers). Signatures are rejected when the timestamp is stale (`SOURCE_SYNC_WEBHOOK_MAX_AGE_SECONDS`).

Poll and webhook catch-up: set `base_url` and `api_token_env` (and `api_email` when the connector needs it). Beat runs `source_sync.recheck_due_sources` on `SOURCE_SYNC_RECHECK_SECONDS`. Catch-up poll runs for webhook sources that have API credentials so missed webhooks are filled in. Listing uses cursor pagination (`page[size]`, default 100 via `SOURCE_SYNC_LIST_PAGE_SIZE`) and stops after `SOURCE_SYNC_LIST_MAX_PAGES` (default 500). A malformed page still fails the poll. If the page cap is hit, creates and updates from the fetched pages still apply, deletes are skipped, and `listing_truncated` is stored on the source. A source whose live listing always exceeds that page cap never gets a complete live set, so deletes are never applied; raise the cap or split the source. Poll and catch-up emit deletes for documents that disappeared from a complete live list, unless that would retire more than `SOURCE_SYNC_DELETE_MAX_RATIO` of the previous set or more than `SOURCE_SYNC_DELETE_MAX_PAGES`. That skip is stored on the source (`pending_delete_count`, `pending_delete_first_seen`) and recorded once as a `retirement_blocked` dead letter. Approve those deletes with `POST /source-sync/sources/{id}/approve-deletes` (or the `source_sync_approve_deletes` MCP tool), sending the `pending_delete_count` and `pending_delete_hash` from `GET /source-sync/sources/{id}`. If the pending id set changes before the next poll, the approval is rejected and must be repeated. A source-sync upsert pointer is saved only after ingest extraction succeeds. A source-sync update that loses the per-document lock is retried, then recorded in the dead-letter store instead of being dropped.

Source records, document heads, seen ids, and dead letters are stored in the system database (Postgres or Mongo), not only Redis. Redis is used for short document locks.

Workers consume `source_sync` and `source_sync_dead` through the Compose `-Q` list.

## Help-center audit (re-crawl)

Configure a crawl spec with `PUT /ingest/crawl-spec` and optionally `POST /ingest/crawl-spec/run`. Requires crawler env names: `CRAWLER_PROVIDER`, `LUMEN_CRAWLER_KEY`, `LUMEN_CRAWLER_ENDPOINT`, `CRAWLER_POLL_TIMEOUT_SECONDS`.

The worker timer (`HELP_CENTER_CRAWL_TICK_SECONDS`) enqueues due specs. A spec already in flight is not started again, including forced manual runs.

Only HTTP 404/410 pages are retired. Previously seen pages that are not linked in this run are fetched again, up to `HELP_CENTER_RECHECK_MAX_PAGES` (default 50) per run. They retire only on a real 404/410. The first crawler quota or rate-limit error stops further re-checks for that run. Unchecked URLs stay in the last-seen set for the next run. Timeouts, blocks, 5xx, and crawl-limit truncation do not retire. A run that would retire more pages than `HELP_CENTER_RETIRE_MAX_RATIO` of the previous set, or more than `HELP_CENTER_RETIRE_MAX_PAGES`, skips retirement and keeps the previous last-seen set. Facts that still cite other live sources are not closed.

Disabling or deleting a spec during a run is kept when the run finishes.

## Spreadsheets

Leave `SPREADSHEET_EXPERIMENT_BRAINS` empty to keep spreadsheet import off. Set it to a comma-separated list of brain keys, or `*`, only when that host should enable it.

## Upgrade

1. Back up with `deploy/brainapi-backup backup` (see `deploy/README.md`).
2. Point `BRAINAPI_IMAGE` at the new image tag in the private env file.
3. `docker compose --env-file ... -f deploy/docker-compose.<profile>.yaml up -d --wait`
4. System tables/collections are created on startup (`CREATE TABLE IF NOT EXISTS` / upsert). Existing brains keep their data.
5. Verify `/health`, then a source-sync webhook or poll, then a crawl-spec run if you use re-crawl.

nginx re-resolves the API and MCP upstreams through Docker DNS (`valid=10s`), so recreating those containers no longer leaves 502s. Restarting nginx after an image change is not required.

Do not mix light and heavy under the same Compose project name. Do not mount a volume over `/app/plugins`; custom plugins belong in `PLUGINS_EXTRA_DIR`.

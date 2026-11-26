# Self-host BrainAPI

Run this engine in your own cloud with the shipped Compose files. Use **light** for a smaller Postgres/pgvector stack, or **heavy** for Neo4j + MongoDB + Milvus. Only nginx publishes host ports (`80` / `443`).

This guide covers a versioned-rules test (source versions + webhook/poll source sync), a help-center audit (scheduled re-crawl), and a paid production pilot. Spreadsheet import stays off unless you set `SPREADSHEET_EXPERIMENT_BRAINS` per host.

Do not put secrets in git. Copy an env template outside the repository and replace every secret there.

## Light or heavy

| | Light | Heavy |
| --- | --- | --- |
| Compose | `deploy/docker-compose.light.yaml` | `deploy/docker-compose.heavy.yaml` |
| Env template | `deploy/env.light.example` | `deploy/env.heavy.example` |
| Data / graph / vectors | PostgreSQL + pgvector + NetworkX | MongoDB + Neo4j + Milvus |
| Suggested footprint | smaller VM | larger VM |

Both profiles run `brainapi` (API + Console), `brainapi-worker`, `brainapi-mcp`, and a single `brainapi-beat`. Help-center re-crawl is triggered from a worker timer thread; source-sync poll/catch-up is scheduled by beat.

## Bring-up

```bash
install -m 600 deploy/env.light.example /etc/brainapi/light.env
# Set BRAINAPI_ENV_FILE to that absolute path inside the env file.
# Set BRAINAPI_IMAGE to the image you will run.

docker compose --env-file /etc/brainapi/light.env \
  -f deploy/docker-compose.light.yaml config -q
docker compose --env-file /etc/brainapi/light.env \
  -f deploy/docker-compose.light.yaml up -d --wait
```

Heavy is the same with `deploy/env.heavy.example` and `deploy/docker-compose.heavy.yaml`.

Required Compose interpolation names (no values here): `BRAINAPI_IMAGE`, `BRAINAPI_ENV_FILE`, `REDIS_PASSWORD`, plus profile backends (`POSTGRES_PASSWORD` on light; `NEO4J_PASSWORD`, `MONGO_PASSWORD`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MILVUS_TOKEN`, `MILVUS_ROOT_PASSWORD` on heavy).

Required application names from the env templates include: `BRAINPAT_TOKEN`, `CELERY_WORKER_CONCURRENCY`, `CELERY_BACKEND`, `REDIS_HOST`, `REDIS_PORT`, `PLUGIN_FAILURE_POLICY`. Light also needs `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USERNAME`, `POSTGRES_SYSTEM_DATABASE`, `DATA_DB`, `VECTOR_DB`, `GRAPH_DB`. Heavy also needs `NEO4J_HOST`, `NEO4J_PORT`, `NEO4J_USERNAME`, `MONGO_HOST`, `MONGO_PORT`, `MONGO_USERNAME`, `MONGO_SYSTEM_DATABASE`, `MILVUS_HOST`, `MILVUS_PORT`.

Model and embedding credentials stay in the operator env file. Do not commit them.

## Plugins

Image plugins (including `source_sync`) load from `PLUGINS_DIR` (`/app/plugins` in Compose). Custom installs use `PLUGINS_EXTRA_DIR` (`/app/custom-plugins`, the `brainapi-plugins` volume). `BRAINAPI_PLUGINS` installs named plugins into the extra directory at startup. Later extra plugins with the same name override the image copy.

## Source sync (versioned rules)

The `source_sync` plugin is baked into the image. Create a source with `POST /source-sync/sources` (`source_type=zendesk_guide`, `sync_mode=webhook` or `poll`).

Webhook mode: set `webhook_secret_env` to the name of the env var that holds the HMAC secret. Send `X-Source-Sync-Signature` and `X-Source-Sync-Timestamp` (or the Zendesk signature headers). Signatures are rejected when the timestamp is stale (`SOURCE_SYNC_WEBHOOK_MAX_AGE_SECONDS`).

Poll and webhook catch-up: set `base_url` and `api_token_env` (and `api_email` when the connector needs it). Beat runs `source_sync.recheck_due_sources` on `SOURCE_SYNC_RECHECK_SECONDS`. Catch-up poll runs for webhook sources that have API credentials so missed webhooks are filled in. Poll also emits deletes for documents that disappeared from the live list.

Source records, document heads, seen ids, and dead letters are stored in the system database (Postgres or Mongo), not only Redis. Redis is used for short document locks.

Workers consume `source_sync` and `source_sync_dead` through the Compose `-Q` list.

## Help-center audit (re-crawl)

Configure a crawl spec with `PUT /ingest/crawl-spec` and optionally `POST /ingest/crawl-spec/run`. Requires crawler env names: `CRAWLER_PROVIDER`, `LUMEN_CRAWLER_KEY`, `LUMEN_CRAWLER_ENDPOINT`, `CRAWLER_POLL_TIMEOUT_SECONDS`.

The worker timer (`HELP_CENTER_CRAWL_TICK_SECONDS`) enqueues due specs. A spec already in flight is not started again, including forced manual runs.

Only HTTP 404/410 pages are retired. Timeouts, blocks, 5xx, and crawl-limit truncation do not retire. A run that would retire more pages than `HELP_CENTER_RETIRE_MAX_RATIO` of the previous set, or more than `HELP_CENTER_RETIRE_MAX_PAGES`, skips retirement and keeps the previous last-seen set. Facts that still cite other live sources are not closed.

Disabling or deleting a spec during a run is kept when the run finishes.

## Spreadsheets

Leave `SPREADSHEET_EXPERIMENT_BRAINS` empty to keep spreadsheet import off. Set it to a comma-separated list of brain keys, or `*`, only when that host should enable it.

## Upgrade

1. Back up with `deploy/brainapi-backup backup` (see `deploy/README.md`).
2. Point `BRAINAPI_IMAGE` at the new image tag in the private env file.
3. `docker compose --env-file ... -f deploy/docker-compose.<profile>.yaml up -d --wait`
4. System tables/collections are created on startup (`CREATE TABLE IF NOT EXISTS` / upsert). Existing brains keep their data.
5. Verify `/health`, then a source-sync webhook or poll, then a crawl-spec run if you use re-crawl.

Do not mix light and heavy under the same Compose project name. Do not mount a volume over `/app/plugins`; custom plugins belong in `PLUGINS_EXTRA_DIR`.

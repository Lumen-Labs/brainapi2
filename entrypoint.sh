#!/bin/bash
set -e

plugin_failure_policy="${PLUGIN_FAILURE_POLICY:-}"
if [ -z "$plugin_failure_policy" ]; then
    if [ "${ENV:-production}" = "development" ]; then
        plugin_failure_policy="warn"
    else
        plugin_failure_policy="fail"
    fi
fi
if [ "$plugin_failure_policy" != "fail" ] && [ "$plugin_failure_policy" != "warn" ]; then
    echo "[brainapi] PLUGIN_FAILURE_POLICY must be 'fail' or 'warn'" >&2
    exit 64
fi

if [ -d /app/.cache ]; then
    chown -R appuser:appuser /app/.cache 2>/dev/null || true
fi

mkdir -p /app/plugins
chown -R appuser:appuser /app/plugins 2>/dev/null || true

if [ "${STAGED_INGESTION_ENABLED:-0}" = "1" ]; then
    if [ -z "${INGEST_PUBLICATION_LOCK_DIR:-}" ] || [ ! -d "$INGEST_PUBLICATION_LOCK_DIR" ]; then
        echo "[brainapi] Staged ingestion requires a shared publication lock directory" >&2
        exit 64
    fi
    if [ "$INGEST_PUBLICATION_LOCK_DIR" = "/app/ingest-publication-locks" ]; then
        chown appuser:appuser /app/ingest-publication-locks
    fi
    if ! setpriv --reuid=appuser --regid=appuser --init-groups -- test -w "$INGEST_PUBLICATION_LOCK_DIR"; then
        echo "[brainapi] The publication lock directory must be writable by appuser" >&2
        exit 64
    fi
fi

if [ -n "$BRAINAPI_PLUGINS" ]; then
    IFS=',' read -ra PLUGINS <<< "$BRAINAPI_PLUGINS"
    for plugin_spec in "${PLUGINS[@]}"; do
        plugin_spec="$(echo "$plugin_spec" | xargs)"
        name="${plugin_spec%%:*}"
        version="${plugin_spec##*:}"
        [ "$name" = "$version" ] && version="latest"
        echo "[brainapi] Installing plugin: $name v$version"
        if ! setpriv --reuid=appuser --regid=appuser --init-groups -- \
            /app/.venv/bin/python -m src.core.plugins.cli plugins install "$name" --version "$version"; then
            if [ "$plugin_failure_policy" = "fail" ]; then
                echo "[brainapi] Failed to install required plugin '$name'" >&2
                exit 1
            fi
            echo "[brainapi] WARNING: Failed to install plugin '$name'" >&2
        fi
    done
fi

exec setpriv --reuid=appuser --regid=appuser --init-groups -- \
    /app/.venv/bin/python "$@"

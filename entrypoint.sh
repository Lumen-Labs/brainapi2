#!/bin/bash
set -e

if [ -d /app/.cache ]; then
    chown -R appuser:appuser /app/.cache 2>/dev/null || true
fi

mkdir -p /app/plugins
chown -R appuser:appuser /app/plugins 2>/dev/null || true

if [ -n "$BRAINAPI_PLUGINS" ]; then
    IFS=',' read -ra PLUGINS <<< "$BRAINAPI_PLUGINS"
    for plugin_spec in "${PLUGINS[@]}"; do
        plugin_spec="$(echo "$plugin_spec" | xargs)"
        name="${plugin_spec%%:*}"
        version="${plugin_spec##*:}"
        [ "$name" = "$version" ] && version="latest"
        echo "[brainapi] Installing plugin: $name v$version"
        gosu appuser /app/.venv/bin/python -m src.core.plugins.cli install "$name" --version "$version" || echo "[brainapi] WARNING: Failed to install plugin '$name'"
    done
fi

exec gosu appuser /app/.venv/bin/python "$@"

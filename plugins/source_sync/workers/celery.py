from __future__ import annotations

import os

from kombu import Queue

QUEUES = (
    Queue("source_sync", routing_key="source_sync"),
    Queue("source_sync_dead", routing_key="source_sync_dead"),
)

ROUTES = {
    "source_sync.apply_event": {"queue": "source_sync"},
    "source_sync.poll_source": {"queue": "source_sync"},
    "source_sync.recheck_due_sources": {"queue": "source_sync"},
}

_interval = os.getenv("SOURCE_SYNC_RECHECK_SECONDS", "60").strip()
try:
    RECHECK_SECONDS = max(15, int(_interval))
except ValueError:
    RECHECK_SECONDS = 60

BEAT_SCHEDULE = {
    "source-sync-recheck": {
        "task": "source_sync.recheck_due_sources",
        "schedule": float(RECHECK_SECONDS),
    }
}

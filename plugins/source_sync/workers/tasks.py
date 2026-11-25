from __future__ import annotations

import sys
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
PLUGIN_DIR_STR = str(_PLUGIN_DIR)
if PLUGIN_DIR_STR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR_STR)

from celery.exceptions import Reject
from src.workers.app import ingestion_app
from source_sync_api import apply_payload_now, poll_source_now
from source_sync_apply import dead_letter
from source_sync_store import default_store


@ingestion_app.task(
    bind=True,
    name="source_sync.apply_event",
    acks_late=True,
    reject_on_worker_lost=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    max_retries=5,
)
def apply_event_task(self, args: dict | None = None):
    payload = args or {}
    try:
        return apply_payload_now(payload)
    except Exception as exc:
        retries = int(getattr(self.request, "retries", 0) or 0)
        maximum = int(getattr(self, "max_retries", 5) or 5)
        if retries >= maximum:
            _dead_letter(payload, exc)
            raise Reject(str(exc), requeue=False) from exc
        raise


@ingestion_app.task(
    bind=True,
    name="source_sync.poll_source",
    acks_late=True,
    reject_on_worker_lost=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    max_retries=5,
)
def poll_source_task(self, args: dict | None = None):
    payload = args or {}
    source_id = str(payload.get("source_id") or "")
    return poll_source_now(source_id)


@ingestion_app.task(
    bind=True,
    name="source_sync.recheck_due_sources",
    acks_late=True,
)
def recheck_due_sources_task(self, args: dict | None = None):
    store = default_store()
    queued = []
    for record in store.list_indexed():
        if not store.due_for_poll(record):
            continue
        result = ingestion_app.send_task(
            "source_sync.poll_source",
            args=[{"source_id": record.get("id")}],
            queue="source_sync",
        )
        queued.append(
            {
                "source_id": record.get("id"),
                "task_id": str(getattr(result, "id", "") or ""),
            }
        )
    return {"queued": queued, "count": len(queued)}


def _dead_letter(payload: dict, exc: Exception) -> None:
    store = default_store()
    source_id = str(payload.get("source_id") or "")
    source = store.get_any(source_id) or {
        "id": source_id,
        "brain_id": payload.get("brain_id") or "default",
    }
    dead_letter(source, payload, str(exc), store=store)

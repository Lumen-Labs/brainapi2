from __future__ import annotations

import logging
import os
from typing import Any, Callable
from uuid import uuid4

from src.core.saving.help_center_crawl import retirement_cap
from src.core.saving.source_citations import close_orphaned_citing_facts
from src.core.saving.source_passages import close_source_document
from src.core.saving.validity import format_instant, parse_instant, utc_now
from source_sync_connector import (
    DocumentListing,
    SourceEvent,
    SourceFetchError,
    get_connector,
)
from source_sync_store import SourceSyncStore, default_store

QueueIngest = Callable[[dict[str, Any]], str | None]
logger = logging.getLogger("source_sync")

DEFAULT_DELETE_MAX_RATIO = 0.2
DEFAULT_DELETE_MAX_PAGES = 50


class DocumentBusy(RuntimeError):
    pass


def source_key_for(source_type: str, document_id: str) -> str:
    return f"{source_type}:{document_id}"


def apply_event(
    event: SourceEvent,
    source: dict[str, Any],
    *,
    store: SourceSyncStore | None = None,
    data: Any | None = None,
    graph: Any | None = None,
    queue_ingest: QueueIngest | None = None,
) -> dict[str, Any]:
    store = store or default_store()
    brain_id = str(source.get("brain_id") or "default")
    source_id = str(source.get("id") or "")
    source_type = str(source.get("source_type") or "")
    action = event.action if event.action in {"upsert", "delete"} else "upsert"
    occurred_at = format_instant(parse_instant(event.occurred_at) or utc_now())
    if not store.acquire_document_lock(brain_id, source_id, event.document_id):
        raise DocumentBusy(f"document {event.document_id} is busy")
    try:
        head = store.document_head(brain_id, source_id, event.document_id)
        decision = _head_decision(head, event.revision, occurred_at, action)
        if decision in {"duplicate", "stale"}:
            return {
                "status": decision,
                "source_id": source_id,
                "document_id": event.document_id,
                "revision": event.revision,
                "action": action,
            }
        key_name = source_key_for(source_type, event.document_id)
        if action == "delete":
            closed_chunks = _close_document(
                data, graph, brain_id=brain_id, source_key=key_name, closed_at=occurred_at
            )
            store.save_document_head(
                brain_id,
                source_id,
                event.document_id,
                {
                    "revision": event.revision,
                    "action": action,
                    "occurred_at": occurred_at,
                    "source_key": key_name,
                },
            )
            store.mark_synced(brain_id, source_id, occurred_at)
            return {
                "status": "deleted",
                "source_id": source_id,
                "document_id": event.document_id,
                "revision": event.revision,
                "source_key": key_name,
                "closed_chunk_ids": closed_chunks,
                "valid_to": occurred_at,
            }
        body = event.body
        if not body:
            body = _fill_body(source, event)
        if not body:
            raise SourceFetchError("upsert requires document body")
        ingest_id = None
        if queue_ingest is not None:
            ingest_id = queue_ingest(
                _ingest_payload(source, event, key_name, body, occurred_at)
            )
            store.mark_synced(brain_id, source_id, occurred_at)
            return {
                "status": "queued",
                "source_id": source_id,
                "document_id": event.document_id,
                "revision": event.revision,
                "source_key": key_name,
                "task_id": ingest_id,
            }
        _write_document_head(
            store,
            brain_id=brain_id,
            source_id=source_id,
            event=event,
            occurred_at=occurred_at,
            source_key=key_name,
        )
        store.mark_synced(brain_id, source_id, occurred_at)
        return {
            "status": "applied",
            "source_id": source_id,
            "document_id": event.document_id,
            "revision": event.revision,
            "source_key": key_name,
            "task_id": ingest_id,
        }
    finally:
        store.release_document_lock(brain_id, source_id, event.document_id)


def apply_poll(
    source: dict[str, Any],
    *,
    store: SourceSyncStore | None = None,
    data: Any | None = None,
    graph: Any | None = None,
    queue_ingest: QueueIngest | None = None,
    enqueue_event: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    store = store or default_store()
    listing = get_connector(str(source.get("source_type") or "")).list_documents(source)
    return _complete_poll(
        source,
        listing,
        store=store,
        data=data,
        graph=graph,
        queue_ingest=queue_ingest,
        enqueue_event=enqueue_event,
    )


def commit_upsert_head(
    *,
    brain_id: str,
    source_id: str,
    document_id: str,
    revision: str,
    occurred_at: str,
    source_key: str,
    store: SourceSyncStore | None = None,
) -> dict[str, Any]:
    store = store or default_store()
    stamp = format_instant(parse_instant(occurred_at) or utc_now())
    if not store.acquire_document_lock(brain_id, source_id, document_id):
        raise DocumentBusy(f"document {document_id} is busy")
    try:
        head = store.document_head(brain_id, source_id, document_id)
        decision = _head_decision(head, revision, stamp, "upsert")
        if decision in {"duplicate", "stale"}:
            return {
                "status": decision,
                "source_id": source_id,
                "document_id": document_id,
                "revision": revision,
            }
        store.save_document_head(
            brain_id,
            source_id,
            document_id,
            {
                "revision": revision,
                "action": "upsert",
                "occurred_at": stamp,
                "source_key": source_key,
                "ingest_status": "committed",
            },
        )
        store.mark_synced(brain_id, source_id, stamp)
        return {
            "status": "committed",
            "source_id": source_id,
            "document_id": document_id,
            "revision": revision,
            "source_key": source_key,
        }
    finally:
        store.release_document_lock(brain_id, source_id, document_id)


def commit_upsert_head_from_payload(
    payload: Any,
    *,
    store: SourceSyncStore | None = None,
) -> dict[str, Any] | None:
    meta = dict(getattr(payload, "meta_keys", None) or {})
    source_id = str(meta.get("source_sync_id") or "")
    document_id = str(meta.get("source_document_id") or "")
    if not source_id or not document_id:
        return None
    brain_id = str(getattr(payload, "brain_id", None) or meta.get("brain_id") or "default")
    source_key = str(
        meta.get("source_key")
        or source_key_for(str(meta.get("source_type") or ""), document_id)
    )
    occurred_at = str(getattr(payload, "source_timestamp", None) or "")
    return commit_upsert_head(
        brain_id=brain_id,
        source_id=source_id,
        document_id=document_id,
        revision=str(meta.get("source_revision") or ""),
        occurred_at=occurred_at,
        source_key=source_key,
        store=store,
    )


def approve_pending_deletes(
    brain_id: str,
    source_id: str,
    *,
    store: SourceSyncStore | None = None,
) -> dict[str, Any]:
    store = store or default_store()
    record = store.get(brain_id, source_id)
    if record is None:
        raise ValueError("source not found")
    pending = int(record.get("pending_delete_count") or 0)
    if pending <= 0:
        raise ValueError("no pending deletions")
    updated = store.arm_delete_approval(brain_id, source_id)
    return updated or record


def dead_letter(
    source: dict[str, Any],
    event: dict[str, Any],
    error: str,
    *,
    store: SourceSyncStore | None = None,
) -> dict[str, Any]:
    store = store or default_store()
    brain_id = str(source.get("brain_id") or event.get("brain_id") or "default")
    return store.record_dead_letter(
        brain_id,
        {
            "source_id": source.get("id") or event.get("source_id"),
            "document_id": event.get("document_id"),
            "revision": event.get("revision"),
            "action": event.get("action"),
            "error": str(error),
        },
    )


def record_failed_event(
    payload: dict[str, Any],
    error: BaseException | str,
    *,
    retries: int,
    max_retries: int,
    store: SourceSyncStore | None = None,
) -> dict[str, Any] | None:
    if int(retries) < int(max_retries):
        return None
    store = store or default_store()
    source_id = str(payload.get("source_id") or "")
    source = store.get_any(source_id) or {
        "id": source_id,
        "brain_id": payload.get("brain_id") or "default",
    }
    return dead_letter(source, payload, str(error), store=store)


def poll_delete_max_ratio() -> float:
    raw = os.getenv("SOURCE_SYNC_DELETE_MAX_RATIO", str(DEFAULT_DELETE_MAX_RATIO))
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_DELETE_MAX_RATIO
    return parsed if parsed > 0 else DEFAULT_DELETE_MAX_RATIO


def poll_delete_max_pages() -> int:
    raw = os.getenv("SOURCE_SYNC_DELETE_MAX_PAGES", str(DEFAULT_DELETE_MAX_PAGES))
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_DELETE_MAX_PAGES
    return parsed if parsed > 0 else DEFAULT_DELETE_MAX_PAGES


def poll_delete_abort_reason(previous_count: int, delete_count: int) -> str | None:
    if delete_count <= 0:
        return None
    cap = retirement_cap(
        previous_count,
        ratio=poll_delete_max_ratio(),
        max_pages=poll_delete_max_pages(),
    )
    if delete_count > cap:
        return f"poll delete count {delete_count} exceeds retirement cap {cap}"
    return None


def event_task_payload(source: dict[str, Any], event: SourceEvent) -> dict[str, Any]:
    return {
        "source_id": source.get("id"),
        "brain_id": source.get("brain_id"),
        "action": event.action,
        "document_id": event.document_id,
        "revision": event.revision,
        "title": event.title,
        "body": event.body,
        "url": event.url,
        "occurred_at": event.occurred_at,
        "metadata": event.metadata,
    }


def event_from_payload(payload: dict[str, Any]) -> SourceEvent:
    return SourceEvent(
        action=str(payload.get("action") or "upsert"),
        document_id=str(payload.get("document_id") or ""),
        revision=str(payload.get("revision") or ""),
        title=str(payload.get("title") or ""),
        body=str(payload.get("body") or ""),
        url=str(payload.get("url") or ""),
        occurred_at=payload.get("occurred_at"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _complete_poll(
    source: dict[str, Any],
    listing: DocumentListing,
    *,
    store: SourceSyncStore,
    data: Any | None,
    graph: Any | None,
    queue_ingest: QueueIngest | None,
    enqueue_event: Callable[[dict[str, Any]], Any] | None,
) -> dict[str, Any]:
    events = listing.events
    brain_id = str(source.get("brain_id") or "default")
    source_id = str(source.get("id") or "")
    live_ids, missing, abort = _poll_gap(
        store, brain_id=brain_id, source_id=source_id, events=events
    )
    skip_deletes = listing.truncated or bool(abort)
    approved = False
    if abort and not listing.truncated:
        approved = store.consume_delete_approval(brain_id, source_id)
        if approved:
            skip_deletes = False
    delete_abort = None
    if skip_deletes:
        delete_abort = abort or "listing truncated; deletes skipped"
    _record_poll_safety(
        store,
        source,
        abort=abort,
        truncated=listing.truncated,
        missing=missing,
        pages_fetched=listing.pages_fetched,
        skip_deletes=skip_deletes,
        approved=approved,
    )
    deletes = _maybe_emit_missing_deletes(
        source,
        missing,
        delete_abort,
        store=store,
        data=data,
        graph=graph,
        queue_ingest=queue_ingest,
        enqueue_event=enqueue_event,
    )
    applied = _apply_listed_events(
        source,
        events,
        store=store,
        data=data,
        graph=graph,
        queue_ingest=queue_ingest,
        enqueue_event=enqueue_event,
    )
    _commit_poll_seen(
        store,
        brain_id=brain_id,
        source_id=source_id,
        live_ids=live_ids,
        missing=missing,
        abort=delete_abort,
    )
    return _poll_result(
        applied,
        deletes,
        delete_abort,
        truncated=listing.truncated,
        pages_fetched=listing.pages_fetched,
    )


def _poll_gap(
    store: SourceSyncStore,
    *,
    brain_id: str,
    source_id: str,
    events: list[SourceEvent],
) -> tuple[set[str], list[str], str | None]:
    live_ids = {event.document_id for event in events if event.document_id}
    previous_ids = store.list_open_document_ids(brain_id, source_id)
    missing = sorted(previous_ids - live_ids)
    return live_ids, missing, poll_delete_abort_reason(len(previous_ids), len(missing))


def _commit_poll_seen(
    store: SourceSyncStore,
    *,
    brain_id: str,
    source_id: str,
    live_ids: set[str],
    missing: list[str],
    abort: str | None,
) -> None:
    retained = set(live_ids)
    if abort:
        retained.update(missing)
    store.set_seen_ids(brain_id, source_id, retained)


def _poll_result(
    applied: dict[str, Any],
    deletes: list[str],
    abort: str | None,
    *,
    truncated: bool = False,
    pages_fetched: int = 0,
) -> dict[str, Any]:
    payload = {
        "status": applied["status"],
        "count": applied["count"],
        "deleted": len(deletes),
        "deleted_ids": deletes,
        "retirement_aborted": abort,
        "listing_truncated": truncated,
        "listing_pages_fetched": pages_fetched,
    }
    if "results" in applied:
        payload["results"] = applied["results"]
    return payload


def _record_poll_safety(
    store: SourceSyncStore,
    source: dict[str, Any],
    *,
    abort: str | None,
    truncated: bool,
    missing: list[str],
    pages_fetched: int,
    skip_deletes: bool,
    approved: bool,
) -> None:
    brain_id = str(source.get("brain_id") or "default")
    source_id = str(source.get("id") or "")
    store.set_listing_status(
        brain_id, source_id, truncated=truncated, pages_fetched=pages_fetched
    )
    if truncated:
        return
    if abort and skip_deletes:
        newly = store.record_blocked_deletes(
            brain_id, source_id, len(missing), abort
        )
        if newly:
            _blocked_delete_dead_letter(store, source, missing, abort)
        return
    store.clear_blocked_deletes(brain_id, source_id)
    if not approved:
        store.consume_delete_approval(brain_id, source_id)


def _blocked_delete_dead_letter(
    store: SourceSyncStore,
    source: dict[str, Any],
    missing: list[str],
    abort: str,
) -> None:
    store.record_dead_letter(
        str(source.get("brain_id") or "default"),
        {
            "kind": "retirement_blocked",
            "source_id": source.get("id"),
            "error": abort,
            "pending_delete_count": len(missing),
        },
    )


def _write_document_head(
    store: SourceSyncStore,
    *,
    brain_id: str,
    source_id: str,
    event: SourceEvent,
    occurred_at: str,
    source_key: str,
) -> None:
    store.save_document_head(
        brain_id,
        source_id,
        event.document_id,
        {
            "revision": event.revision,
            "action": event.action if event.action in {"upsert", "delete"} else "upsert",
            "occurred_at": occurred_at,
            "source_key": source_key,
            "ingest_status": "committed",
        },
    )


def _maybe_emit_missing_deletes(
    source: dict[str, Any],
    missing: list[str],
    abort: str | None,
    *,
    store: SourceSyncStore,
    data: Any | None,
    graph: Any | None,
    queue_ingest: QueueIngest | None,
    enqueue_event: Callable[[dict[str, Any]], Any] | None,
) -> list[str]:
    if abort:
        logger.warning(
            "source_sync poll skipped deletes source_id=%s count=%s reason=%s",
            source.get("id"),
            len(missing),
            abort,
        )
        return []
    return _emit_missing_deletes(
        source,
        missing,
        store=store,
        data=data,
        graph=graph,
        queue_ingest=queue_ingest,
        enqueue_event=enqueue_event,
    )


def _emit_missing_deletes(
    source: dict[str, Any],
    missing: list[str],
    *,
    store: SourceSyncStore,
    data: Any | None,
    graph: Any | None,
    queue_ingest: QueueIngest | None,
    enqueue_event: Callable[[dict[str, Any]], Any] | None,
) -> list[str]:
    deletes = []
    closed_at = format_instant(utc_now())
    for document_id in missing:
        delete_event = SourceEvent(
            action="delete",
            document_id=document_id,
            revision=closed_at,
            occurred_at=closed_at,
        )
        deletes.append(document_id)
        if enqueue_event is not None:
            enqueue_event(event_task_payload(source, delete_event))
            continue
        apply_event(
            delete_event,
            source,
            store=store,
            data=data,
            graph=graph,
            queue_ingest=queue_ingest,
        )
    return deletes


def _apply_listed_events(
    source: dict[str, Any],
    events: list[SourceEvent],
    *,
    store: SourceSyncStore,
    data: Any | None,
    graph: Any | None,
    queue_ingest: QueueIngest | None,
    enqueue_event: Callable[[dict[str, Any]], Any] | None,
) -> dict[str, Any]:
    if enqueue_event is not None:
        for event in events:
            enqueue_event(event_task_payload(source, event))
        return {"status": "enqueued", "count": len(events)}
    results = [
        apply_event(
            event,
            source,
            store=store,
            data=data,
            graph=graph,
            queue_ingest=queue_ingest,
        )
        for event in events
    ]
    return {"status": "applied", "count": len(results), "results": results}


def close_citing_facts(
    graph: Any,
    brain_id: str,
    chunk_ids: list[str],
    closed_at: str,
    *,
    data: Any | None = None,
) -> list[str]:
    return close_orphaned_citing_facts(
        graph,
        brain_id,
        chunk_ids,
        closed_at,
        data=data,
        reason="source document deleted",
    )


def _head_decision(
    head: dict[str, Any] | None,
    revision: str,
    occurred_at: str,
    action: str,
) -> str:
    if not head:
        return "apply"
    current_revision = str(head.get("revision") or "")
    current_action = str(head.get("action") or "upsert")
    if current_revision and current_revision == str(revision or "") and current_action == action:
        return "duplicate"
    incoming = _event_rank(occurred_at, revision, action)
    current = _event_rank(
        str(head.get("occurred_at") or ""),
        current_revision,
        current_action,
    )
    if incoming == current:
        return "duplicate"
    if incoming < current:
        return "stale"
    return "apply"


def _event_rank(occurred_at: str, revision: str, action: str) -> tuple:
    parsed = parse_instant(occurred_at)
    timestamp = parsed.timestamp() if parsed is not None else 0.0
    action_boost = 1 if action == "delete" else 0
    return (timestamp, str(revision or ""), action_boost)


def _fill_body(source: dict[str, Any], event: SourceEvent) -> str:
    if event.body:
        return event.body
    connector = get_connector(str(source.get("source_type") or ""))
    fetched = connector.fetch_document(source, event.document_id)
    if fetched is None:
        return ""
    return fetched.body


def _ingest_payload(
    source: dict[str, Any],
    event: SourceEvent,
    source_key: str,
    body: str,
    occurred_at: str,
) -> dict[str, Any]:
    text = event.title.strip()
    content = body.strip()
    if text and content:
        combined = f"{text}\n\n{content}"
    else:
        combined = content or text
    return {
        "data": {"data_type": "text", "text_data": combined},
        "brain_id": source.get("brain_id") or "default",
        "meta_keys": {
            "source_key": source_key,
            "source_type": source.get("source_type"),
            "source_document_id": event.document_id,
            "source_revision": event.revision,
            "source_url": event.url or None,
            "source_sync_id": source.get("id"),
        },
        "source_timestamp": occurred_at,
        "skip_enrichment": bool(source.get("skip_enrichment")),
        "kind": "text",
    }


def _close_document(
    data: Any | None,
    graph: Any | None,
    *,
    brain_id: str,
    source_key: str,
    closed_at: str,
) -> list[str]:
    adapter = data if data is not None else _data_adapter()
    closed = close_source_document(
        adapter,
        brain_id=brain_id,
        source_key=source_key,
        closed_at=closed_at,
    )
    graph_adapter = graph if graph is not None else _graph_adapter()
    if graph_adapter is not None and closed:
        close_citing_facts(graph_adapter, brain_id, closed, closed_at, data=adapter)
    return closed


def _data_adapter() -> Any:
    from src.core.instances import data_adapter

    return data_adapter


def _graph_adapter() -> Any | None:
    try:
        from src.core.instances import graph_adapter

        return graph_adapter
    except Exception:
        return None


def queue_ingest_task(payload: dict[str, Any]) -> str:
    from src.workers.tasks.ingestion import ingest_data, set_ingestion_task_status

    task_id = uuid4().hex
    brain_id = str(payload.get("brain_id") or "default")
    set_ingestion_task_status(task_id, brain_id, "queued", stage="queued")
    ingest_data.apply_async(args=[payload], task_id=task_id)
    return task_id

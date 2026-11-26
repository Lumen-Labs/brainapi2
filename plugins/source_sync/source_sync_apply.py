from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from src.core.saving.source_citations import close_orphaned_citing_facts
from src.core.saving.source_passages import close_source_document
from src.core.saving.validity import format_instant, parse_instant, utc_now
from source_sync_connector import SourceEvent, SourceFetchError, get_connector
from source_sync_store import SourceSyncStore, default_store

QueueIngest = Callable[[dict[str, Any]], str | None]


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
            "status": "queued" if ingest_id else "applied",
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
    connector = get_connector(str(source.get("source_type") or ""))
    events = connector.list_documents(source)
    brain_id = str(source.get("brain_id") or "default")
    source_id = str(source.get("id") or "")
    live_ids = {event.document_id for event in events if event.document_id}
    previous_ids = store.list_open_document_ids(brain_id, source_id)
    missing = sorted(previous_ids - live_ids)
    queued = []
    deletes = []
    for document_id in missing:
        delete_event = SourceEvent(
            action="delete",
            document_id=document_id,
            revision=format_instant(utc_now()),
            occurred_at=format_instant(utc_now()),
        )
        payload = event_task_payload(source, delete_event)
        deletes.append(document_id)
        if enqueue_event is not None:
            enqueue_event(payload)
        else:
            apply_event(
                delete_event,
                source,
                store=store,
                data=data,
                graph=graph,
                queue_ingest=queue_ingest,
            )
    results = []
    if enqueue_event is not None:
        for event in events:
            enqueue_event(event_task_payload(source, event))
            queued.append(event.document_id)
        store.set_seen_ids(brain_id, source_id, live_ids)
        return {
            "status": "enqueued",
            "count": len(queued),
            "deleted": len(deletes),
            "deleted_ids": deletes,
        }
    for event in events:
        results.append(
            apply_event(
                event,
                source,
                store=store,
                data=data,
                graph=graph,
                queue_ingest=queue_ingest,
            )
        )
    store.set_seen_ids(brain_id, source_id, live_ids)
    return {
        "status": "applied",
        "count": len(results),
        "results": results,
        "deleted": len(deletes),
        "deleted_ids": deletes,
    }


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

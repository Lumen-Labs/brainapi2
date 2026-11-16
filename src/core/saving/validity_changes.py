"""Unified added / superseded / invalidated changes for a time window."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.core.saving.validity import (
    SUPERSEDED_BY,
    VALID_FROM,
    VALID_TO,
    change_kind,
    event_timestamp,
    format_instant,
    parse_instant,
)

_CHUNK_PAGE = 100
_REL_PAGE = 100


def _jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    iso = getattr(value, "iso_format", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            pass
    if isinstance(value, datetime):
        return format_instant(value)
    if isinstance(value, dict):
        return {str(key): _jsonish(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            return _jsonish(dump(mode="python"))
    return str(value)


def _dump(row: Any) -> Any:
    return _jsonish(row)


def _chunk_change(chunk: Any, kind: str) -> dict[str, Any]:
    meta = getattr(chunk, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    return {
        "kind": kind,
        "target": "chunk",
        "id": str(getattr(chunk, "id", "") or ""),
        "timestamp": event_timestamp(
            meta, kind, inserted_at=getattr(chunk, "inserted_at", None)
        ),
        "valid_from": meta.get(VALID_FROM),
        "valid_to": meta.get(VALID_TO) or meta.get("superseded_at"),
        "superseded_by": meta.get(SUPERSEDED_BY),
        "source": {
            "source_key": meta.get("source_key"),
            "chunk_id": str(getattr(chunk, "id", "") or ""),
        },
        "text": getattr(chunk, "text", None),
    }


def _relationship_change(triple: Any, kind: str) -> dict[str, Any]:
    subject, predicate, obj = triple
    props = getattr(predicate, "properties", None) or {}
    if not isinstance(props, dict):
        props = {}
    source_chunk_ids = props.get("source_chunk_ids") or props.get("source_chunk_id")
    if source_chunk_ids is not None and not isinstance(source_chunk_ids, list):
        source_chunk_ids = [source_chunk_ids]
    return {
        "kind": kind,
        "target": "relationship",
        "id": str(getattr(predicate, "uuid", "") or props.get("uuid") or ""),
        "timestamp": event_timestamp(props, kind),
        "valid_from": props.get(VALID_FROM) or props.get("valid_at"),
        "valid_to": props.get(VALID_TO) or props.get("invalid_at"),
        "superseded_by": props.get(SUPERSEDED_BY),
        "source": {
            "source_chunk_ids": source_chunk_ids or [],
            "source_timestamp": props.get("source_timestamp"),
        },
        "subject": _dump(subject),
        "predicate": _dump(predicate),
        "object": _dump(obj),
    }


def _changelog_change(row: Any) -> dict[str, Any]:
    payload = _dump(row)
    stamp = parse_instant(getattr(row, "timestamp", None))
    return {
        "kind": "added"
        if str(getattr(row, "type", "") or "").endswith("created")
        else "invalidated"
        if "deprecated" in str(getattr(row, "type", "") or "")
        else "superseded",
        "target": "changelog",
        "id": str(getattr(row, "id", "") or ""),
        "timestamp": format_instant(stamp) if stamp else event_timestamp({}, "added"),
        "changelog": payload,
    }


def _iter_chunks(adapter: Any, brain_id: str):
    skip = 0
    while True:
        chunks, total = adapter.get_text_chunks(
            brain_id=brain_id, limit=_CHUNK_PAGE, skip=skip, order="desc"
        )
        if not chunks:
            break
        for chunk in chunks:
            yield chunk
        skip += len(chunks)
        if total is not None and skip >= int(total):
            break
        if len(chunks) < _CHUNK_PAGE:
            break


def _iter_relationships(
    graph: Any,
    brain_id: str,
    since: datetime | None = None,
    until: datetime | None = None,
):
    method = getattr(graph, "list_validity_relationships", None)
    if callable(method):
        skip = 0
        while True:
            triples, total = method(
                brain_id,
                skip=skip,
                limit=_REL_PAGE,
                since=format_instant(since) if since else None,
                until=format_instant(until) if until else None,
            )
            if not triples:
                break
            for triple in triples:
                yield triple
            skip += len(triples)
            if total is not None and skip >= int(total):
                break
            if len(triples) < _REL_PAGE:
                break
        return
    skip = 0
    while True:
        result = graph.search_relationships(brain_id, limit=_REL_PAGE, skip=skip)
        triples = getattr(result, "results", None) or result or []
        if not triples:
            break
        for triple in triples:
            yield (triple.subject, triple.predicate, triple.object)
        skip += len(triples)
        total = getattr(result, "total", None)
        if total is not None and skip >= int(total):
            break
        if len(triples) < _REL_PAGE:
            break


def _iter_changelogs(
    adapter: Any,
    brain_id: str,
    types: list[str] | None,
    query_text: str | None,
    since: datetime | None,
    until: datetime | None,
):
    skip = 0
    while True:
        try:
            rows = adapter.get_changelogs_list(
                brain_id,
                _CHUNK_PAGE,
                skip,
                types,
                query_text,
                since,
                until,
            )
        except TypeError:
            rows = adapter.get_changelogs_list(
                brain_id, _CHUNK_PAGE, skip, types, query_text
            )
        if not rows:
            break
        for row in rows:
            yield row
        skip += len(rows)
        if len(rows) < _CHUNK_PAGE:
            break


def collect_validity_changes(
    adapter: Any,
    graph: Any | None,
    brain_id: str,
    *,
    since: datetime,
    until: datetime,
    limit: int = 10,
    skip: int = 0,
    query_text: str | None = None,
    types: list[str] | None = None,
) -> dict[str, Any]:
    changes: list[dict[str, Any]] = []
    for chunk in _iter_chunks(adapter, brain_id):
        meta = getattr(chunk, "metadata", None) or {}
        kind = change_kind(
            meta, since, until, inserted_at=getattr(chunk, "inserted_at", None)
        )
        if kind:
            changes.append(_chunk_change(chunk, kind))
    if graph is not None:
        for triple in _iter_relationships(graph, brain_id, since, until):
            predicate = triple[1]
            props = getattr(predicate, "properties", None) or {}
            kind = change_kind(props, since, until)
            if kind:
                changes.append(_relationship_change(triple, kind))
    for row in _iter_changelogs(adapter, brain_id, types, query_text, since, until):
        changes.append(_changelog_change(row))
    changes.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
    total = len(changes)
    page = changes[skip : skip + limit]
    return {
        "changes": page,
        "count": len(page),
        "total": total,
        "since": format_instant(since),
        "until": format_instant(until),
    }

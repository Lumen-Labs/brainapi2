"""Idempotent validity backfill from inserted_at / superseded_at / invalid_at.

Safe on a live brain: only fills *missing* ``valid_from`` / ``valid_to``.
Existing stamps are left untouched (including legacy date formats). Live
copies without ``superseded_at`` / ``invalid_at`` are never closed. Run
twice; the second pass writes nothing. See ``docs/validity-backfill.md``.
"""

from __future__ import annotations

from typing import Any, Iterator

from src.core.saving.validity import (
    VALID_FROM,
    VALID_TO,
    format_instant,
    parse_instant,
    stamp_valid_from,
)

_PAGE = 100
_MAX = 10000


def _iter_chunks(adapter: Any, brain_id: str) -> Iterator[Any]:
    skip = 0
    while skip < _MAX:
        chunks, total = adapter.get_text_chunks(
            brain_id=brain_id, limit=_PAGE, skip=skip, order="asc"
        )
        if not chunks:
            break
        for chunk in chunks:
            yield chunk
        skip += len(chunks)
        if total is not None and skip >= int(total):
            break


def _relationship_triples(graph: Any, brain_id: str, skip: int, limit: int):
    method = getattr(graph, "list_validity_relationships", None)
    if callable(method):
        triples, total = method(brain_id, skip=skip, limit=limit)
        return list(triples or []), total
    result = graph.search_relationships(brain_id, limit=limit, skip=skip)
    rows = getattr(result, "results", None) or []
    triples = []
    for row in rows:
        if isinstance(row, (tuple, list)) and len(row) >= 3:
            triples.append((row[0], row[1], row[2]))
        else:
            triples.append((row.subject, row.predicate, row.object))
    return triples, getattr(result, "total", None)


def _chunk_patch(chunk: Any) -> dict[str, Any] | None:
    meta = dict(getattr(chunk, "metadata", None) or {})
    before = dict(meta)
    if not meta.get(VALID_FROM):
        meta = stamp_valid_from(
            meta,
            format_instant(parse_instant(getattr(chunk, "inserted_at", None))),
        )
    if meta.get("superseded_at") and not meta.get(VALID_TO):
        parsed = parse_instant(meta.get("superseded_at"))
        if parsed is not None:
            meta[VALID_TO] = format_instant(parsed)
    if meta == before:
        return None
    return meta


def _relationship_patch(predicate: Any) -> tuple[str | None, dict[str, Any]]:
    props = dict(getattr(predicate, "properties", None) or {})
    nested = props.get("properties")
    if isinstance(nested, dict):
        for key in ("valid_at", "invalid_at", "source_timestamp", VALID_FROM, VALID_TO):
            if not props.get(key) and nested.get(key):
                props[key] = nested[key]
    patch: dict[str, Any] = {}
    if not props.get(VALID_FROM):
        stamped = stamp_valid_from(
            props,
            props.get("valid_at") or props.get("source_timestamp"),
        )
        if stamped.get(VALID_FROM):
            patch[VALID_FROM] = stamped[VALID_FROM]
    if props.get("invalid_at") and not props.get(VALID_TO):
        parsed = parse_instant(props.get("invalid_at"))
        if parsed is not None:
            patch[VALID_TO] = format_instant(parsed)
    uuid = getattr(predicate, "uuid", None) or props.get("uuid")
    return uuid, patch


def backfill_chunk_validity(
    adapter: Any, brain_id: str, *, dry_run: bool = False
) -> int:
    updated = 0
    for chunk in _iter_chunks(adapter, brain_id):
        meta = _chunk_patch(chunk)
        if meta is None:
            continue
        if not dry_run:
            chunk.metadata = meta
            adapter.save_text_chunk(chunk, brain_id)
        updated += 1
    return updated


def backfill_relationship_validity(
    graph: Any, brain_id: str, *, dry_run: bool = False
) -> int:
    updated = 0
    skip = 0
    while skip < _MAX:
        triples, total = _relationship_triples(graph, brain_id, skip, _PAGE)
        if not triples:
            break
        for triple in triples:
            predicate = triple[1] if isinstance(triple, (tuple, list)) else triple.predicate
            uuid, patch = _relationship_patch(predicate)
            if not patch or not uuid:
                continue
            if not dry_run:
                graph.update_properties(uuid, "relationship", brain_id, patch, [])
                props = getattr(predicate, "properties", None)
                if isinstance(props, dict):
                    props.update(patch)
            updated += 1
        skip += len(triples)
        if total is not None and skip >= int(total):
            break
    return updated


def backfill_brain_validity(
    adapter: Any,
    graph: Any | None,
    brain_id: str,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    chunks = backfill_chunk_validity(adapter, brain_id, dry_run=dry_run)
    relationships = (
        backfill_relationship_validity(graph, brain_id, dry_run=dry_run)
        if graph is not None
        else 0
    )
    return {"chunks": chunks, "relationships": relationships}

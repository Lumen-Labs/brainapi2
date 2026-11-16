"""Idempotent validity backfill from inserted_at / superseded_at / invalid_at."""

from __future__ import annotations

from typing import Any

from src.core.saving.validity import (
    SUPERSEDED_BY,
    VALID_FROM,
    VALID_TO,
    format_instant,
    parse_instant,
    stamp_valid_from,
    utc_now,
)

_PAGE = 100
_MAX = 10000


def backfill_chunk_validity(adapter: Any, brain_id: str) -> int:
    updated = 0
    skip = 0
    while skip < _MAX:
        chunks, total = adapter.get_text_chunks(
            brain_id=brain_id, limit=_PAGE, skip=skip, order="asc"
        )
        if not chunks:
            break
        for chunk in chunks:
            meta = dict(getattr(chunk, "metadata", None) or {})
            before = dict(meta)
            if not meta.get(VALID_FROM):
                meta = stamp_valid_from(
                    meta,
                    format_instant(parse_instant(getattr(chunk, "inserted_at", None))),
                )
            if meta.get("superseded_at") and not meta.get(VALID_TO):
                meta[VALID_TO] = format_instant(
                    parse_instant(meta.get("superseded_at")) or utc_now()
                )
            if meta == before:
                continue
            chunk.metadata = meta
            adapter.save_text_chunk(chunk, brain_id)
            updated += 1
        skip += len(chunks)
        if total is not None and skip >= int(total):
            break
    return updated


def backfill_relationship_validity(graph: Any, brain_id: str) -> int:
    updated = 0
    skip = 0
    while skip < _MAX:
        result = graph.search_relationships(brain_id, limit=_PAGE, skip=skip)
        triples = getattr(result, "results", None) or []
        if not triples:
            break
        for triple in triples:
            predicate = triple.predicate
            props = dict(getattr(predicate, "properties", None) or {})
            patch: dict[str, Any] = {}
            if not props.get(VALID_FROM):
                stamped = stamp_valid_from(
                    props,
                    props.get("valid_at") or props.get("source_timestamp"),
                )
                if stamped.get(VALID_FROM):
                    patch[VALID_FROM] = stamped[VALID_FROM]
            end = props.get(VALID_TO) or props.get("invalid_at")
            if props.get("invalid_at") and not props.get(VALID_TO):
                parsed = parse_instant(props.get("invalid_at"))
                patch[VALID_TO] = format_instant(parsed or utc_now())
            if props.get("deprecated") is True and not (end or patch.get(VALID_TO)):
                patch[VALID_TO] = format_instant(utc_now())
            if not patch:
                continue
            uuid = getattr(predicate, "uuid", None) or props.get("uuid")
            if not uuid:
                continue
            graph.update_properties(uuid, "relationship", brain_id, patch, [])
            updated += 1
        skip += len(triples)
        total = getattr(result, "total", None)
        if total is not None and skip >= int(total):
            break
    return updated


def backfill_brain_validity(adapter: Any, graph: Any | None, brain_id: str) -> dict[str, int]:
    chunks = backfill_chunk_validity(adapter, brain_id)
    relationships = (
        backfill_relationship_validity(graph, brain_id) if graph is not None else 0
    )
    return {"chunks": chunks, "relationships": relationships}

"""Opt-in source identity for passages.

Chunks without ``source_key`` are never read or rewritten by this module.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from typing import Any, Literal

from src.constants.data import TextChunk
from src.core.saving.validity import (
    VALID_FROM,
    chunk_is_valid_at,
    format_instant,
    parse_instant,
    stamp_valid_from,
    utc_now,
)

Revision = Literal["plain", "insert", "unchanged"]
_PAGE = 100
_LOOKUP_PAGE = 80
_MAX_CHUNKS = 500


def source_content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def source_key_from_meta(meta: dict | None) -> str | None:
    if not isinstance(meta, dict):
        return None
    value = meta.get("source_key")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def visible_passage_body(
    metadata: dict | None,
    text: str | None,
    *,
    include_history: bool,
    as_of: datetime.datetime | None = None,
) -> str | None:
    body = (text or "").strip()
    if not body:
        return None
    if as_of is not None:
        if not chunk_is_valid_at(
            type("Chunk", (), {"metadata": metadata, "inserted_at": None})(),
            as_of,
        ):
            return None
        return body
    superseded = None
    if isinstance(metadata, dict):
        superseded = metadata.get("superseded_at") or metadata.get("valid_to")
    if not superseded:
        return body
    if not include_history:
        return None
    if body.startswith("[superseded] "):
        return body
    return f"[superseded] {body}"


def _stamp(source_timestamp: str | None) -> str:
    if source_timestamp and str(source_timestamp).strip():
        return str(source_timestamp).strip()
    return format_instant(utc_now())


def _current_chunks(adapter: Any, brain_id: str, source_key: str) -> list[TextChunk]:
    found: list[TextChunk] = []
    skip = 0
    total = None
    while skip < _MAX_CHUNKS:
        chunks, total = adapter.get_text_chunks(
            brain_id=brain_id,
            limit=_PAGE,
            skip=skip,
            metadata_eq={"source_key": source_key},
        )
        if not chunks:
            break
        for chunk in chunks:
            meta = getattr(chunk, "metadata", None) or {}
            if not meta.get("superseded_at"):
                found.append(chunk)
        skip += len(chunks)
        if total is not None and skip >= int(total):
            break
    return found


def prepare_source_revision(
    adapter: Any,
    *,
    brain_id: str,
    source_text: str,
    meta_keys: dict | None,
    source_timestamp: str | None,
    out_superseded: list[str] | None = None,
) -> Revision:
    """Mark older current chunks for this key, or report that the hash is unchanged.

    Mutates ``meta_keys`` to record ``source_content_hash`` when a new copy will be stored.
    """
    source_key = source_key_from_meta(meta_keys)
    if source_key is None or not isinstance(meta_keys, dict):
        return "plain"
    digest = source_content_hash(source_text)
    try:
        current = _current_chunks(adapter, brain_id, source_key)
    except Exception as exc:
        print(f"[!] source revision lookup failed for {source_key}: {exc}")
        meta_keys["source_content_hash"] = digest
        return "insert"
    same_ids = {
        str(chunk.id)
        for chunk in current
        if (getattr(chunk, "metadata", None) or {}).get("source_content_hash") == digest
    }
    stale = [chunk for chunk in current if str(chunk.id) not in same_ids]
    same = [chunk for chunk in current if str(chunk.id) in same_ids]
    closed_at = _stamp(source_timestamp)
    if same:
        _record_superseded(
            out_superseded,
            _mark(adapter, brain_id, stale, closed_at),
        )
        return "unchanged"
    _record_superseded(
        out_superseded,
        _mark(adapter, brain_id, current, closed_at),
    )
    meta_keys["source_content_hash"] = digest
    stamped = stamp_valid_from(meta_keys, source_timestamp or closed_at)
    meta_keys[VALID_FROM] = stamped[VALID_FROM]
    return "insert"


def _record_superseded(out_superseded: list[str] | None, ids: list[str]) -> None:
    if out_superseded is None:
        return
    out_superseded.extend(ids)


def _mark(
    adapter: Any,
    brain_id: str,
    chunks: list[TextChunk],
    superseded_at: str,
    superseded_by: str | None = None,
) -> list[str]:
    ids = [str(chunk.id) for chunk in chunks if getattr(chunk, "id", None)]
    if not ids:
        return []
    valid_to = format_instant(parse_instant(superseded_at) or utc_now())
    adapter.mark_text_chunks_superseded(
        ids,
        superseded_at,
        brain_id,
        superseded_by=superseded_by,
        valid_to=valid_to,
    )
    for chunk in chunks:
        meta = getattr(chunk, "metadata", None)
        if not isinstance(meta, dict):
            continue
        meta["superseded_at"] = superseded_at
        meta["valid_to"] = valid_to
        if superseded_by:
            meta["superseded_by"] = superseded_by
    return ids


def _inserted_time(chunk: TextChunk) -> float:
    value = getattr(chunk, "inserted_at", None)
    if isinstance(value, datetime.datetime):
        try:
            return value.timestamp()
        except Exception:
            return 0.0
    return 0.0


def _latest_current_ids(adapter: Any, brain_id: str, source_key: str) -> set[str] | None:
    try:
        current = _current_chunks(adapter, brain_id, source_key)
    except Exception:
        return None
    if not current:
        return set()
    newest = max(_inserted_time(chunk) for chunk in current)
    return {str(chunk.id) for chunk in current if _inserted_time(chunk) == newest}


def _older_copy(adapter: Any, brain_id: str, chunk: TextChunk, cache: dict[str, set[str] | None]) -> bool:
    meta = getattr(chunk, "metadata", None) or {}
    if meta.get("superseded_at"):
        return True
    source_key = source_key_from_meta(meta)
    if not source_key:
        return False
    if source_key not in cache:
        cache[source_key] = _latest_current_ids(adapter, brain_id, source_key)
    latest = cache[source_key]
    if latest is None:
        return False
    return str(chunk.id) not in latest


def _chunks_from_lookup(result: Any) -> list[Any]:
    if isinstance(result, tuple):
        return list(result[0] or [])
    return list(result or [])


def _load_chunks_by_ids(adapter: Any, ids: list[str], brain_id: str) -> dict[str, Any]:
    unique = list(dict.fromkeys(ids))
    loaded: dict[str, Any] = {}
    for start in range(0, len(unique), _LOOKUP_PAGE):
        batch = unique[start : start + _LOOKUP_PAGE]
        chunks = _chunks_from_lookup(
            adapter.get_text_chunks_by_ids(batch, False, brain_id)
        )
        for chunk in chunks:
            loaded[str(chunk.id)] = chunk
    return loaded


def current_fact_candidates(
    adapter: Any,
    brain_id: str,
    candidates: list[dict],
    *,
    include_history: bool,
    as_of: datetime.datetime | None = None,
) -> list[dict]:
    """Drop facts whose source chunks are all older copies.

    Facts with no chunk id stay. A fact stays when any of its chunks is current
    or missing. The newest chunk for a source key is current until an older one
    is stamped superseded.
    """
    if (include_history and as_of is None) or not candidates:
        return candidates
    ids: list[str] = []
    for candidate in candidates:
        ids.extend(str(item) for item in (candidate.get("chunk_ids") or []) if item)
    if not ids:
        return candidates
    try:
        loaded = _load_chunks_by_ids(adapter, ids, brain_id)
    except Exception:
        return [] if as_of is not None else candidates
    cache: dict[str, set[str] | None] = {}
    kept: list[dict] = []
    for candidate in candidates:
        chunk_ids = [str(item) for item in (candidate.get("chunk_ids") or []) if item]
        if not chunk_ids:
            kept.append(candidate)
            continue
        if any(chunk_id not in loaded for chunk_id in chunk_ids):
            if as_of is None:
                kept.append(candidate)
            continue
        loaded_chunks = [loaded[chunk_id] for chunk_id in chunk_ids]
        if as_of is not None:
            if any(chunk_is_valid_at(chunk, as_of) for chunk in loaded_chunks):
                kept.append(candidate)
            continue
        if all(_older_copy(adapter, brain_id, chunk, cache) for chunk in loaded_chunks):
            continue
        kept.append(candidate)
    return kept


def screen_passage_hits(
    adapter: Any,
    brain_id: str,
    hits: list[tuple[str, float, str]],
    *,
    include_history: bool,
    as_of: datetime.datetime | None = None,
) -> list[tuple[str, float, str]]:
    ids = [str(chunk_id) for chunk_id, _, _ in hits if chunk_id]
    if not ids:
        return hits
    try:
        by_id = _load_chunks_by_ids(adapter, ids, brain_id)
    except Exception:
        return [] if as_of is not None else hits
    cache: dict[str, set[str] | None] = {}
    screened: list[tuple[str, float, str]] = []
    for chunk_id, score, body in hits:
        chunk = by_id.get(str(chunk_id))
        if chunk is None:
            if as_of is None:
                screened.append((chunk_id, score, body))
            continue
        if as_of is not None:
            if not chunk_is_valid_at(chunk, as_of):
                continue
            screened.append((chunk_id, score, body or getattr(chunk, "text", None) or ""))
            continue
        if not include_history and _older_copy(adapter, brain_id, chunk, cache):
            continue
        meta = getattr(chunk, "metadata", None) or {}
        if not meta.get("superseded_at") and not meta.get("valid_to"):
            screened.append((chunk_id, score, body))
            continue
        visible = visible_passage_body(
            getattr(chunk, "metadata", None),
            body or getattr(chunk, "text", None),
            include_history=include_history,
            as_of=as_of,
        )
        if visible is None:
            continue
        screened.append((chunk_id, score, visible))
    return screened


def prior_revision_texts(
    adapter: Any,
    brain_id: str,
    hits: list[tuple[str, float, str]],
    *,
    limit: int,
) -> list[str]:
    """Load superseded copies for the source keys already in these hits.

    Search does not have to rank the older copy. The key on a current hit is enough.
    """
    cap = max(0, limit)
    if cap == 0:
        return []
    ids = [str(chunk_id) for chunk_id, _, _ in hits if chunk_id]
    if not ids:
        return []
    try:
        chunks, _ = adapter.get_text_chunks_by_ids(ids, False, brain_id)
    except Exception:
        return []
    keys: list[str] = []
    for chunk in chunks:
        key = source_key_from_meta(getattr(chunk, "metadata", None))
        if key and key not in keys:
            keys.append(key)
    texts: list[str] = []
    seen: set[str] = set()
    for key in keys:
        try:
            stored, _ = adapter.get_text_chunks(
                brain_id=brain_id,
                limit=_MAX_CHUNKS,
                skip=0,
                metadata_eq={"source_key": key},
            )
        except Exception:
            continue
        older = [
            chunk
            for chunk in stored
            if (getattr(chunk, "metadata", None) or {}).get("superseded_at")
        ]
        older.sort(
            key=lambda chunk: str((getattr(chunk, "metadata", None) or {}).get("superseded_at")),
            reverse=True,
        )
        for chunk in older:
            body = visible_passage_body(
                getattr(chunk, "metadata", None),
                getattr(chunk, "text", None),
                include_history=True,
            )
            if not body or body in seen:
                continue
            seen.add(body)
            texts.append(body)
            if len(texts) >= cap:
                return texts
    return texts


def screened_passage_texts(
    adapter: Any,
    brain_id: str,
    passages: list[tuple[str, float, str]],
    *,
    include_history: bool,
    limit: int,
    as_of: datetime.datetime | None = None,
) -> list[str]:
    screened = screen_passage_hits(
        adapter,
        brain_id,
        passages,
        include_history=include_history,
        as_of=as_of,
    )
    return [text for _, _, text in screened if text][: max(0, limit)]


def current_record_texts(
    adapter: Any,
    brain_id: str,
    records: list,
    chunks: list[TextChunk],
    *,
    include_history: bool,
    as_of: datetime.datetime | None = None,
) -> list[str]:
    """JSON for structured records that are still a current passage.

    A record is left out when its text matches an older copy and no current
    copy. Records with other text stay. History keeps every record.
    """
    cache: dict[str, set[str] | None] = {}
    older: set[str] = set()
    current: set[str] = set()
    if as_of is not None:
        for chunk in chunks:
            body = (getattr(chunk, "text", None) or "").strip()
            if not body:
                continue
            if chunk_is_valid_at(chunk, as_of):
                current.add(body)
            else:
                older.add(body)
    elif not include_history:
        for chunk in chunks:
            body = (getattr(chunk, "text", None) or "").strip()
            if not body:
                continue
            if _older_copy(adapter, brain_id, chunk, cache):
                older.add(body)
            else:
                current.add(body)
    lines: list[str] = []
    for record in records:
        if as_of is not None:
            inserted = parse_instant(getattr(record, "inserted_at", None))
            if inserted is not None and inserted > as_of:
                continue
        data = getattr(record, "data", None)
        if not isinstance(data, dict) or not data:
            continue
        body = data.get("text")
        if (
            (as_of is not None or not include_history)
            and isinstance(body, str)
            and body.strip() in older
            and body.strip() not in current
        ):
            continue
        lines.append(json.dumps(data))
    return lines


def current_chunk_body(
    adapter: Any,
    brain_id: str,
    chunk: TextChunk,
    *,
    include_history: bool,
    cache: dict[str, set[str] | None],
    as_of: datetime.datetime | None = None,
) -> str | None:
    if as_of is not None:
        if not chunk_is_valid_at(chunk, as_of):
            return None
        return (getattr(chunk, "text", None) or "").strip() or None
    if not include_history and _older_copy(adapter, brain_id, chunk, cache):
        return None
    return visible_passage_body(
        getattr(chunk, "metadata", None),
        getattr(chunk, "text", None),
        include_history=include_history,
        as_of=as_of,
    )

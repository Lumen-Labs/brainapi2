"""Opt-in source identity for passages.

Chunks without ``source_key`` are never read or rewritten by this module.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Callable
from typing import Any, Literal

from src.constants.data import TextChunk
from src.core.saving.history_intent import cosine

Revision = Literal["plain", "insert", "unchanged"]
_PAGE = 100
_MAX_CHUNKS = 500
_EMBED_CANDIDATE_FACTOR = 3
_FETCH_BUFFER = 16


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
) -> str | None:
    body = (text or "").strip()
    if not body:
        return None
    superseded = None
    if isinstance(metadata, dict):
        superseded = metadata.get("superseded_at")
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
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


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
    if same:
        _mark(adapter, brain_id, stale, _stamp(source_timestamp))
        return "unchanged"
    _mark(adapter, brain_id, current, _stamp(source_timestamp))
    meta_keys["source_content_hash"] = digest
    return "insert"


def _mark(adapter: Any, brain_id: str, chunks: list[TextChunk], superseded_at: str) -> None:
    ids = [str(chunk.id) for chunk in chunks if getattr(chunk, "id", None)]
    if not ids:
        return
    adapter.mark_text_chunks_superseded(ids, superseded_at, brain_id)


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


def current_fact_candidates(
    adapter: Any,
    brain_id: str,
    candidates: list[dict],
    *,
    include_history: bool,
) -> list[dict]:
    """Drop facts whose source chunks are all older copies.

    Facts with no chunk id stay. A fact stays when any of its chunks is current
    or missing. The newest chunk for a source key is current until an older one
    is stamped superseded.
    """
    if include_history or not candidates:
        return candidates
    ids: list[str] = []
    for candidate in candidates:
        ids.extend(str(item) for item in (candidate.get("chunk_ids") or []) if item)
    if not ids:
        return candidates
    try:
        chunks, _ = adapter.get_text_chunks_by_ids(
            list(dict.fromkeys(ids))[:80], False, brain_id
        )
    except Exception:
        return candidates
    loaded = {str(chunk.id): chunk for chunk in chunks}
    cache: dict[str, set[str] | None] = {}
    kept: list[dict] = []
    for candidate in candidates:
        chunk_ids = [str(item) for item in (candidate.get("chunk_ids") or []) if item]
        if not chunk_ids or any(chunk_id not in loaded for chunk_id in chunk_ids):
            kept.append(candidate)
            continue
        if all(_older_copy(adapter, brain_id, loaded[chunk_id], cache) for chunk_id in chunk_ids):
            continue
        kept.append(candidate)
    return kept


def screen_passage_hits(
    adapter: Any,
    brain_id: str,
    hits: list[tuple[str, float, str]],
    *,
    include_history: bool,
) -> list[tuple[str, float, str]]:
    ids = [str(chunk_id) for chunk_id, _, _ in hits if chunk_id]
    if not ids:
        return hits
    try:
        chunks, _ = adapter.get_text_chunks_by_ids(ids, False, brain_id)
    except Exception:
        return hits
    by_id = {str(chunk.id): chunk for chunk in chunks}
    cache: dict[str, set[str] | None] = {}
    screened: list[tuple[str, float, str]] = []
    for chunk_id, score, body in hits:
        chunk = by_id.get(str(chunk_id))
        if chunk is None:
            screened.append((chunk_id, score, body))
            continue
        if not include_history and _older_copy(adapter, brain_id, chunk, cache):
            continue
        meta = getattr(chunk, "metadata", None) or {}
        if not meta.get("superseded_at"):
            screened.append((chunk_id, score, body))
            continue
        visible = visible_passage_body(
            getattr(chunk, "metadata", None),
            body or getattr(chunk, "text", None),
            include_history=include_history,
        )
        if visible is None:
            continue
        screened.append((chunk_id, score, visible))
    return screened


def embed_candidate_cap(limit: int) -> int:
    return max(0, limit) * _EMBED_CANDIDATE_FACTOR


def _as_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    embeddings = getattr(value, "embeddings", None)
    if embeddings is not None:
        value = embeddings
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _stored_embedding(chunk: Any) -> list[float] | None:
    for attr in ("embeddings", "embedding", "vector"):
        found = _as_vector(getattr(chunk, attr, None))
        if found:
            return found
    meta = getattr(chunk, "metadata", None) or {}
    if not isinstance(meta, dict):
        return None
    for key in ("embeddings", "embedding", "vector"):
        found = _as_vector(meta.get(key))
        if found:
            return found
    return None


def _lexical_overlap(query_text: str | None, body: str) -> float:
    if not query_text:
        return 0.0
    query_terms = set(query_text.lower().split())
    if not query_terms:
        return 0.0
    return float(len(query_terms & set(body.lower().split())))


def _batch_embed(
    texts: list[str],
    embed: Callable[[str], list[float]] | None,
    embed_many: Callable[[list[str]], list[list[float]]] | None,
) -> list[list[float] | None]:
    if not texts:
        return []
    for batch in (embed_many, getattr(embed, "embed_texts", None)):
        if not callable(batch):
            continue
        try:
            vectors = batch(texts)
        except Exception:
            continue
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            continue
        return [_as_vector(item) for item in vectors]
    return [None] * len(texts)


def _candidate_vectors(
    bodies: list[str],
    stored: list[list[float] | None],
    embed: Callable[[str], list[float]] | None,
    embed_many: Callable[[list[str]], list[list[float]]] | None,
) -> list[list[float] | None]:
    vectors = list(stored)
    missing = [index for index, item in enumerate(vectors) if not item]
    if not missing:
        return vectors
    batched = _batch_embed([bodies[index] for index in missing], embed, embed_many)
    if any(batched):
        for index, vector in zip(missing, batched):
            if vector:
                vectors[index] = vector
        missing = [index for index, item in enumerate(vectors) if not item]
    if missing and callable(embed) and not callable(getattr(embed, "embed_texts", None)):
        for index in missing:
            try:
                vectors[index] = _as_vector(embed(bodies[index]))
            except Exception:
                vectors[index] = None
    return vectors


def _revision_score(
    body: str,
    recency: str,
    *,
    query_text: str | None,
    query_vector: list[float] | None,
    vector: list[float] | None,
) -> tuple[float, str]:
    if query_vector and vector:
        return (cosine(query_vector, vector), recency)
    return (_lexical_overlap(query_text, body), recency)


def prior_revision_texts(
    adapter: Any,
    brain_id: str,
    hits: list[tuple[str, float, str]],
    *,
    limit: int,
    query_text: str | None = None,
    query_vector: list[float] | None = None,
    embed: Callable[[str], list[float]] | None = None,
    embed_many: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[str]:
    """Load superseded copies for the source keys already in these hits.

    Search does not have to rank the older copy. The key on a current hit is enough.
    When a query is provided, older copies are ordered by relevance, then recency.
    Embedding is capped to the newest copies and prefers stored vectors.
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
    embed_cap = embed_candidate_cap(cap)
    fetch_limit = min(_MAX_CHUNKS, max(embed_cap + _FETCH_BUFFER, embed_cap))
    collected: list[tuple[str, str, list[float] | None]] = []
    seen: set[str] = set()
    for key in keys:
        try:
            stored, _ = adapter.get_text_chunks(
                brain_id=brain_id,
                limit=fetch_limit,
                skip=0,
                metadata_eq={"source_key": key},
                order="desc",
            )
        except TypeError:
            try:
                stored, _ = adapter.get_text_chunks(
                    brain_id=brain_id,
                    limit=fetch_limit,
                    skip=0,
                    metadata_eq={"source_key": key},
                )
            except Exception:
                continue
        except Exception:
            continue
        older = [
            chunk
            for chunk in stored
            if (getattr(chunk, "metadata", None) or {}).get("superseded_at")
        ]
        older.sort(
            key=lambda chunk: str(
                (getattr(chunk, "metadata", None) or {}).get("superseded_at") or ""
            ),
            reverse=True,
        )
        for chunk in older:
            meta = getattr(chunk, "metadata", None) or {}
            body = visible_passage_body(
                meta,
                getattr(chunk, "text", None),
                include_history=True,
            )
            if not body or body in seen:
                continue
            seen.add(body)
            recency = str(meta.get("superseded_at") or "") if isinstance(meta, dict) else ""
            collected.append((recency, body, _stored_embedding(chunk)))
    collected.sort(key=lambda item: item[0], reverse=True)
    candidates = collected[:embed_cap]
    bodies = [body for _recency, body, _vector in candidates]
    vectors = _candidate_vectors(
        bodies,
        [vector for _recency, _body, vector in candidates],
        embed,
        embed_many,
    )
    ranked = [
        (
            *_revision_score(
                body,
                recency,
                query_text=query_text,
                query_vector=query_vector,
                vector=vector,
            ),
            body,
        )
        for (recency, body, _stored), vector in zip(candidates, vectors)
    ]
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [body for _score, _recency, body in ranked[:cap]]


def screened_passage_texts(
    adapter: Any,
    brain_id: str,
    passages: list[tuple[str, float, str]],
    *,
    include_history: bool,
    limit: int,
) -> list[str]:
    screened = screen_passage_hits(
        adapter,
        brain_id,
        passages,
        include_history=include_history,
    )
    return [text for _, _, text in screened if text][: max(0, limit)]


def current_record_texts(
    adapter: Any,
    brain_id: str,
    records: list,
    chunks: list[TextChunk],
    *,
    include_history: bool,
) -> list[str]:
    """JSON for structured records that are still a current passage.

    A record is left out when its text matches an older copy and no current
    copy. Records with other text stay. History keeps every record.
    """
    cache: dict[str, set[str] | None] = {}
    older: set[str] = set()
    current: set[str] = set()
    if not include_history:
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
        data = getattr(record, "data", None)
        if not isinstance(data, dict) or not data:
            continue
        body = data.get("text")
        if (
            not include_history
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
) -> str | None:
    if not include_history and _older_copy(adapter, brain_id, chunk, cache):
        return None
    return visible_passage_body(
        getattr(chunk, "metadata", None),
        getattr(chunk, "text", None),
        include_history=include_history,
    )

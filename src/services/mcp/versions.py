from __future__ import annotations

from typing import Any, Callable

from src.core.saving.source_passages import source_key_from_meta

CHANGELOG_LIMIT = 50
SOURCE_VERSION_LIMIT = 20
_SOURCE_FETCH = 500


def clamp_page(value: Any, default: int, cap: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed, cap))


def _dump(row: Any) -> Any:
    dump = getattr(row, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return row


def _copy(chunk: Any) -> dict[str, Any]:
    meta = getattr(chunk, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    item: dict[str, Any] = {
        "id": str(getattr(chunk, "id", "") or ""),
        "text": getattr(chunk, "text", None) or "",
        "superseded_at": meta.get("superseded_at"),
    }
    operation = meta.get("operation")
    if operation:
        item["operation"] = operation
    return item


def changelog_page(
    adapter: Any,
    brain_id: str,
    *,
    query_text: str | None = None,
    types: list[str] | None = None,
    limit: Any = 10,
    skip: Any = 0,
    since: Any = None,
    until: Any = None,
    graph: Any | None = None,
) -> dict[str, Any]:
    from src.core.saving.validity import InvalidAsOf, optional_instant, utc_now
    from src.core.saving.validity_changes import collect_validity_changes
    from datetime import datetime, timezone

    bounded = clamp_page(limit, 10, CHANGELOG_LIMIT)
    try:
        offset = max(0, int(skip or 0))
    except (TypeError, ValueError):
        offset = 0
    try:
        since_dt = optional_instant(since, name="since")
        until_dt = optional_instant(until, name="until")
    except InvalidAsOf as exc:
        return f"Error: {exc}"
    if since_dt is not None or until_dt is not None:
        window_start = since_dt or datetime(1970, 1, 1, tzinfo=timezone.utc)
        window_end = until_dt or utc_now()
        payload = collect_validity_changes(
            adapter,
            graph,
            brain_id,
            since=window_start,
            until=window_end,
            limit=bounded,
            skip=offset,
            query_text=query_text,
            types=types,
        )
        try:
            rows = adapter.get_changelogs_list(
                brain_id,
                bounded,
                offset,
                types or None,
                query_text or None,
                window_start,
                window_end,
            )
        except TypeError:
            rows = adapter.get_changelogs_list(
                brain_id,
                bounded,
                offset,
                types or None,
                query_text or None,
            )
        payload["changelogs"] = [_dump(row) for row in rows or []]
        return payload
    try:
        rows = adapter.get_changelogs_list(
            brain_id,
            bounded,
            offset,
            types or None,
            query_text or None,
        )
    except TypeError:
        rows = adapter.get_changelogs_list(
            brain_id,
            bounded,
            offset,
            types or None,
            query_text or None,
        )
    changelogs = [_dump(row) for row in rows or []]
    return {"changelogs": changelogs, "count": len(changelogs)}


def changelog_entry(adapter: Any, brain_id: str, changelog_id: str) -> dict[str, Any] | str:
    if not str(changelog_id or "").strip():
        return "Error: changelog id is required"
    row = adapter.get_changelog_by_id(str(changelog_id).strip(), brain_id)
    if not row:
        return "Error: changelog not found"
    return {"changelog": _dump(row)}


def read_changelogs(
    adapter: Any,
    brain_id: str,
    *,
    authorized: bool,
    query_text: str | None = None,
    types: list[str] | None = None,
    limit: Any = 10,
    skip: Any = 0,
    since: Any = None,
    until: Any = None,
    graph: Any | None = None,
) -> dict[str, Any] | str:
    if not authorized:
        return "Unauthorized"
    try:
        return changelog_page(
            adapter,
            brain_id,
            query_text=query_text,
            types=types,
            limit=limit,
            skip=skip,
            since=since,
            until=until,
            graph=graph,
        )
    except Exception as exc:
        return f"Error listing changelogs: {exc}"


def read_changelog(
    adapter: Any,
    brain_id: str,
    changelog_id: str,
    *,
    authorized: bool,
) -> dict[str, Any] | str:
    if not authorized:
        return "Unauthorized"
    try:
        return changelog_entry(adapter, brain_id, changelog_id)
    except Exception as exc:
        return f"Error reading changelog: {exc}"


def _versions_for_key(adapter: Any, brain_id: str, source_key: str, limit: int) -> dict[str, Any]:
    stored, _ = adapter.get_text_chunks(
        brain_id=brain_id,
        limit=_SOURCE_FETCH,
        skip=0,
        metadata_eq={"source_key": source_key},
    )
    current: list[dict[str, Any]] = []
    previous: list[dict[str, Any]] = []
    for chunk in stored or []:
        meta = getattr(chunk, "metadata", None) or {}
        item = _copy(chunk)
        if isinstance(meta, dict) and meta.get("superseded_at"):
            previous.append(item)
        else:
            current.append(item)
    previous.sort(key=lambda item: str(item.get("superseded_at") or ""), reverse=True)
    return {
        "source_key": source_key,
        "current": current[:limit],
        "previous": previous[:limit],
    }


def _keys_from_hits(adapter: Any, brain_id: str, hits: list[Any]) -> list[str]:
    ids: list[str] = []
    for hit in hits or []:
        meta = getattr(hit, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        resource_id = meta.get("resource_id") or getattr(hit, "id", None)
        if resource_id:
            ids.append(str(resource_id))
    if not ids:
        return []
    chunks, _ = adapter.get_text_chunks_by_ids(ids, False, brain_id)
    keys: list[str] = []
    for chunk in chunks or []:
        key = source_key_from_meta(getattr(chunk, "metadata", None))
        if key and key not in keys:
            keys.append(key)
    return keys


def list_source_versions(
    adapter: Any,
    brain_id: str,
    *,
    source_key: str = "",
    query: str = "",
    limit: Any = 10,
    embed: Callable[[str], Any] | None = None,
    search: Callable[[Any, str], list[Any]] | None = None,
) -> dict[str, Any] | str:
    bounded = clamp_page(limit, 10, SOURCE_VERSION_LIMIT)
    key = str(source_key or "").strip()
    text = str(query or "").strip()
    if key:
        try:
            return {"sources": [_versions_for_key(adapter, brain_id, key, bounded)]}
        except Exception:
            return "Error: could not load source versions"
    if not text:
        return "Error: provide source_key or query"
    if embed is None or search is None:
        return "Error: embedding search failed"
    try:
        vector = embed(text)
        embeddings = getattr(vector, "embeddings", None)
        if embeddings is not None:
            vector = embeddings
        if not vector:
            return "Error: embedding search failed"
        hits = search(vector, brain_id)
    except Exception:
        return "Error: embedding search failed"
    try:
        keys = _keys_from_hits(adapter, brain_id, hits)[:bounded]
        return {
            "sources": [_versions_for_key(adapter, brain_id, item, bounded) for item in keys]
        }
    except Exception:
        return "Error: could not load source versions"


def read_source_versions(
    adapter: Any,
    brain_id: str,
    *,
    authorized: bool,
    source_key: str = "",
    query: str = "",
    limit: Any = 10,
    embed: Callable[[str], Any] | None = None,
    search: Callable[[Any, str], list[Any]] | None = None,
) -> dict[str, Any] | str:
    if not authorized:
        return "Unauthorized"
    return list_source_versions(
        adapter,
        brain_id,
        source_key=source_key,
        query=query,
        limit=limit,
        embed=embed,
        search=search,
    )


def invalidated_hops(hops: list[dict] | None) -> list[dict]:
    selected: list[dict] = []
    for hop in hops or []:
        predicate = hop.get("predicate") or {}
        if not isinstance(predicate, dict):
            continue
        props = predicate.get("properties") or {}
        if not isinstance(props, dict):
            continue
        if props.get("invalid_at") or props.get("deprecated") is True:
            selected.append(hop)
    return selected

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, Field

from src.core.saving.source_passages import source_key_from_meta
from src.core.saving.validity import (
    close_validity,
    format_instant,
    parse_instant,
    utc_now,
)

DEFAULT_INTERVAL_DAYS = 7
MAX_PAGES = 999
STALE_CLAIM = timedelta(hours=1)
CHUNK_PAGE = 100
CHUNK_SCAN_CAP = 5000
REMOVED_REASON = "source page removed"


class CrawlSpec(BaseModel):
    brain_id: str
    url: str
    max_pages: int = Field(default=10, ge=1, le=MAX_PAGES)
    allowed_domains: list[str] = Field(default_factory=list)
    allowed_path_prefixes: list[str] = Field(default_factory=list)
    interval_days: int = Field(default=DEFAULT_INTERVAL_DAYS, ge=1, le=365)
    enabled: bool = True
    skip_enrichment: bool = False
    group_tracking: Optional[str] = None
    last_crawled_at: Optional[str] = None
    last_started_at: Optional[str] = None
    last_seen_source_keys: list[str] = Field(default_factory=list)
    last_task_id: Optional[str] = None
    last_error: Optional[str] = None
    last_conflict_scan_task_id: Optional[str] = None
    pages_ingested: int = 0
    pages_removed: int = 0

    def crawl_root(self) -> str:
        return crawl_root_key(self.url)


def spec_from_document(doc: Any, brain_id: str | None = None) -> CrawlSpec | None:
    if not isinstance(doc, dict):
        return None
    payload = dict(doc)
    if brain_id:
        payload["brain_id"] = brain_id
    try:
        return CrawlSpec.model_validate(payload)
    except Exception:
        return None


def page_source_key(page: Any) -> Optional[str]:
    canonical = getattr(page, "canonical_url", None)
    url = getattr(page, "url", None)
    return normalize_source_key(canonical or url)


def normalize_source_key(url: str | None) -> Optional[str]:
    raw = str(url or "").strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    host = (parsed.hostname or parsed.netloc).lower()
    if parsed.port and parsed.port != 443:
        host = f"{host}:{parsed.port}"
    path = parsed.path or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"https://{host}{path}{query}"


def crawl_root_key(url: str | None) -> str:
    key = normalize_source_key(url)
    if key:
        return key
    return str(url or "").strip()


def spec_is_due(
    spec: CrawlSpec | None,
    *,
    now: datetime | None = None,
    force: bool = False,
) -> bool:
    if spec is None or not spec.enabled:
        return False
    if force:
        return True
    instant = now or utc_now()
    crawled = parse_instant(spec.last_crawled_at)
    if crawled is not None and crawled + timedelta(days=spec.interval_days) > instant:
        return False
    started = parse_instant(spec.last_started_at)
    if started is None:
        return True
    if crawled is not None and started < crawled:
        return True
    return started + STALE_CLAIM <= instant


def mark_started(spec: CrawlSpec, *, now: datetime | None = None, task_id: str | None = None) -> CrawlSpec:
    stamp = format_instant(now or utc_now())
    payload = spec.model_copy(deep=True)
    payload.last_started_at = stamp
    payload.last_error = None
    if task_id:
        payload.last_task_id = task_id
    return payload


def mark_finished(
    spec: CrawlSpec,
    *,
    live_source_keys: Iterable[str],
    removed_source_keys: Iterable[str],
    conflict_scan_task_id: str | None,
    now: datetime | None = None,
    error: str | None = None,
) -> CrawlSpec:
    stamp = format_instant(now or utc_now())
    payload = spec.model_copy(deep=True)
    live = sorted({str(key) for key in live_source_keys if key})
    payload.last_seen_source_keys = live
    payload.last_conflict_scan_task_id = conflict_scan_task_id
    payload.pages_ingested = len(live)
    payload.pages_removed = len({str(key) for key in removed_source_keys if key})
    if error:
        payload.last_error = str(error).strip() or error
    else:
        payload.last_crawled_at = stamp
        payload.last_error = None
    return payload


def claim_crawl_spec(
    adapter: Any,
    brain_id: str,
    *,
    now: datetime | None = None,
    force: bool = False,
    task_id: str | None = None,
) -> CrawlSpec | None:
    claimer = getattr(adapter, "claim_due_crawl_spec", None)
    if callable(claimer):
        return claimer(brain_id, now=now, force=force, task_id=task_id)
    getter = getattr(adapter, "get_crawl_spec", None)
    saver = getattr(adapter, "save_crawl_spec", None)
    if not callable(getter) or not callable(saver):
        return None
    spec = getter(brain_id)
    if not spec_is_due(spec, now=now, force=force):
        return None
    claimed = mark_started(spec, now=now, task_id=task_id)
    return saver(claimed, brain_id)


def _is_current_chunk(chunk: Any) -> bool:
    meta = getattr(chunk, "metadata", None) or {}
    return not (meta.get("superseded_at") or meta.get("valid_to"))


def list_current_crawl_chunks(adapter: Any, brain_id: str, crawl_root: str) -> list[Any]:
    getter = getattr(adapter, "get_text_chunks", None)
    if not callable(getter) or not crawl_root:
        return []
    found: list[Any] = []
    skip = 0
    while skip < CHUNK_SCAN_CAP:
        chunks, total = getter(
            brain_id=brain_id,
            limit=CHUNK_PAGE,
            skip=skip,
            metadata_eq={"crawl_root": crawl_root},
        )
        if not chunks:
            break
        found.extend(chunk for chunk in chunks if _is_current_chunk(chunk))
        skip += len(chunks)
        if total is not None and skip >= int(total):
            break
    return found


def list_chunks_for_source_keys(
    adapter: Any, brain_id: str, source_keys: Iterable[str]
) -> list[Any]:
    getter = getattr(adapter, "get_text_chunks", None)
    if not callable(getter):
        return []
    found: list[Any] = []
    seen: set[str] = set()
    for key in source_keys:
        if not key or key in seen:
            continue
        seen.add(key)
        chunks, _ = getter(
            brain_id=brain_id,
            limit=CHUNK_PAGE,
            skip=0,
            metadata_eq={"source_key": key},
        )
        found.extend(chunk for chunk in chunks if _is_current_chunk(chunk))
    return found


def close_removed_pages(
    data: Any,
    graph: Any,
    brain_id: str,
    *,
    live_source_keys: Iterable[str],
    crawl_root: str,
    previous_source_keys: Iterable[str] | None = None,
    closed_at: str | None = None,
) -> list[str]:
    live = {str(key) for key in live_source_keys if key}
    stamp = format_instant(parse_instant(closed_at) or utc_now())
    chunks = list_current_crawl_chunks(data, brain_id, crawl_root_key(crawl_root))
    missing = [
        str(key)
        for key in (previous_source_keys or [])
        if key and str(key) not in live
    ]
    if missing:
        chunks.extend(list_chunks_for_source_keys(data, brain_id, missing))
    removed_keys: list[str] = []
    ids: list[str] = []
    seen_ids: set[str] = set()
    for chunk in chunks:
        key = source_key_from_meta(getattr(chunk, "metadata", None))
        if not key or key in live:
            continue
        removed_keys.append(key)
        chunk_id = str(getattr(chunk, "id", "") or "")
        if chunk_id and chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            ids.append(chunk_id)
    if ids:
        data.mark_text_chunks_superseded(
            ids,
            stamp,
            brain_id,
            valid_to=stamp,
        )
        close_facts_for_chunks(graph, brain_id, ids, stamp)
    return sorted(set(removed_keys))


def close_facts_for_chunks(
    graph: Any,
    brain_id: str,
    chunk_ids: list[str],
    closed_at: str,
) -> list[str]:
    if not graph or not chunk_ids:
        return []
    citing = getattr(graph, "nodes_and_edges_citing", None)
    if not callable(citing):
        return []
    nodes, triples = citing(brain_id, chunk_ids, [])
    closed: list[str] = []
    for triple in triples or []:
        predicate = triple[1] if len(triple) > 1 else None
        fact_id = _entity_id(predicate)
        if _close_open_fact(graph, brain_id, fact_id, predicate, "relationship", closed_at):
            closed.append(fact_id)
    from src.core.verification.facts import node_is_verifiable_event

    for node in nodes or []:
        if not node_is_verifiable_event(getattr(node, "labels", None)):
            continue
        fact_id = _entity_id(node)
        if _close_open_fact(graph, brain_id, fact_id, node, "node", closed_at):
            closed.append(fact_id)
    return closed


def enqueue_conflict_scan(brain_id: str) -> str:
    from src.workers.tasks.conflicts import scan_conflicts
    from src.workers.tasks.ingestion import set_ingestion_task_status

    task_id = str(uuid4())
    set_ingestion_task_status(task_id, brain_id, "queued", stage="conflict_scan")
    scan_conflicts.apply_async(
        args=[{"brain_id": brain_id, "use_llm": False}],
        task_id=task_id,
    )
    return task_id


def _entity_id(entity: Any) -> str:
    if entity is None:
        return ""
    return str(getattr(entity, "uuid", "") or "")


def _props_map(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        nested = value.get("properties")
        if isinstance(nested, Mapping):
            merged = dict(nested)
            merged.update({k: v for k, v in value.items() if k != "properties"})
            return merged
        return dict(value)
    props = getattr(value, "properties", None)
    return dict(props) if isinstance(props, Mapping) else {}


def _is_open(props: Mapping[str, Any]) -> bool:
    return not (props.get("invalid_at") or props.get("valid_to"))


def _close_open_fact(
    graph: Any,
    brain_id: str,
    fact_id: str,
    entity: Any,
    updating: str,
    closed_at: str,
) -> bool:
    if not fact_id:
        return False
    props = _props_map(entity)
    if not _is_open(props):
        return False
    closed_props = close_validity(props, closed_at=closed_at, reason=REMOVED_REASON)
    patch = {
        "invalid_at": closed_props.get("invalid_at") or closed_at,
        "deprecated": True,
        "valid_to": closed_props.get("valid_to") or closed_at,
        "retired_reason": closed_props.get("retired_reason") or REMOVED_REASON,
    }
    if updating == "relationship":
        from src.core.intent.retire import apply_open_update

        return apply_open_update(graph, fact_id, brain_id, patch)
    updater = getattr(graph, "update_properties", None)
    if not callable(updater):
        return False
    try:
        updated = updater(
            fact_id,
            "node",
            brain_id=brain_id,
            new_properties=patch,
        )
    except TypeError:
        updated = updater(fact_id, "node", brain_id, patch, [])
    return updated is not None

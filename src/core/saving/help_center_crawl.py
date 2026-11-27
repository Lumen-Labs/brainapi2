from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, Field

from src.core.saving.source_citations import close_orphaned_citing_facts
from src.core.saving.source_passages import source_key_from_meta
from src.core.saving.validity import (
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
GONE_STATUS_CODES = frozenset({404, 410})
DEFAULT_RETIRE_MAX_RATIO = 0.2
DEFAULT_RETIRE_MAX_PAGES = 50
DEFAULT_RECHECK_MAX_PAGES = 50


class RecheckQuotaStop(RuntimeError):
    pass


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


def spec_in_flight(
    spec: CrawlSpec | None,
    *,
    now: datetime | None = None,
) -> bool:
    if spec is None:
        return False
    instant = now or utc_now()
    started = parse_instant(spec.last_started_at)
    if started is None:
        return False
    crawled = parse_instant(spec.last_crawled_at)
    if crawled is not None and started < crawled:
        return False
    return started + STALE_CLAIM > instant


def spec_is_due(
    spec: CrawlSpec | None,
    *,
    now: datetime | None = None,
    force: bool = False,
) -> bool:
    if spec is None or not spec.enabled:
        return False
    if spec_in_flight(spec, now=now):
        return False
    if force:
        return True
    instant = now or utc_now()
    crawled = parse_instant(spec.last_crawled_at)
    if crawled is not None and crawled + timedelta(days=spec.interval_days) > instant:
        return False
    return True


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
    warning: str | None = None,
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
        payload.last_error = str(warning).strip() if warning else None
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


def page_is_gone(page: Any) -> bool:
    try:
        status = int(getattr(page, "status_code", None) or 0)
    except (TypeError, ValueError):
        status = 0
    return status in GONE_STATUS_CODES


def page_is_failed(page: Any) -> bool:
    if page_is_gone(page):
        return False
    try:
        status = int(getattr(page, "status_code", None) or 0)
    except (TypeError, ValueError):
        status = 0
    if status >= 500 or status in {401, 403, 429}:
        return True
    text = str(getattr(page, "text", None) or "").strip()
    skip_reason = str(getattr(page, "skip_reason", None) or "").strip()
    if skip_reason and not text:
        return True
    return False


def classify_crawl_pages(
    pages: Iterable[Any],
    spec: CrawlSpec,
) -> dict[str, Any]:
    live_keys: list[str] = []
    gone_keys: list[str] = []
    failed_keys: list[str] = []
    ingestible: list[Any] = []
    seen = 0
    for page in pages:
        seen += 1
        key = page_source_key(page)
        if page_is_gone(page):
            if key:
                gone_keys.append(key)
            continue
        if page_is_failed(page):
            if key:
                failed_keys.append(key)
            continue
        text = str(getattr(page, "text", None) or "").strip()
        if text:
            if key:
                live_keys.append(key)
            ingestible.append(page)
        elif key:
            failed_keys.append(key)
    max_pages = int(spec.max_pages or 0)
    return {
        "live_keys": live_keys,
        "gone_keys": gone_keys,
        "failed_keys": failed_keys,
        "ingestible": ingestible,
        "page_count": seen,
        "truncated": bool(max_pages and seen >= max_pages),
    }


def retirement_cap(
    previous_count: int,
    *,
    ratio: float | None = None,
    max_pages: int | None = None,
) -> int:
    cap_ratio = DEFAULT_RETIRE_MAX_RATIO if ratio is None else float(ratio)
    cap_pages = DEFAULT_RETIRE_MAX_PAGES if max_pages is None else int(max_pages)
    if cap_pages < 1:
        cap_pages = DEFAULT_RETIRE_MAX_PAGES
    if previous_count < 1:
        return cap_pages
    scaled = int(previous_count * cap_ratio)
    return max(1, min(cap_pages, scaled if scaled > 0 else 1))


def retirement_abort_reason(
    classified: dict[str, Any],
    spec: CrawlSpec,
    *,
    ratio: float | None = None,
    max_pages: int | None = None,
) -> str | None:
    gone = {str(key) for key in classified.get("gone_keys") or [] if key}
    failed = {str(key) for key in classified.get("failed_keys") or [] if key}
    live = {str(key) for key in classified.get("live_keys") or [] if key}
    previous = {str(key) for key in (spec.last_seen_source_keys or []) if key}
    if failed and not live and not gone and previous:
        return "crawl produced no usable pages; retirement skipped"
    cap = retirement_cap(len(previous), ratio=ratio, max_pages=max_pages)
    if len(gone) > cap:
        return f"gone page count {len(gone)} exceeds retirement cap {cap}"
    return None


def discovered_source_keys(classified: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("live_keys", "gone_keys", "failed_keys"):
        keys.update(str(key) for key in classified.get(field) or [] if key)
    return keys


def previous_keys_not_discovered(
    classified: dict[str, Any],
    previous_keys: Iterable[str] | None,
) -> list[str]:
    seen = discovered_source_keys(classified)
    found = []
    for key in previous_keys or []:
        text = str(key or "").strip()
        if text and text not in seen and text not in found:
            found.append(text)
    return found


def merge_classified(
    classified: dict[str, Any], extra: dict[str, Any]
) -> dict[str, Any]:
    merged = {
        "live_keys": list(classified.get("live_keys") or [])
        + list(extra.get("live_keys") or []),
        "gone_keys": list(classified.get("gone_keys") or [])
        + list(extra.get("gone_keys") or []),
        "failed_keys": list(classified.get("failed_keys") or [])
        + list(extra.get("failed_keys") or []),
        "ingestible": list(classified.get("ingestible") or [])
        + list(extra.get("ingestible") or []),
        "page_count": int(classified.get("page_count") or 0)
        + int(extra.get("page_count") or 0),
        "truncated": bool(classified.get("truncated"))
        or bool(extra.get("truncated")),
    }
    return merged


def recheck_unseen_previous_pages(
    classified: dict[str, Any],
    spec: CrawlSpec,
    fetch_pages: Callable[[str], Iterable[Any]],
    *,
    budget: int | None = None,
) -> dict[str, Any]:
    urls = previous_keys_not_discovered(classified, spec.last_seen_source_keys)
    extra_pages: list[Any] = []
    remaining: list[str] = []
    stopped = None
    limit = None if budget is None else max(0, int(budget))
    for index, url in enumerate(urls):
        if limit is not None and index >= limit:
            remaining = urls[index:]
            stopped = f"recheck budget {limit} exhausted"
            break
        try:
            fetched = list(fetch_pages(url) or [])
        except RecheckQuotaStop as exc:
            remaining = urls[index:]
            stopped = str(exc) or "recheck stopped on quota"
            break
        extra_pages.extend(fetched or [_failed_recheck_page(url)])
    merged = classified
    if extra_pages:
        merged = merge_classified(classified, classify_crawl_pages(extra_pages, spec))
    merged["recheck_remaining"] = remaining
    merged["recheck_stopped"] = stopped
    return merged


def retained_source_keys(
    spec: CrawlSpec,
    classified: dict[str, Any],
    removed: Iterable[str],
    *,
    aborted: bool,
) -> list[str]:
    previous = {str(key) for key in (spec.last_seen_source_keys or []) if key}
    live = {str(key) for key in classified.get("live_keys") or [] if key}
    if aborted:
        return sorted(previous)
    retired = {str(key) for key in removed if key}
    return sorted((previous | live) - retired)


def _failed_recheck_page(url: str) -> Any:
    return SimpleNamespace(
        url=url,
        canonical_url=url,
        text="",
        status_code=None,
        skip_reason="recheck failed",
    )


def close_removed_pages(
    data: Any,
    graph: Any,
    brain_id: str,
    *,
    live_source_keys: Iterable[str] | None = None,
    gone_source_keys: Iterable[str] | None = None,
    crawl_root: str,
    previous_source_keys: Iterable[str] | None = None,
    closed_at: str | None = None,
) -> list[str]:
    gone = {str(key) for key in (gone_source_keys or []) if key}
    if not gone:
        return []
    stamp = format_instant(parse_instant(closed_at) or utc_now())
    chunks = list_chunks_for_source_keys(data, brain_id, gone)
    if crawl_root:
        for chunk in list_current_crawl_chunks(data, brain_id, crawl_root_key(crawl_root)):
            key = source_key_from_meta(getattr(chunk, "metadata", None))
            if key in gone:
                chunks.append(chunk)
    removed_keys: list[str] = []
    ids: list[str] = []
    seen_ids: set[str] = set()
    for chunk in chunks:
        key = source_key_from_meta(getattr(chunk, "metadata", None))
        if not key or key not in gone:
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
        close_facts_for_chunks(graph, brain_id, ids, stamp, data=data)
    return sorted(set(removed_keys))


def close_facts_for_chunks(
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
        reason=REMOVED_REASON,
    )


def finish_crawl_spec(
    adapter: Any,
    brain_id: str,
    *,
    live_source_keys: Iterable[str],
    removed_source_keys: Iterable[str],
    conflict_scan_task_id: str | None,
    task_id: str | None = None,
    now: datetime | None = None,
    error: str | None = None,
    warning: str | None = None,
):
    getter = getattr(adapter, "get_crawl_spec", None)
    saver = getattr(adapter, "save_crawl_spec", None)
    finisher = getattr(adapter, "finish_crawl_spec", None)
    if callable(finisher):
        return finisher(
            brain_id,
            live_source_keys=live_source_keys,
            removed_source_keys=removed_source_keys,
            conflict_scan_task_id=conflict_scan_task_id,
            task_id=task_id,
            now=now,
            error=error,
            warning=warning,
        )
    if not callable(getter) or not callable(saver):
        return None
    current = getter(brain_id)
    if current is None:
        return None
    if (
        task_id
        and current.last_task_id
        and current.last_task_id != task_id
    ):
        return current
    finished = mark_finished(
        current,
        live_source_keys=live_source_keys,
        removed_source_keys=removed_source_keys,
        conflict_scan_task_id=conflict_scan_task_id,
        now=now,
        error=error,
        warning=warning,
    )
    return saver(finished, brain_id)


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

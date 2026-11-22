"""Atomic unstaged OCR page storage and one-shot dispatch."""

from time import time

from src.core.saving.ingest_staging import (
    SMALL_REMOTE_OCR_DEADLINE_SECONDS,
    TERMINAL_TASK_STATUSES,
    _ocr_partial_error,
    is_remote_ocr_run,
)

PAGE_PREFIX = "p:"
FAIL_PREFIX = "f:"
TOTAL_FIELD = "total"
LOCK_FIELD = "lock"
ANON_FIELD = "anon"
PAGE_TTL_SECONDS = 3600 * 24 * 7
INTERNAL_TASK_KEYS = (
    "ocr_page_texts",
    "ocr_failed_pages",
    "ocr_dispatch_started",
)


class SnapshotUnavailable(Exception):
    """Raised when the OCR page hash cannot be read."""


class Snapshot:
    def __init__(self, pages, failed, total, dispatched=False):
        self.pages = pages
        self.failed = failed
        self.total = total
        self.dispatched = dispatched

    @property
    def complete(self):
        return bool(self.total) and (len(self.pages) + len(self.failed)) >= int(self.total)


def pages_key(task_id):
    return f"ocr_pages:{task_id}"


def _cache():
    from src.workers.tasks.ingestion import cache_adapter

    return cache_adapter


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _hash_call(name, *args, **kwargs):
    method = getattr(_cache(), name, None)
    if not callable(method):
        return None
    return method(*args, **kwargs)


def _touch(task_id, brain_id):
    _hash_call("expire", pages_key(task_id), PAGE_TTL_SECONDS, brain_id=brain_id)


def load_pages(brain_id, task_id, cached=None):
    cached = cached or {}
    pages = {
        str(key): value for key, value in dict(cached.get("ocr_page_texts") or {}).items()
    }
    failed = [str(item) for item in (cached.get("ocr_failed_pages") or [])]
    total = cached.get("ocr_total_pages")
    dispatched = bool(cached.get("ocr_dispatch_started"))
    try:
        raw = _hash_call("hgetall", pages_key(task_id), brain_id=brain_id)
    except Exception as exc:
        raise SnapshotUnavailable from exc
    parsed = {_decode(key): _decode(value) for key, value in (raw or {}).items()}
    for field, value in parsed.items():
        if field.startswith(PAGE_PREFIX) and value:
            pages[field[len(PAGE_PREFIX):]] = value
        elif field.startswith(FAIL_PREFIX):
            key = field[len(FAIL_PREFIX):]
            if key not in failed:
                failed.append(key)
        elif field == TOTAL_FIELD and value:
            total = _raised_total(total, value)
        elif field == LOCK_FIELD and value:
            dispatched = True
    failed = [key for key in failed if key not in pages]
    return Snapshot(pages, failed, _raised_total(None, total), dispatched)


def _raised_total(stored, incoming):
    values = []
    for value in (stored, incoming):
        if value in (None, "", False):
            continue
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def page_key(brain_id, task_id, page_index):
    if page_index is not None:
        return str(int(page_index))
    count = _hash_call("hincrby", pages_key(task_id), ANON_FIELD, 1, brain_id=brain_id)
    _touch(task_id, brain_id)
    if count is None:
        snapshot = load_pages(brain_id, task_id)
        return f"anon:{len(snapshot.pages) + len(snapshot.failed)}"
    return f"anon:{int(count) - 1}"


def _store_field(brain_id, task_id, field, value):
    stored = _hash_call(
        "hset_unless_locked",
        pages_key(task_id),
        LOCK_FIELD,
        field,
        value,
        brain_id=brain_id,
    )
    if stored is None:
        if load_pages(brain_id, task_id).dispatched:
            return False
        _hash_call("hset", pages_key(task_id), field, value, brain_id=brain_id)
        stored = True
    if stored:
        _touch(task_id, brain_id)
    return bool(stored)


def record_page(brain_id, task_id, key, text):
    stored = _store_field(brain_id, task_id, f"{PAGE_PREFIX}{key}", text)
    if stored:
        _hash_call("hdel", pages_key(task_id), f"{FAIL_PREFIX}{key}", brain_id=brain_id)
    return stored


def record_failure(brain_id, task_id, key):
    snapshot = load_pages(brain_id, task_id)
    if snapshot.dispatched:
        return False
    if key in snapshot.pages or key in snapshot.failed:
        return True
    return _store_field(brain_id, task_id, f"{FAIL_PREFIX}{key}", "1")


def raise_total(brain_id, task_id, total):
    incoming = _raised_total(None, total)
    if incoming is None:
        return load_pages(brain_id, task_id).total
    raised = _hash_call(
        "hmax", pages_key(task_id), TOTAL_FIELD, incoming, brain_id=brain_id,
    )
    if raised is None:
        stored = load_pages(brain_id, task_id).total or 0
        if incoming > stored:
            _hash_call(
                "hset", pages_key(task_id), TOTAL_FIELD, str(incoming), brain_id=brain_id,
            )
            raised = incoming
        else:
            raised = stored
    _touch(task_id, brain_id)
    return int(raised) if raised else None


def claim_dispatch(brain_id, task_id):
    return bool(
        _hash_call("hsetnx", pages_key(task_id), LOCK_FIELD, "1", brain_id=brain_id)
    )


def release_claim(brain_id, task_id):
    _hash_call("hdel", pages_key(task_id), LOCK_FIELD, brain_id=brain_id)


def clear_page_text(brain_id, task_id, snapshot=None):
    fields = []
    if snapshot is None:
        try:
            raw = _hash_call("hgetall", pages_key(task_id), brain_id=brain_id) or {}
        except Exception:
            return
        for field in raw:
            name = _decode(field)
            if name.startswith(PAGE_PREFIX) or name.startswith(FAIL_PREFIX):
                fields.append(name)
    else:
        fields = [f"{PAGE_PREFIX}{key}" for key in snapshot.pages]
        fields.extend(f"{FAIL_PREFIX}{key}" for key in snapshot.failed)
    for name in fields:
        _hash_call("hdel", pages_key(task_id), name, brain_id=brain_id)


def _page_order(key):
    text = str(key)
    if text.isdigit():
        return (0, int(text))
    if text.startswith("anon:"):
        suffix = text[5:]
        if suffix.isdigit():
            return (1, int(suffix))
        return (1, suffix)
    return (2, text)


def joined_text(pages, total=None):
    items = [(_page_order(key), text) for key, text in (pages or {}).items()]
    items.sort()
    if total:
        items = [
            item for item in items
            if item[0][0] != 0 or item[0][1] < int(total)
        ]
    return "\n\n".join(text for _, text in items if text)


def task_extra(cached, snapshot, *, first=False, closing=False):
    extra = {
        "ocr_pages_accepted": len(snapshot.pages),
        "ocr_pages_failed": len(snapshot.failed),
        "ocr_page_texts": None,
        "ocr_failed_pages": None,
    }
    if snapshot.total:
        extra["ocr_total_pages"] = snapshot.total
    if cached.get("small_remote_ocr") and first:
        extra["remote_ocr_deadline"] = time() + SMALL_REMOTE_OCR_DEADLINE_SECONDS
    incomplete = bool(snapshot.failed) or (
        not snapshot.total
        or len(snapshot.pages) + len(snapshot.failed) < int(snapshot.total)
    )
    if closing and incomplete:
        extra["ocr_partial"] = True
    return extra


def public_task_extra(extra):
    extra = dict(extra or {})
    extra["ocr_page_texts"] = None
    extra["ocr_failed_pages"] = None
    extra["ocr_dispatch_started"] = True
    return extra


def already_dispatched(brain_id, task_id, cached=None):
    cached = cached or {}
    if cached.get("ocr_dispatch_started") or cached.get("status") in TERMINAL_TASK_STATUSES:
        return True
    return load_pages(brain_id, task_id, cached).dispatched


def record_callback(brain_id, task_id, cached, *, page_index, total_pages, text, failed):
    prior = load_pages(brain_id, task_id, cached)
    first = not prior.pages and not prior.failed
    if prior.dispatched:
        return prior, first, None
    key = page_key(brain_id, task_id, page_index)
    stored = record_failure(brain_id, task_id, key) if failed else record_page(
        brain_id, task_id, key, text,
    )
    if stored:
        raise_total(brain_id, task_id, total_pages)
    return load_pages(brain_id, task_id, cached), first, key


def _enqueue_text(brain_id, task_id, payload, context=None):
    from src.core.saving.ingest_cancellation import register
    from src.workers.tasks.ingestion import ingest_data as ingest_data_task

    headers = register(brain_id, task_id)
    if context:
        headers["pipeline_operation"] = context
    ingest_data_task.apply_async(
        args=[payload],
        task_id=task_id,
        **({"headers": headers} if headers else {}),
    )


def _write_dispatch_status(brain_id, task_id, cached, extra, error=None):
    from src.workers.tasks.ingestion import set_ingestion_task_status

    kwargs = {
        "stage": "processing",
        "group_tracking": cached.get("group_tracking"),
        "extra": public_task_extra(extra),
    }
    if error:
        kwargs["error"] = error
    return set_ingestion_task_status(task_id, brain_id, "started", **kwargs)


_LIVE_SNAPSHOT_KEYS = {
    "ocr_pages_accepted",
    "ocr_pages_failed",
    "ocr_total_pages",
    "ocr_page_texts",
    "ocr_failed_pages",
}


def _merged_extra(cached, snapshot, extra):
    built = task_extra(cached, snapshot, closing=True)
    if not extra:
        return built
    if extra.get("ocr_partial"):
        built["ocr_partial"] = True
    for key, value in extra.items():
        if key in _LIVE_SNAPSHOT_KEYS or key == "ocr_partial":
            continue
        built[key] = value
    return built


def _claim_live_snapshot(brain_id, task_id, cached, snapshot):
    if already_dispatched(brain_id, task_id, cached) or not snapshot.pages:
        return None
    if not claim_dispatch(brain_id, task_id):
        return None
    try:
        _touch(task_id, brain_id)
        current = load_pages(brain_id, task_id, cached)
        if current.pages:
            return current
    except Exception:
        release_claim(brain_id, task_id)
        raise
    release_claim(brain_id, task_id)
    return None


def dispatch_snapshot(brain_id, task_id, cached, snapshot, context=None, extra=None):
    snapshot = _claim_live_snapshot(brain_id, task_id, cached, snapshot)
    if not snapshot:
        return False
    extra = _merged_extra(cached, snapshot, extra)
    payload = {
        "data": {"data_type": "text", "text_data": joined_text(snapshot.pages, snapshot.total)},
        "brain_id": brain_id,
    }
    group_key = cached.get("group_tracking")
    if group_key:
        payload["group_tracking"] = group_key
    error = None
    if extra.get("ocr_partial"):
        error = _ocr_partial_error(
            len(snapshot.pages), len(snapshot.failed), snapshot.total,
        )
    try:
        _enqueue_text(brain_id, task_id, payload, context)
    except Exception:
        release_claim(brain_id, task_id)
        raise
    updated = _write_dispatch_status(brain_id, task_id, cached, extra, error)
    clear_page_text(brain_id, task_id, snapshot)
    return updated if isinstance(updated, dict) else True


def _mark_failed(brain_id, task_id, extra, error):
    from src.workers.tasks.ingestion import set_ingestion_task_status

    return set_ingestion_task_status(
        task_id,
        brain_id,
        "failed",
        stage="failed",
        error=str(error)[:2000],
        extra=public_task_extra(extra),
    )


def _is_staged_ocr_run(brain_id, task_id):
    try:
        return is_remote_ocr_run(brain_id, task_id)
    except Exception:
        return False


def _dispatch_expired_pages(brain_id, task_id, cached, snapshot):
    extra = task_extra(cached, snapshot, closing=True)
    extra["ocr_partial"] = True
    try:
        result = dispatch_snapshot(brain_id, task_id, cached, snapshot, extra=extra)
    except Exception:
        return cached
    return result if isinstance(result, dict) else cached


def settle_expired(brain_id, task_id, cached):
    cached = dict(cached or {})
    deadline = cached.get("remote_ocr_deadline")
    if (
        cached.get("status") in TERMINAL_TASK_STATUSES
        or _is_staged_ocr_run(brain_id, task_id)
        or deadline is None
        or deadline > time()
    ):
        return None
    try:
        snapshot = load_pages(brain_id, task_id, cached)
    except SnapshotUnavailable:
        return cached
    if already_dispatched(brain_id, task_id, cached):
        return cached
    if snapshot.pages:
        return _dispatch_expired_pages(brain_id, task_id, cached, snapshot)
    if snapshot.failed:
        extra = task_extra(cached, snapshot)
        return _mark_failed(
            brain_id, task_id, extra, _ocr_partial_error(0, len(snapshot.failed), snapshot.total),
        )
    return None

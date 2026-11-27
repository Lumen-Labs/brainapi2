from __future__ import annotations

import threading
import time
from typing import Any
from uuid import uuid4

from src.core.saving.validity import format_instant, parse_instant, utc_now

SOURCE_TYPES = ("zendesk_guide",)
SYNC_MODES = ("webhook", "poll")
SOURCES_BUCKET = "sources"
INDEX_BUCKET = "index"
HEADS_BUCKET = "heads"
SEEN_BUCKET = "seen"
DEAD_BUCKET = "dead"
INDEX_BRAIN = "system"
LOCK_TTL_SECONDS = 120


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def public_source(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "name": record.get("name"),
        "brain_id": record.get("brain_id"),
        "source_type": record.get("source_type"),
        "sync_mode": record.get("sync_mode"),
        "poll_interval_seconds": record.get("poll_interval_seconds"),
        "base_url": record.get("base_url") or None,
        "locale": record.get("locale") or None,
        "webhook_secret_env": record.get("webhook_secret_env") or None,
        "api_token_env": record.get("api_token_env") or None,
        "skip_enrichment": bool(record.get("skip_enrichment")),
        "last_synced": record.get("last_synced"),
        "pending_delete_count": int(record.get("pending_delete_count") or 0),
        "pending_delete_first_seen": record.get("pending_delete_first_seen"),
        "pending_delete_reason": record.get("pending_delete_reason"),
        "listing_truncated": bool(record.get("listing_truncated")),
        "listing_pages_fetched": record.get("listing_pages_fetched"),
        "approve_pending_deletes": bool(record.get("approve_pending_deletes")),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
    }


class MemoryDurable:
    def __init__(self):
        self._data: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def put(self, brain_id: str, bucket: str, item_id: str, document: dict[str, Any]) -> None:
        with self._lock:
            slot = self._data.setdefault(f"{brain_id}:{bucket}", {})
            slot[item_id] = dict(document)

    def get(self, brain_id: str, bucket: str, item_id: str) -> dict[str, Any] | None:
        with self._lock:
            slot = self._data.get(f"{brain_id}:{bucket}") or {}
            value = slot.get(item_id)
            return dict(value) if isinstance(value, dict) else None

    def list(self, brain_id: str, bucket: str) -> dict[str, dict[str, Any]]:
        with self._lock:
            slot = self._data.get(f"{brain_id}:{bucket}") or {}
            return {
                key: dict(value)
                for key, value in slot.items()
                if isinstance(value, dict)
            }

    def delete(self, brain_id: str, bucket: str, item_id: str) -> bool:
        with self._lock:
            slot = self._data.get(f"{brain_id}:{bucket}")
            if not slot or item_id not in slot:
                return False
            slot.pop(item_id, None)
            return True


class AdapterDurable:
    def __init__(self, adapter: Any):
        self.adapter = adapter

    def put(self, brain_id: str, bucket: str, item_id: str, document: dict[str, Any]) -> None:
        self.adapter.source_sync_put(brain_id, bucket, item_id, document)

    def get(self, brain_id: str, bucket: str, item_id: str) -> dict[str, Any] | None:
        value = self.adapter.source_sync_get(brain_id, bucket, item_id)
        return dict(value) if isinstance(value, dict) else None

    def list(self, brain_id: str, bucket: str) -> dict[str, dict[str, Any]]:
        values = self.adapter.source_sync_list(brain_id, bucket) or {}
        return {
            str(key): dict(value)
            for key, value in values.items()
            if isinstance(value, dict)
        }

    def delete(self, brain_id: str, bucket: str, item_id: str) -> bool:
        return bool(self.adapter.source_sync_delete(brain_id, bucket, item_id))


class SourceSyncStore:
    def __init__(self, cache: Any | None = None, durable: Any | None = None):
        self.cache = cache
        self.durable = durable or MemoryDurable()
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def create(self, brain_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        source_type = _text(payload.get("source_type"))
        if source_type not in SOURCE_TYPES:
            raise ValueError("source_type must be zendesk_guide")
        sync_mode = _text(payload.get("sync_mode") or "webhook")
        if sync_mode not in SYNC_MODES:
            raise ValueError("sync_mode must be webhook or poll")
        name = _text(payload.get("name"))
        if not name:
            raise ValueError("name is required")
        now = format_instant(utc_now())
        record = {
            "id": _text(payload.get("id")) or uuid4().hex,
            "name": name,
            "brain_id": brain_id,
            "source_type": source_type,
            "sync_mode": sync_mode,
            "poll_interval_seconds": _positive_int(
                payload.get("poll_interval_seconds"), 3600
            ),
            "base_url": _text(payload.get("base_url")),
            "locale": _text(payload.get("locale")),
            "webhook_secret_env": _text(payload.get("webhook_secret_env")),
            "api_token_env": _text(payload.get("api_token_env")),
            "api_email": _text(payload.get("api_email")),
            "skip_enrichment": bool(payload.get("skip_enrichment")),
            "last_synced": None,
            "pending_delete_count": 0,
            "pending_delete_first_seen": None,
            "pending_delete_reason": None,
            "listing_truncated": False,
            "listing_pages_fetched": None,
            "approve_pending_deletes": False,
            "created_at": now,
            "updated_at": now,
        }
        self._put(brain_id, record["id"], record)
        self._index_put(record["id"], brain_id)
        return record

    def get(self, brain_id: str, source_id: str) -> dict[str, Any] | None:
        record = self.durable.get(brain_id, SOURCES_BUCKET, source_id)
        return record if isinstance(record, dict) else None

    def get_any(self, source_id: str) -> dict[str, Any] | None:
        brain_id = self._index_brain(source_id)
        if not brain_id:
            return None
        return self.get(brain_id, source_id)

    def list(self, brain_id: str) -> list[dict[str, Any]]:
        records = [
            item
            for item in self.durable.list(brain_id, SOURCES_BUCKET).values()
            if isinstance(item, dict)
        ]
        records.sort(key=lambda item: str(item.get("name") or item.get("id") or ""))
        return records

    def list_indexed(self) -> list[dict[str, Any]]:
        found = []
        for source_id in self.durable.list(INDEX_BRAIN, INDEX_BUCKET):
            record = self.get_any(source_id)
            if record is not None:
                found.append(record)
        return found

    def delete(self, brain_id: str, source_id: str) -> bool:
        existing = self.get(brain_id, source_id)
        if existing is None:
            return False
        self.durable.delete(brain_id, SOURCES_BUCKET, source_id)
        self.durable.delete(INDEX_BRAIN, INDEX_BUCKET, source_id)
        return True

    def mark_synced(self, brain_id: str, source_id: str, when: str | None = None) -> dict[str, Any] | None:
        record = self.get(brain_id, source_id)
        if record is None:
            return None
        stamp = format_instant(parse_instant(when) or utc_now())
        record["last_synced"] = stamp
        record["updated_at"] = stamp
        self._put(brain_id, source_id, record)
        return record

    def due_for_poll(self, record: dict[str, Any], now: float | None = None) -> bool:
        mode = _text(record.get("sync_mode"))
        if mode not in SYNC_MODES:
            return False
        if mode == "webhook" and not self.can_catchup(record):
            return False
        last = parse_instant(record.get("last_synced"))
        interval = _positive_int(record.get("poll_interval_seconds"), 3600)
        current = now if now is not None else time.time()
        if last is None:
            return True
        return (last.timestamp() + interval) <= current

    def can_catchup(self, record: dict[str, Any]) -> bool:
        return bool(_text(record.get("api_token_env")) and _text(record.get("base_url")))

    def document_head(self, brain_id: str, source_id: str, document_id: str) -> dict[str, Any] | None:
        return self.durable.get(brain_id, HEADS_BUCKET, _head_id(source_id, document_id))

    def save_document_head(
        self,
        brain_id: str,
        source_id: str,
        document_id: str,
        payload: dict[str, Any],
    ) -> None:
        self.durable.put(
            brain_id,
            HEADS_BUCKET,
            _head_id(source_id, document_id),
            {
                "source_id": source_id,
                "document_id": document_id,
                **payload,
            },
        )

    def list_open_document_ids(self, brain_id: str, source_id: str) -> set[str]:
        open_ids = set(self.get_seen_ids(brain_id, source_id))
        for item in self.durable.list(brain_id, HEADS_BUCKET).values():
            if str(item.get("source_id") or "") != source_id:
                continue
            if str(item.get("action") or "") == "delete":
                continue
            document_id = _text(item.get("document_id"))
            if document_id:
                open_ids.add(document_id)
        return open_ids

    def get_seen_ids(self, brain_id: str, source_id: str) -> set[str]:
        record = self.durable.get(brain_id, SEEN_BUCKET, source_id) or {}
        raw = record.get("document_ids") or []
        return {_text(item) for item in raw if _text(item)}

    def set_seen_ids(self, brain_id: str, source_id: str, document_ids: set[str] | list[str]) -> None:
        self.durable.put(
            brain_id,
            SEEN_BUCKET,
            source_id,
            {"document_ids": sorted({_text(item) for item in document_ids if _text(item)})},
        )

    def acquire_document_lock(self, brain_id: str, source_id: str, document_id: str) -> bool:
        key = f"source_sync:lock:{source_id}:{document_id}"
        setter = getattr(self.cache, "set_nx", None) if self.cache is not None else None
        if callable(setter):
            return bool(setter(key, "1", brain_id, LOCK_TTL_SECONDS))
        with self._locks_guard:
            lock = self._locks.setdefault(f"{brain_id}:{key}", threading.Lock())
        return lock.acquire(blocking=False)

    def release_document_lock(self, brain_id: str, source_id: str, document_id: str) -> None:
        key = f"source_sync:lock:{source_id}:{document_id}"
        deleter = getattr(self.cache, "delete", None) if self.cache is not None else None
        if callable(deleter) and getattr(self.cache, "set_nx", None):
            deleter(key, brain_id)
            return
        with self._locks_guard:
            lock = self._locks.get(f"{brain_id}:{key}")
        if lock is not None and lock.locked():
            lock.release()

    def record_blocked_deletes(
        self,
        brain_id: str,
        source_id: str,
        count: int,
        reason: str,
    ) -> bool:
        record = self.get(brain_id, source_id)
        if record is None:
            return False
        stamp = format_instant(utc_now())
        first_seen = record.get("pending_delete_first_seen")
        newly = not first_seen
        record["pending_delete_count"] = int(count)
        record["pending_delete_reason"] = reason
        if newly:
            record["pending_delete_first_seen"] = stamp
        record["updated_at"] = stamp
        self._put(brain_id, source_id, record)
        return newly

    def clear_blocked_deletes(self, brain_id: str, source_id: str) -> None:
        record = self.get(brain_id, source_id)
        if record is None:
            return
        if not (
            record.get("pending_delete_count")
            or record.get("pending_delete_first_seen")
            or record.get("pending_delete_reason")
        ):
            return
        record["pending_delete_count"] = 0
        record["pending_delete_first_seen"] = None
        record["pending_delete_reason"] = None
        record["updated_at"] = format_instant(utc_now())
        self._put(brain_id, source_id, record)

    def arm_delete_approval(self, brain_id: str, source_id: str) -> dict[str, Any] | None:
        record = self.get(brain_id, source_id)
        if record is None:
            return None
        record["approve_pending_deletes"] = True
        record["updated_at"] = format_instant(utc_now())
        self._put(brain_id, source_id, record)
        return record

    def consume_delete_approval(self, brain_id: str, source_id: str) -> bool:
        record = self.get(brain_id, source_id)
        if record is None or not record.get("approve_pending_deletes"):
            return False
        record["approve_pending_deletes"] = False
        record["updated_at"] = format_instant(utc_now())
        self._put(brain_id, source_id, record)
        return True

    def set_listing_status(
        self,
        brain_id: str,
        source_id: str,
        *,
        truncated: bool,
        pages_fetched: int | None,
    ) -> None:
        record = self.get(brain_id, source_id)
        if record is None:
            return
        record["listing_truncated"] = bool(truncated)
        record["listing_pages_fetched"] = pages_fetched
        record["updated_at"] = format_instant(utc_now())
        self._put(brain_id, source_id, record)

    def record_dead_letter(self, brain_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        item = {
            "id": uuid4().hex,
            "brain_id": brain_id,
            "recorded_at": format_instant(utc_now()),
            **payload,
        }
        self.durable.put(brain_id, DEAD_BUCKET, item["id"], item)
        return item

    def list_dead_letters(self, brain_id: str) -> list[dict[str, Any]]:
        items = [
            item
            for item in self.durable.list(brain_id, DEAD_BUCKET).values()
            if isinstance(item, dict)
        ]
        items.sort(key=lambda item: str(item.get("recorded_at") or ""), reverse=True)
        return items

    def _put(self, brain_id: str, source_id: str, record: dict[str, Any]) -> None:
        self.durable.put(brain_id, SOURCES_BUCKET, source_id, record)

    def _index_put(self, source_id: str, brain_id: str) -> None:
        self.durable.put(INDEX_BRAIN, INDEX_BUCKET, source_id, {"brain_id": brain_id})

    def _index_brain(self, source_id: str) -> str:
        value = self.durable.get(INDEX_BRAIN, INDEX_BUCKET, source_id)
        if isinstance(value, dict):
            return _text(value.get("brain_id"))
        return _text(value)


def _head_id(source_id: str, document_id: str) -> str:
    return f"{source_id}:{document_id}"


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def default_durable() -> Any:
    try:
        from src.core.instances import data_adapter

        if getattr(data_adapter, "data", None) is not None and callable(
            getattr(data_adapter, "source_sync_put", None)
        ):
            return AdapterDurable(data_adapter)
    except Exception:
        pass
    return MemoryDurable()


def default_store() -> SourceSyncStore:
    cache = None
    try:
        from src.core.instances import cache_adapter

        cache = cache_adapter
    except Exception:
        cache = None
    return SourceSyncStore(cache=cache, durable=default_durable())

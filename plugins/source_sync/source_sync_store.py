from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4

from src.core.saving.validity import format_instant, parse_instant, utc_now

SOURCE_TYPES = ("zendesk_guide",)
SYNC_MODES = ("webhook", "poll")
SOURCES_HASH = "source_sync:sources"
INDEX_HASH = "source_sync:index"
REVISIONS_HASH = "source_sync:revisions"
DEAD_HASH = "source_sync:dead"
INDEX_BRAIN = "system"


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
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
    }


class SourceSyncStore:
    def __init__(self, cache: Any | None = None):
        self.cache = cache
        self._memory: dict[str, dict[str, str]] = {}

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
            "created_at": now,
            "updated_at": now,
        }
        self._put(brain_id, record["id"], record)
        self._index_put(record["id"], brain_id)
        return record

    def get(self, brain_id: str, source_id: str) -> dict[str, Any] | None:
        raw = self._hget(SOURCES_HASH, source_id, brain_id)
        if not raw:
            return None
        record = json.loads(raw)
        if not isinstance(record, dict):
            return None
        return record

    def get_any(self, source_id: str) -> dict[str, Any] | None:
        brain_id = self._hget(INDEX_HASH, source_id, INDEX_BRAIN)
        if not brain_id:
            return None
        return self.get(brain_id, source_id)

    def list(self, brain_id: str) -> list[dict[str, Any]]:
        values = self._hgetall(SOURCES_HASH, brain_id)
        records = []
        for raw in values.values():
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
        records.sort(key=lambda item: str(item.get("name") or item.get("id") or ""))
        return records

    def list_indexed(self) -> list[dict[str, Any]]:
        found = []
        for source_id, brain_id in self._hgetall(INDEX_HASH, INDEX_BRAIN).items():
            record = self.get(brain_id, source_id)
            if record is not None:
                found.append(record)
        return found

    def delete(self, brain_id: str, source_id: str) -> bool:
        existing = self.get(brain_id, source_id)
        if existing is None:
            return False
        self._hdel(SOURCES_HASH, source_id, brain_id)
        self._hdel(INDEX_HASH, source_id, INDEX_BRAIN)
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
        if _text(record.get("sync_mode")) != "poll":
            return False
        last = parse_instant(record.get("last_synced"))
        interval = _positive_int(record.get("poll_interval_seconds"), 3600)
        current = now if now is not None else time.time()
        if last is None:
            return True
        return (last.timestamp() + interval) <= current

    def claim_revision(self, key: str, brain_id: str) -> bool:
        return self._hsetnx(REVISIONS_HASH, key, "1", brain_id)

    def release_revision(self, key: str, brain_id: str) -> None:
        self._hdel(REVISIONS_HASH, key, brain_id)

    def record_dead_letter(self, brain_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        item = {
            "id": uuid4().hex,
            "brain_id": brain_id,
            "recorded_at": format_instant(utc_now()),
            **payload,
        }
        self._hset(DEAD_HASH, item["id"], json.dumps(item), brain_id)
        return item

    def list_dead_letters(self, brain_id: str) -> list[dict[str, Any]]:
        items = []
        for raw in self._hgetall(DEAD_HASH, brain_id).values():
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                items.append(item)
        items.sort(key=lambda item: str(item.get("recorded_at") or ""), reverse=True)
        return items

    def _put(self, brain_id: str, source_id: str, record: dict[str, Any]) -> None:
        self._hset(SOURCES_HASH, source_id, json.dumps(record), brain_id)

    def _index_put(self, source_id: str, brain_id: str) -> None:
        self._hset(INDEX_HASH, source_id, brain_id, INDEX_BRAIN)

    def _bucket(self, brain_id: str, key: str) -> dict[str, str]:
        slot = f"{brain_id}:{key}"
        return self._memory.setdefault(slot, {})

    def _hset(self, key: str, field: str, value: str, brain_id: str) -> int:
        if self.cache is not None and hasattr(self.cache, "hset"):
            return int(self.cache.hset(key, field, value, brain_id=brain_id) or 0)
        bucket = self._bucket(brain_id, key)
        existed = field in bucket
        bucket[field] = value
        return 0 if existed else 1

    def _hsetnx(self, key: str, field: str, value: str, brain_id: str) -> bool:
        if self.cache is not None and hasattr(self.cache, "hsetnx"):
            return bool(self.cache.hsetnx(key, field, value, brain_id=brain_id))
        bucket = self._bucket(brain_id, key)
        if field in bucket:
            return False
        bucket[field] = value
        return True

    def _hget(self, key: str, field: str, brain_id: str) -> str | None:
        if self.cache is not None and hasattr(self.cache, "hgetall"):
            values = self.cache.hgetall(key, brain_id=brain_id) or {}
            value = values.get(field)
            return str(value) if value is not None else None
        return self._bucket(brain_id, key).get(field)

    def _hgetall(self, key: str, brain_id: str) -> dict[str, str]:
        if self.cache is not None and hasattr(self.cache, "hgetall"):
            return dict(self.cache.hgetall(key, brain_id=brain_id) or {})
        return dict(self._bucket(brain_id, key))

    def _hdel(self, key: str, field: str, brain_id: str) -> int:
        if self.cache is not None and hasattr(self.cache, "hdel"):
            return int(self.cache.hdel(key, field, brain_id=brain_id) or 0)
        bucket = self._bucket(brain_id, key)
        return 1 if bucket.pop(field, None) is not None else 0


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def default_store() -> SourceSyncStore:
    try:
        from src.core.instances import cache_adapter

        return SourceSyncStore(cache_adapter)
    except Exception:
        return SourceSyncStore()

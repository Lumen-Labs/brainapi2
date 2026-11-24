from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from src.core.saving.conflicts import normalize_predicate

_LOCAL_LOCKS: dict[str, threading.RLock] = {}
_LOCAL_GUARD = threading.Lock()
_HELD = threading.local()


def _lock_key(brain_id: str, subject_uuid: str, predicate: str) -> str:
    return f"{brain_id}:{subject_uuid}:{normalize_predicate(predicate)}"


def _local_lock(key: str) -> threading.RLock:
    with _LOCAL_GUARD:
        lock = _LOCAL_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCAL_LOCKS[key] = lock
        return lock


def _held_keys() -> set[str]:
    keys = getattr(_HELD, "keys", None)
    if keys is None:
        keys = set()
        _HELD.keys = keys
    return keys


def _redis_lock(key: str):
    try:
        from src.lib.redis.client import _redis_client

        client = getattr(_redis_client, "client", None)
        if client is None or not hasattr(client, "lock"):
            return None
        lock = client.lock(
            f"intent:retire:{key}",
            timeout=30,
            blocking_timeout=10,
        )
        if not lock.acquire(blocking=True):
            return None
        return lock
    except Exception:
        return None


@contextmanager
def subject_write_lock(
    brain_id: str,
    subject_uuid: str,
    predicate: str = "",
) -> Iterator[None]:
    """Serialize remember/correct retirement for one brain+subject+predicate.

    Uses Redis when available so workers share the lock, plus an in-process
    RLock so tests and single-process fallbacks stay ordered. Re-entrant for
    the same thread (correct → remember).
    """
    if not brain_id or not subject_uuid:
        yield
        return
    key = _lock_key(brain_id, subject_uuid, predicate)
    held = _held_keys()
    if key in held:
        yield
        return
    redis_lock = _redis_lock(key)
    local = _local_lock(key)
    local.acquire()
    held.add(key)
    try:
        yield
    finally:
        held.discard(key)
        try:
            local.release()
        except Exception:
            pass
        if redis_lock is not None:
            try:
                redis_lock.release()
            except Exception:
                pass

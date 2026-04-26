from __future__ import annotations

import logging
import os
import threading
import time
from queue import Empty

_PATCHED = False
_LOGGER = logging.getLogger(__name__)
_UNBLOCKED_WINDOW_SECONDS = float(os.getenv("CELERY_UNBLOCKED_WINDOW_SECONDS", "60"))
_UNBLOCKED_MAX_RETRIES = int(os.getenv("CELERY_UNBLOCKED_MAX_RETRIES", "20"))
_UNBLOCKED_STATE_LOCK = threading.Lock()
_UNBLOCKED_WINDOW_STARTED_AT = 0.0
_UNBLOCKED_EVENT_COUNT = 0


def _record_unblocked_event() -> tuple[int, bool]:
    global _UNBLOCKED_WINDOW_STARTED_AT, _UNBLOCKED_EVENT_COUNT

    now = time.monotonic()
    with _UNBLOCKED_STATE_LOCK:
        if (
            _UNBLOCKED_WINDOW_STARTED_AT == 0.0
            or now - _UNBLOCKED_WINDOW_STARTED_AT > _UNBLOCKED_WINDOW_SECONDS
        ):
            _UNBLOCKED_WINDOW_STARTED_AT = now
            _UNBLOCKED_EVENT_COUNT = 0

        _UNBLOCKED_EVENT_COUNT += 1
        current_count = _UNBLOCKED_EVENT_COUNT
        breached = current_count >= _UNBLOCKED_MAX_RETRIES
        if breached:
            _UNBLOCKED_WINDOW_STARTED_AT = 0.0
            _UNBLOCKED_EVENT_COUNT = 0
        return current_count, breached


def apply_kombu_redis_unblocked_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from kombu.transport.redis import Channel

    _orig_brpop_read = Channel._brpop_read

    def _brpop_read(self, **options):
        try:
            return _orig_brpop_read(self, **options)
        except self.ResponseError as exc:
            if str(exc).startswith("UNBLOCKED"):
                try:
                    self.client.connection.disconnect()
                except Exception:
                    _LOGGER.exception("Celery Redis patch failed to disconnect after UNBLOCKED")
                event_count, breached = _record_unblocked_event()
                if event_count == 1:
                    _LOGGER.warning(
                        "Celery Redis consumer received UNBLOCKED; treating as transient and retrying"
                    )
                if breached:
                    _LOGGER.error(
                        "Celery Redis consumer exceeded UNBLOCKED threshold; re-raising to force restart"
                    )
                    raise
                raise Empty() from exc
            raise

    Channel._brpop_read = _brpop_read
    _PATCHED = True

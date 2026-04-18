from __future__ import annotations

from queue import Empty

_PATCHED = False


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
                self.client.connection.disconnect()
                raise Empty() from exc
            raise

    Channel._brpop_read = _brpop_read
    _PATCHED = True

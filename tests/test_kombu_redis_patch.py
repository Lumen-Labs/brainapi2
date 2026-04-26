from queue import Empty
from unittest.mock import MagicMock

import pytest
from redis.exceptions import ResponseError

from src.workers.kombu_redis_patch import apply_kombu_redis_unblocked_patch


@pytest.fixture(autouse=True)
def reset_kombu_channel_and_patch_flag():
    from kombu.transport.redis import Channel

    import src.workers.kombu_redis_patch as patch_mod

    saved = Channel._brpop_read
    saved_monotonic = patch_mod.time.monotonic
    patch_mod._PATCHED = False
    patch_mod._UNBLOCKED_WINDOW_SECONDS = 60
    patch_mod._UNBLOCKED_MAX_RETRIES = 20
    patch_mod._UNBLOCKED_WINDOW_STARTED_AT = 0.0
    patch_mod._UNBLOCKED_EVENT_COUNT = 0
    yield
    Channel._brpop_read = saved
    patch_mod.time.monotonic = saved_monotonic
    patch_mod._PATCHED = False
    patch_mod._UNBLOCKED_WINDOW_SECONDS = 60
    patch_mod._UNBLOCKED_MAX_RETRIES = 20
    patch_mod._UNBLOCKED_WINDOW_STARTED_AT = 0.0
    patch_mod._UNBLOCKED_EVENT_COUNT = 0


def test_unblocked_response_error_becomes_empty():
    from kombu.transport.redis import Channel

    def fake_brpop(self, **options):
        raise ResponseError(
            "UNBLOCKED force unblock from blocking operation, "
            "instance state changed (master -> replica?)"
        )

    Channel._brpop_read = fake_brpop
    apply_kombu_redis_unblocked_patch()

    ch = MagicMock()
    ch.client.connection.disconnect = MagicMock()
    ch.connection_errors = ()
    ch.ResponseError = ResponseError

    with pytest.raises(Empty):
        Channel._brpop_read(ch)
    ch.client.connection.disconnect.assert_called_once()


def test_other_response_error_reraised():
    from kombu.transport.redis import Channel

    def fake_brpop(self, **options):
        raise ResponseError("WRONGTYPE")

    Channel._brpop_read = fake_brpop
    apply_kombu_redis_unblocked_patch()

    ch = MagicMock()
    ch.client.connection.disconnect = MagicMock()
    ch.connection_errors = ()
    ch.ResponseError = ResponseError

    with pytest.raises(ResponseError, match="WRONGTYPE"):
        Channel._brpop_read(ch)
    ch.client.connection.disconnect.assert_not_called()


def test_unblocked_repeats_below_threshold_keep_returning_empty():
    from kombu.transport.redis import Channel

    import src.workers.kombu_redis_patch as patch_mod

    def fake_brpop(self, **options):
        raise ResponseError(
            "UNBLOCKED force unblock from blocking operation, "
            "instance state changed (master -> replica?)"
        )

    Channel._brpop_read = fake_brpop
    patch_mod._UNBLOCKED_MAX_RETRIES = 3
    apply_kombu_redis_unblocked_patch()

    ch = MagicMock()
    ch.client.connection.disconnect = MagicMock()
    ch.connection_errors = ()
    ch.ResponseError = ResponseError

    with pytest.raises(Empty):
        Channel._brpop_read(ch)
    with pytest.raises(Empty):
        Channel._brpop_read(ch)
    assert ch.client.connection.disconnect.call_count == 2


def test_unblocked_threshold_breaches_and_reraises_response_error():
    from kombu.transport.redis import Channel

    import src.workers.kombu_redis_patch as patch_mod

    def fake_brpop(self, **options):
        raise ResponseError(
            "UNBLOCKED force unblock from blocking operation, "
            "instance state changed (master -> replica?)"
        )

    Channel._brpop_read = fake_brpop
    patch_mod._UNBLOCKED_MAX_RETRIES = 2
    apply_kombu_redis_unblocked_patch()

    ch = MagicMock()
    ch.client.connection.disconnect = MagicMock()
    ch.connection_errors = ()
    ch.ResponseError = ResponseError

    with pytest.raises(Empty):
        Channel._brpop_read(ch)
    with pytest.raises(ResponseError, match="UNBLOCKED"):
        Channel._brpop_read(ch)
    assert ch.client.connection.disconnect.call_count == 2


def test_unblocked_counter_resets_after_window_expires():
    from kombu.transport.redis import Channel

    import src.workers.kombu_redis_patch as patch_mod

    class MonotonicSequence:
        def __init__(self, values):
            self._it = iter(values)

        def __call__(self):
            return next(self._it)

    def fake_brpop(self, **options):
        raise ResponseError(
            "UNBLOCKED force unblock from blocking operation, "
            "instance state changed (master -> replica?)"
        )

    Channel._brpop_read = fake_brpop
    patch_mod._UNBLOCKED_MAX_RETRIES = 2
    patch_mod._UNBLOCKED_WINDOW_SECONDS = 5
    patch_mod.time.monotonic = MonotonicSequence([1.0, 10.0, 10.1])
    apply_kombu_redis_unblocked_patch()

    ch = MagicMock()
    ch.client.connection.disconnect = MagicMock()
    ch.connection_errors = ()
    ch.ResponseError = ResponseError

    with pytest.raises(Empty):
        Channel._brpop_read(ch)
    with pytest.raises(Empty):
        Channel._brpop_read(ch)
    with pytest.raises(ResponseError, match="UNBLOCKED"):
        Channel._brpop_read(ch)

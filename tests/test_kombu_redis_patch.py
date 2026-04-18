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
    patch_mod._PATCHED = False
    yield
    Channel._brpop_read = saved
    patch_mod._PATCHED = False


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

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch


def test_lru_does_not_close_a_pool_with_a_borrowed_connection():
    from src.lib.postgresql import _provisioning as provisioning

    registry = provisioning._PoolRegistry()
    first, second = MagicMock(), MagicMock()

    with (
        patch.object(provisioning, "_MAX_OPEN_POOLS", 1),
        patch.object(provisioning.config, "postgresql", MagicMock()),
        patch.object(
            provisioning.psycopg2.pool,
            "ThreadedConnectionPool",
            side_effect=[first, second],
        ),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        def borrow_other_brain():
            with registry.borrow("brain-b"):
                first.closeall.assert_not_called()
                second.closeall.assert_not_called()

        with registry.borrow("brain-a") as connection:
            executor.submit(borrow_other_brain).result(timeout=5)
            first.closeall.assert_not_called()
            second.closeall.assert_called_once()
        first.putconn.assert_called_once_with(connection)

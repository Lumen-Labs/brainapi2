from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.lib.postgresql._naming import brain_db_name
from src.lib.postgresql._provisioning import drop_brain_database


def test_drop_brain_database_terminates_sessions_then_drops():
    cursor = MagicMock()
    cursor.fetchone.return_value = (1,)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    with (
        patch("src.lib.postgresql._provisioning.psycopg2.connect", return_value=conn) as connect,
        patch("src.lib.postgresql._provisioning._registry.close") as close_pool,
    ):
        assert drop_brain_database("demo") is True

    close_pool.assert_called_once_with(brain_db_name("demo"))
    connect.assert_called_once()
    assert conn.autocommit is True
    statements = [call.args[0] for call in cursor.execute.call_args_list]
    assert any("pg_database" in str(statement) for statement in statements)
    assert any("pg_terminate_backend" in str(statement) for statement in statements)
    assert any("DROP DATABASE" in str(statement) for statement in statements)
    conn.close.assert_called_once()


def test_drop_brain_database_is_idempotent_when_missing():
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    with patch("src.lib.postgresql._provisioning.psycopg2.connect", return_value=conn):
        assert drop_brain_database("demo") is False

    statements = [call.args[0] for call in cursor.execute.call_args_list]
    assert not any("DROP DATABASE" in str(statement) for statement in statements)

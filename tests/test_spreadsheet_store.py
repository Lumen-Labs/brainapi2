from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from src.core.spreadsheets.query import WorkbookQuery
from src.services.api.routes.spreadsheets import save_workbook, workbook_caps


class ScriptedCursor:
    def __init__(self, fetches):
        self.fetches = list(fetches)
        self.statements = []

    def execute(self, sql, args=None):
        self.statements.append((sql, args))

    def fetchone(self):
        return self.fetches.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def cursor(self):
        return self._cursor


@contextmanager
def fake_connect(cursor):
    yield FakeConn(cursor)


def test_schema_backfills_legacy_cell_counts():
    from src.services.api.routes.spreadsheets import SCHEMA

    assert "cell_count = totals.cells" in SCHEMA
    assert "book.cell_count = 0" in SCHEMA


def test_identical_upload_returns_existing_workbook_id():
    existing = ("11111111-1111-1111-1111-111111111111", "grid.csv", [{"name": "CSV"}])
    cursor = ScriptedCursor([existing])
    with patch("src.services.api.routes.spreadsheets.remember_workbook"):
        result = save_workbook(
            fake_connect(cursor), "grid.csv", b"A,B\n1,2\n", "new-id", "demo",
        )
    assert result["id"] == existing[0]
    assert result["deduplicated"] is True
    assert not any("INSERT INTO spreadsheets" in sql for sql, _ in cursor.statements)


def test_workbook_count_cap_is_checked_before_insert(monkeypatch):
    monkeypatch.setenv("SPREADSHEET_MAX_WORKBOOKS_PER_BRAIN", "500")
    cursor = ScriptedCursor([None, (500, 0, 0)])
    with pytest.raises(HTTPException) as exc:
        save_workbook(fake_connect(cursor), "grid.csv", b"A,B\n1,2\n", "new-id", "demo")
    assert exc.value.status_code == 409
    assert "count" in str(exc.value.detail).lower()
    assert not any("INSERT INTO spreadsheets" in sql for sql, _ in cursor.statements)


def test_workbook_caps_read_env(monkeypatch):
    monkeypatch.setenv("SPREADSHEET_MAX_WORKBOOKS_PER_BRAIN", "500")
    monkeypatch.setenv("SPREADSHEET_MAX_CELLS_PER_BRAIN", "1000")
    monkeypatch.setenv("SPREADSHEET_MAX_BYTES_PER_BRAIN", "4096")
    assert workbook_caps() == (500, 1000, 4096)


def test_group_by_is_accepted_on_the_query_model():
    query = WorkbookQuery(
        workbook="00000000-0000-0000-0000-000000000001",
        group_by=["When"],
        aggregate={"op": "count"},
    )
    assert query.group_by == ["When"]

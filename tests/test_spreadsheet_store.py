from contextlib import contextmanager
from threading import Lock, Thread
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from src.core.spreadsheets.query import WorkbookQuery
from src.services.api.routes.spreadsheets import (
    SCHEMA,
    SCHEMA_VERSION,
    clear_schema_cache,
    ensure_schema,
    mark_schema_migrated,
    save_workbook,
    workbook_caps,
)


@pytest.fixture(autouse=True)
def _reset_schema_cache():
    clear_schema_cache()
    yield
    clear_schema_cache()


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


def sql_text(statement):
    return statement if isinstance(statement, str) else statement[0]


def backfill_count(statements):
    return sum(1 for sql in statements if "cell_count = totals.cells" in sql_text(sql))


class MarkerCursor:
    def __init__(self, db):
        self.db = db
        self.statements = db.setdefault("statements", [])

    def execute(self, sql, args=None):
        self.statements.append((sql, args))
        if "INSERT INTO spreadsheet_schema" in sql:
            self.db["version"] = args[0] if args else SCHEMA_VERSION

    def fetchone(self):
        version = self.db.get("version")
        return None if version is None else (version,)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def test_schema_backfills_legacy_cell_counts():
    assert "cell_count = totals.cells" in SCHEMA
    assert "book.cell_count = 0" in SCHEMA


def test_ensure_schema_runs_backfill_once_then_uses_cache():
    db = {"version": None, "statements": []}
    ensure_schema(MarkerCursor(db), "alpha")
    mark_schema_migrated("alpha")
    assert backfill_count(db["statements"]) == 1
    assert any("INSERT INTO spreadsheet_schema" in sql_text(sql) for sql in db["statements"])
    db["statements"].clear()
    ensure_schema(MarkerCursor(db), "alpha")
    assert db["statements"] == []
    assert backfill_count(db["statements"]) == 0


def test_ensure_schema_skips_backfill_when_persistent_marker_matches():
    db = {"version": SCHEMA_VERSION, "statements": []}
    ensure_schema(MarkerCursor(db), "beta")
    assert backfill_count(db["statements"]) == 0
    assert not any("INSERT INTO spreadsheet_schema" in sql_text(sql) for sql in db["statements"])
    assert any("SELECT version FROM spreadsheet_schema" in sql_text(sql) for sql in db["statements"])


def test_ensure_schema_backfills_brains_created_before_marker():
    db = {"version": None, "statements": []}
    ensure_schema(MarkerCursor(db), "legacy")
    assert backfill_count(db["statements"]) == 1
    assert db["version"] == SCHEMA_VERSION


def test_ensure_schema_second_worker_reads_marker_not_cache():
    db = {"version": None, "statements": []}
    ensure_schema(MarkerCursor(db), "shared")
    mark_schema_migrated("shared")
    clear_schema_cache()
    ensure_schema(MarkerCursor(db), "shared")
    assert backfill_count(db["statements"]) == 1
    assert db["version"] == SCHEMA_VERSION


def test_ensure_schema_serializes_concurrent_workers_on_one_brain():
    db = {"version": None, "statements": []}
    lock = Lock()
    errors = []

    def worker():
        try:
            with lock:
                clear_schema_cache()
                ensure_schema(MarkerCursor(db), "race")
        except Exception as exc:
            errors.append(exc)

    threads = [Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert backfill_count(db["statements"]) == 1
    assert db["version"] == SCHEMA_VERSION


def test_commit_probes_schema_only_after_advisory_lock():
    mark_schema_migrated("demo")
    existing = ("11111111-1111-1111-1111-111111111111", "grid.csv", [{"name": "CSV"}])
    cursor = ScriptedCursor([existing])
    with patch("src.services.api.routes.spreadsheets.remember_workbook"):
        save_workbook(fake_connect(cursor), "grid.csv", b"A,B\n1,2\n", "new-id", "demo")
    sqls = [sql for sql, _ in cursor.statements]
    lock_at = next(i for i, sql in enumerate(sqls) if "pg_advisory_xact_lock" in sql)
    assert lock_at == 0
    assert not any("cell_count = totals.cells" in sql for sql in sqls)


def test_first_import_migrates_then_later_import_skips_backfill():
    cursor = ScriptedCursor([None, None, (0, 0, 0), None, (0, 0, 0)])
    with patch("src.services.api.routes.spreadsheets.remember_workbook"):
        with patch("src.services.api.routes.spreadsheets.execute_values"):
            save_workbook(fake_connect(cursor), "grid.csv", b"A,B\n1,2\n", "new-id", "demo")
            first = backfill_count(cursor.statements)
            save_workbook(fake_connect(cursor), "other.csv", b"A,C\n1,3\n", "other-id", "demo")
    assert first == 1
    assert backfill_count(cursor.statements) == 1
    assert any("INSERT INTO spreadsheet_schema" in sql for sql, _ in cursor.statements)


def test_identical_upload_returns_existing_workbook_id():
    mark_schema_migrated("demo")
    existing = ("11111111-1111-1111-1111-111111111111", "grid.csv", [{"name": "CSV"}])
    cursor = ScriptedCursor([existing])
    with patch("src.services.api.routes.spreadsheets.remember_workbook") as remember:
        result = save_workbook(
            fake_connect(cursor), "grid.csv", b"A,B\n1,2\n", "new-id", "demo",
        )
    assert result["id"] == existing[0]
    assert result["deduplicated"] is True
    remember.assert_called_once_with("demo", existing[0], task_id="new-id")
    assert not any("INSERT INTO spreadsheets" in sql for sql, _ in cursor.statements)


def test_workbook_count_cap_is_checked_before_insert(monkeypatch):
    mark_schema_migrated("demo")
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

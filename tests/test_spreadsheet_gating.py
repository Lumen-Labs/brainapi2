import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.lib.postgresql._provisioning import PoolTimeout
from src.services.api.dependencies import get_brain_id
from src.services.api.routes.spreadsheets import (
    enabled,
    router,
    supports_file,
    try_import_file,
)


@pytest.fixture
def api():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_brain_id] = lambda: "demo"
    return TestClient(app)


def test_enabled_empty_flag(monkeypatch):
    monkeypatch.delenv("SPREADSHEET_EXPERIMENT_BRAINS", raising=False)
    assert enabled("demo") is False
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "")
    assert enabled("demo") is False
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", " , ")
    assert enabled("demo") is False


def test_enabled_specific_keys(monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "alpha, beta")
    assert enabled("alpha") is True
    assert enabled("beta") is True
    assert enabled("gamma") is False
    assert enabled("*") is False


def test_enabled_star_enables_all(monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "*")
    assert enabled("demo") is True
    assert enabled("other") is True


def test_supports_file_suffixes():
    assert supports_file("grid.xlsx") is True
    assert supports_file("grid.CSV") is True
    assert supports_file("notes.pdf") is False
    assert supports_file("") is False
    assert supports_file(None) is False


@pytest.mark.parametrize("flag", ["", "other"])
def test_try_import_file_falls_through_when_disabled(monkeypatch, flag):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", flag)
    upload = AsyncMock()
    upload.filename = "grid.xlsx"
    result = asyncio.run(try_import_file(upload, "demo", workbook_id="task-1"))
    assert result is None
    upload.read.assert_not_called()


def test_try_import_file_falls_through_without_postgres(monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "demo")
    upload = AsyncMock()
    upload.filename = "grid.xlsx"
    with patch("src.services.api.routes.spreadsheets.postgres_connection", return_value=None):
        result = asyncio.run(try_import_file(upload, "demo", workbook_id="task-1"))
    assert result is None


def test_retrieve_routes_404_when_flag_empty(api, monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "")
    listed = api.get("/retrieve/spreadsheets")
    queried = api.post(
        "/retrieve/spreadsheets/query",
        json={"workbook": "00000000-0000-0000-0000-000000000001"},
    )
    uploaded = api.post("/ingest/spreadsheets", files={"file": ("grid.csv", b"a,b\n1,2\n")})
    assert listed.status_code == 404
    assert queried.status_code == 404
    assert uploaded.status_code == 404


def test_retrieve_routes_404_for_ungated_brain(api, monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "other")
    assert api.get("/retrieve/spreadsheets").status_code == 404


def test_star_gate_reaches_postgres_lookup(api, monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "*")

    def missing(_brain_id):
        raise HTTPException(404, "Spreadsheet queries are unavailable for this brain.")

    with patch("src.services.api.routes.spreadsheets.postgres_connection", return_value=missing):
        response = api.get("/retrieve/spreadsheets")
    assert response.status_code == 404


def test_try_import_file_rejects_xls_when_gated(monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "demo")
    upload = AsyncMock()
    upload.filename = "grid.xls"
    with patch("src.services.api.routes.spreadsheets.postgres_connection", return_value=object()):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(try_import_file(upload, "demo", workbook_id="task-1"))
    assert exc.value.status_code == 422
    upload.read.assert_not_called()


def test_try_import_file_rejects_corrupt_xlsx_when_gated(monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "demo")
    upload = AsyncMock()
    upload.filename = "grid.xlsx"
    upload.read = AsyncMock(return_value=b"not-a-workbook")
    with patch("src.services.api.routes.spreadsheets.postgres_connection", return_value=object()):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(try_import_file(upload, "demo", workbook_id="task-1"))
    assert exc.value.status_code == 422
    assert "xlsx" in str(exc.value.detail).lower()


def test_query_pool_timeout_returns_503(api, monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "*")

    @contextmanager
    def busy(_brain_id):
        raise PoolTimeout("connection pool timeout")
        yield

    with patch("src.services.api.routes.spreadsheets.postgres_connection", return_value=busy):
        response = api.post(
            "/retrieve/spreadsheets/query",
            json={"workbook": "00000000-0000-0000-0000-000000000001"},
        )
    assert response.status_code == 503
    assert response.headers.get("retry-after") == "1"
    assert response.json()["detail"]


def test_list_accepts_pagination_query(api, monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "")
    response = api.get("/retrieve/spreadsheets", params={"offset": 0, "limit": 10})
    assert response.status_code == 404


def test_delete_workbook_returns_204(api, monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "*")

    class Cursor:
        def execute(self, sql, args=None):
            self.sql = sql

        def fetchone(self):
            return ("00000000-0000-0000-0000-000000000001",)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def cursor(self):
            return Cursor()

    @contextmanager
    def connect(_brain_id):
        yield Conn()

    with patch("src.services.api.routes.spreadsheets.postgres_connection", return_value=connect):
        response = api.delete("/retrieve/spreadsheets/00000000-0000-0000-0000-000000000001")
    assert response.status_code == 204

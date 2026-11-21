import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

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

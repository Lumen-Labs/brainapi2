from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.spreadsheets.import_workbook import MAX_COLUMNS, WorkbookError, read_workbook
from src.services.api.dependencies import get_brain_id
from src.services.api.routes.ingest import ingest_router


def test_csv_rejects_wide_header_before_cells():
    header = ",".join(f"c{i}" for i in range(MAX_COLUMNS + 1))
    try:
        read_workbook("wide.csv", f"{header}\n1\n".encode())
    except WorkbookError as exc:
        assert "column limit" in str(exc)
        return
    raise AssertionError("expected WorkbookError")


def test_csv_rejects_wide_row_before_cells():
    row = "," * (MAX_COLUMNS + 1)
    try:
        read_workbook("wide.csv", f"A\n{row}\n".encode())
    except WorkbookError as exc:
        assert "column limit" in str(exc)
        return
    raise AssertionError("expected WorkbookError")


def test_csv_accepts_bounded_sheet():
    name, columns, records = read_workbook("ok.csv", b"A,B\n1,2\n")[0]
    assert name == "CSV"
    assert columns == ["A", "B"]
    assert records[1][0]["type"] == "number"


def test_workbook_import_returns_202_when_status_write_fails():
    app = FastAPI()
    app.include_router(ingest_router)
    app.dependency_overrides[get_brain_id] = lambda: "demo"
    workbook = {"id": "t", "filename": "grid.csv", "sheets": [{"name": "CSV"}]}
    with (
        patch(
            "src.services.api.routes.ingest.spreadsheets.try_import_file",
            new=AsyncMock(return_value=workbook),
        ),
        patch(
            "src.services.api.routes.ingest.set_ingestion_task_status",
            side_effect=RuntimeError("cache down"),
        ),
    ):
        response = TestClient(app).post(
            "/ingest/file", files={"file": ("grid.csv", b"a,b\n1,2\n")}
        )
    assert response.status_code == 202
    assert response.json()["message"] == "Workbook imported"
    assert response.json()["task_id"]

from datetime import date, time
from io import BytesIO
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook

from src.core.spreadsheets.import_workbook import (
    MAX_COLUMNS,
    OLE_MAGIC,
    WorkbookError,
    cell,
    csv_cell,
    read_sheet,
    read_workbook,
)
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
    assert records[0]  # header stored
    assert len(records) == 2


def test_csv_infers_dates_currency_and_percentages():
    payload = (
        b"When,Amount,Rate\n"
        b'2026-01-31,"$1,234.56",15%\n'
        b'1/31/2026,"($99.00)",12.5%\n'
    )
    _, columns, records = read_workbook("grid.csv", payload)[0]
    assert columns == ["When", "Amount", "Rate"]
    assert records[1][0] == {"type": "date", "value": "2026-01-31"}
    assert records[1][1]["type"] == "number"
    assert records[1][1]["value"] == "1234.56"
    assert records[1][2]["type"] == "number"
    assert records[1][2]["value"] == "0.15"
    assert records[1][2]["display"] == "percent"
    assert records[2][0]["type"] == "date"
    assert records[2][1]["value"] == "-99.00"
    assert csv_cell("2026-01-31T00:00:00")["type"] == "date"
    assert csv_cell("2026-01-31T00:00:00")["value"] == "2026-01-31"
    assert csv_cell("2026-01-31T00:00:00.500000") == {
        "type": "date",
        "value": "2026-01-31T00:00:00.500000",
    }
    assert csv_cell("2026-01-31T00:00:00.000000")["value"] == "2026-01-31"


def test_empty_and_ole_files_are_rejected():
    try:
        read_workbook("empty.csv", b"")
    except WorkbookError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("expected empty workbook error")
    try:
        read_workbook("legacy.xls", OLE_MAGIC + b"not-xlsx")
    except WorkbookError as exc:
        assert "legacy" in str(exc).lower() or "encrypted" in str(exc).lower()
    else:
        raise AssertionError("expected OLE workbook error")
    try:
        read_workbook("bad.xlsx", b"not-a-zip")
    except WorkbookError as exc:
        assert "xlsx" in str(exc).lower()
    else:
        raise AssertionError("expected invalid xlsx error")


def _xlsx_bytes(build):
    workbook = Workbook()
    build(workbook.active)
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_xlsx_dates_percentages_and_merged_cells():
    def build(sheet):
        sheet["A1"] = "When"
        sheet["B1"] = "Rate"
        sheet["C1"] = "Note"
        sheet["A2"] = date(2026, 1, 31)
        sheet["B2"] = 0.15
        sheet["B2"].number_format = "0%"
        sheet["C2"] = "hello"
        sheet.merge_cells("C2:C3")

    _, columns, records = read_workbook("grid.xlsx", _xlsx_bytes(build))[0]
    assert columns == ["When", "Rate", "Note"]
    assert records[1][0]["type"] == "date"
    assert records[1][0]["value"] == "2026-01-31"
    assert records[1][1]["type"] == "number"
    assert records[1][1]["display"] == "percent"
    assert records[1][2]["value"] == "hello"
    assert records[2][2]["value"] == "hello"


def test_xlsx_time_only_cells_are_not_dates():
    def build(sheet):
        sheet["A1"] = "When"
        sheet["A2"] = time(12, 30, 0)

    _, _, records = read_workbook("times.xlsx", _xlsx_bytes(build))[0]
    assert records[1][0]["type"] == "time"
    assert records[1][0]["value"] == "12:30:00"


def test_read_sheet_enforces_budget_before_retaining_rows():
    produced = {"n": 0}

    def rows():
        yield [cell("A")]
        while True:
            produced["n"] += 1
            yield [cell(produced["n"])]

    try:
        read_sheet("CSV", rows(), budget=[5])
    except WorkbookError as exc:
        assert "250,000" in str(exc)
    else:
        raise AssertionError("expected over-budget error")
    assert produced["n"] <= 5


def test_merged_expansion_respects_cell_budget():
    records = ([cell("A")], [cell("x")])
    try:
        read_sheet("S", records, merges=[(1, 0, 3, 0)], budget=[2])
    except WorkbookError as exc:
        assert "250,000" in str(exc)
        return
    raise AssertionError("expected over-budget merged expansion")


def test_catalog_row_count_excludes_header():
    _, _, records = read_workbook("ok.csv", b"A\n1\n2\n")[0]
    from src.services.api.routes.spreadsheets import sheet_catalog

    catalog = sheet_catalog([("CSV", ["A"], records)])
    assert catalog[0]["rows"] == 2
    assert len(records) == 3


def test_workbook_import_returns_202_when_status_write_fails():
    app = FastAPI()
    app.include_router(ingest_router)
    app.dependency_overrides[get_brain_id] = lambda: "demo"
    workbook = {"id": "t", "filename": "grid.csv", "sheets": [{"name": "CSV"}]}
    with (
        patch(
            "src.services.api.routes.spreadsheets.try_import_file",
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

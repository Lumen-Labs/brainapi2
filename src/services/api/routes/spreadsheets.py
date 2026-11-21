"""Workbook upload and retrieval for PostgreSQL brains."""

import os
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from psycopg2 import errors
from psycopg2.extras import Json, execute_values
from starlette.concurrency import run_in_threadpool

from src.core.spreadsheets.import_workbook import MAX_UPLOAD_BYTES, WorkbookError, read_workbook
from src.core.spreadsheets.query import WorkbookQuery, query_workbook
from src.services.api.constants.responses import (
    SpreadsheetQueryResponse,
    SpreadsheetWorkbookListResponse,
    SpreadsheetWorkbookResponse,
)
from src.services.api.dependencies import get_brain_id

router = APIRouter(tags=["spreadsheets"])
SCHEMA = """
CREATE TABLE IF NOT EXISTS spreadsheets (
    id uuid PRIMARY KEY, filename text NOT NULL, sheets jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS spreadsheet_rows (
    workbook uuid NOT NULL REFERENCES spreadsheets(id) ON DELETE CASCADE,
    sheet integer NOT NULL, row_number integer NOT NULL, cells jsonb NOT NULL,
    PRIMARY KEY (workbook, sheet, row_number)
)
"""


def enabled(brain_id):
    keys = {key.strip() for key in os.getenv("SPREADSHEET_EXPERIMENT_BRAINS", "").split(",") if key.strip()}
    return "*" in keys or brain_id in keys


def supports_file(filename):
    return Path(filename or "").suffix.lower() in {".xlsx", ".csv"}


def postgres_connection():
    from src.services.data.main import data_adapter

    return getattr(data_adapter.data, "_brain_connection", None)


def connection(brain_id):
    connect = postgres_connection() if enabled(brain_id) else None
    if connect is None:
        raise HTTPException(404, "Spreadsheet queries are unavailable for this brain.")
    return connect(brain_id)


async def try_import_file(file, brain_id, *, workbook_id):
    if not enabled(brain_id) or not supports_file(file.filename) or postgres_connection() is None:
        return None
    try:
        return await import_uploaded_file(file, brain_id, workbook_id=workbook_id)
    except HTTPException as exc:
        if exc.status_code != 422:
            raise
        # Files outside the tabular import contract can still use the document parser.
        await file.seek(0)
        return None


def save_workbook(connect, filename, payload, workbook_id):
    try:
        sheets = read_workbook(filename, payload)
    except WorkbookError as exc:
        raise HTTPException(422, str(exc)) from exc
    workbook = workbook_id
    catalog = [
        {"name": name, "columns": columns, "index": index, "rows": len(rows)}
        for index, (name, columns, rows) in enumerate(sheets)
    ]
    with connect as conn, conn, conn.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(767947831)")
        cursor.execute(SCHEMA)
        cursor.execute("INSERT INTO spreadsheets (id, filename, sheets) VALUES (%s,%s,%s)",
                       (workbook, filename, Json(catalog)))
        records = (
            (workbook, index, row, Json(cells))
            for index, (_, _, rows) in enumerate(sheets)
            for row, cells in enumerate(rows, start=1)
        )
        execute_values(cursor, "INSERT INTO spreadsheet_rows VALUES %s", records, page_size=250)
    return {"id": workbook, "filename": filename, "sheets": catalog}


@router.post(
    "/ingest/spreadsheets",
    status_code=201,
    response_model=SpreadsheetWorkbookResponse,
)
async def upload_workbook(
    file: Annotated[UploadFile, File()],
    brain_id: str = Depends(get_brain_id),
):
    """
    Import an .xlsx or .csv workbook into PostgreSQL for the current brain.
    """
    return await import_uploaded_file(file, brain_id, workbook_id=str(uuid4()))


async def import_uploaded_file(file, brain_id, *, workbook_id):
    connect = connection(brain_id)
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    return await run_in_threadpool(
        save_workbook, connect, Path(file.filename or "").name, payload, workbook_id,
    )


@router.get("/retrieve/spreadsheets", response_model=SpreadsheetWorkbookListResponse)
def list_workbooks(brain_id: str = Depends(get_brain_id)):
    """
    List imported workbooks for the current brain.
    """
    with connection(brain_id) as conn, conn, conn.cursor() as cursor:
        try:
            cursor.execute("SELECT id, filename, sheets FROM spreadsheets ORDER BY created_at DESC, id")
        except errors.UndefinedTable:
            conn.rollback()
            return {"workbooks": []}
        return {"workbooks": [
            {"id": str(workbook), "filename": filename, "sheets": sheets}
            for workbook, filename, sheets in cursor
        ]}


@router.post("/retrieve/spreadsheets/query", response_model=SpreadsheetQueryResponse)
def execute_query(body: WorkbookQuery, brain_id: str = Depends(get_brain_id)):
    """
    Filter, project, or aggregate rows from an imported workbook.
    """
    try:
        with connection(brain_id) as conn, conn, conn.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            cursor.execute("SET LOCAL statement_timeout = '15s'")
            cursor.execute("SELECT sheets FROM spreadsheets WHERE id = %s", (str(body.workbook),))
            workbook = cursor.fetchone()
            if workbook is None:
                raise HTTPException(404, "Workbook not found.")
            return query_workbook(cursor, body, workbook[0])
    except errors.UndefinedTable as exc:
        raise HTTPException(404, "Workbook not found.") from exc
    except errors.DivisionByZero as exc:
        raise HTTPException(422, "Cannot divide by zero.") from exc
    except WorkbookError as exc:
        raise HTTPException(422, str(exc)) from exc
    except errors.QueryCanceled as exc:
        raise HTTPException(503, "Spreadsheet query exceeded its time limit.") from exc

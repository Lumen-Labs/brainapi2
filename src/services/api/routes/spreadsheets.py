"""Workbook upload and retrieval for PostgreSQL brains."""

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from psycopg2 import errors
from psycopg2.extras import Json, execute_values
from starlette.concurrency import run_in_threadpool

from src.core.saving.ingest_cost import IngestCostLedger
from src.core.spreadsheets.import_workbook import MAX_UPLOAD_BYTES, WorkbookError, read_workbook
from src.core.spreadsheets.query import PAGE_SIZE, WorkbookQuery, query_workbook
from src.lib.postgresql._provisioning import PoolTimeout
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
);
ALTER TABLE spreadsheets ADD COLUMN IF NOT EXISTS content_hash text;
ALTER TABLE spreadsheets ADD COLUMN IF NOT EXISTS cell_count integer NOT NULL DEFAULT 0;
ALTER TABLE spreadsheets ADD COLUMN IF NOT EXISTS byte_size integer NOT NULL DEFAULT 0;
CREATE UNIQUE INDEX IF NOT EXISTS spreadsheets_content_hash ON spreadsheets (content_hash);
"""
DEFAULT_MAX_WORKBOOKS = 500
DEFAULT_MAX_CELLS = 1_000_000
DEFAULT_MAX_BYTES = 100 * 1024 * 1024
BUSY = "Spreadsheet database is busy. Retry shortly."
RETRY_AFTER = {"Retry-After": "1"}


def enabled(brain_id):
    keys = {key.strip() for key in os.getenv("SPREADSHEET_EXPERIMENT_BRAINS", "").split(",") if key.strip()}
    return "*" in keys or brain_id in keys


def supports_file(filename):
    return Path(filename or "").suffix.lower() in {".xlsx", ".csv"}


def env_int(name, default):
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return default if value < 0 else value


def workbook_caps():
    return (
        env_int("SPREADSHEET_MAX_WORKBOOKS_PER_BRAIN", DEFAULT_MAX_WORKBOOKS),
        env_int("SPREADSHEET_MAX_CELLS_PER_BRAIN", DEFAULT_MAX_CELLS),
        env_int("SPREADSHEET_MAX_BYTES_PER_BRAIN", DEFAULT_MAX_BYTES),
    )


def postgres_connection():
    from src.services.data.main import data_adapter

    return getattr(data_adapter.data, "_brain_connection", None)


@contextmanager
def connection(brain_id):
    connect = postgres_connection() if enabled(brain_id) else None
    if connect is None:
        raise HTTPException(404, "Spreadsheet queries are unavailable for this brain.")
    try:
        with connect(brain_id) as conn:
            yield conn
    except PoolTimeout as exc:
        raise HTTPException(503, BUSY, headers=RETRY_AFTER) from exc


async def try_import_file(file, brain_id, *, workbook_id):
    if not enabled(brain_id) or postgres_connection() is None:
        return None
    suffix = Path(file.filename or "").suffix.lower()
    if suffix == ".xls":
        raise HTTPException(422, "Expected an .xlsx or .csv file.")
    if suffix not in {".xlsx", ".csv"}:
        return None
    return await import_uploaded_file(file, brain_id, workbook_id=workbook_id)


def sheet_catalog(sheets):
    return [
        {"name": name, "columns": columns, "index": index, "rows": max(0, len(rows) - 1)}
        for index, (name, columns, rows) in enumerate(sheets)
    ]


def cell_count(sheets):
    return sum(len(rows) * len(columns) for _, columns, rows in sheets)


def workbook_cost(sheets):
    texts = [
        str(item["value"])
        for _, _, rows in sheets
        for row in rows
        for item in row
        if item.get("value") is not None
    ]
    ledger = IngestCostLedger()
    ledger.set_source_text("\n".join(texts))
    return ledger.to_dict()


def workbook_payload(workbook, filename, catalog, *, cells, cost, deduplicated=False):
    result = {"id": workbook, "filename": filename, "sheets": catalog, "cells": cells, "cost": cost}
    if deduplicated:
        result["deduplicated"] = True
    return result


def duplicated_workbook(cursor, digest, cells, cost):
    cursor.execute(
        "SELECT id, filename, sheets FROM spreadsheets WHERE content_hash = %s",
        (digest,),
    )
    existing = cursor.fetchone()
    if not existing:
        return None
    return workbook_payload(
        str(existing[0]), existing[1], existing[2],
        cells=cells, cost=cost, deduplicated=True,
    )


def enforce_caps(cursor, cells, payload_size):
    max_workbooks, max_cells, max_bytes = workbook_caps()
    cursor.execute(
        "SELECT COUNT(*), COALESCE(SUM(cell_count), 0), COALESCE(SUM(byte_size), 0) FROM spreadsheets"
    )
    stored, stored_cells, stored_bytes = cursor.fetchone()
    if max_workbooks and stored >= max_workbooks:
        raise HTTPException(409, "Workbook count limit reached for this brain.")
    if max_cells and stored_cells + cells > max_cells:
        raise HTTPException(409, "Stored spreadsheet cell limit reached for this brain.")
    if max_bytes and stored_bytes + payload_size > max_bytes:
        raise HTTPException(409, "Stored spreadsheet size limit reached for this brain.")


def insert_workbook(cursor, workbook_id, filename, catalog, digest, cells, payload, sheets):
    cursor.execute(
        "INSERT INTO spreadsheets (id, filename, sheets, content_hash, cell_count, byte_size) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        (workbook_id, filename, Json(catalog), digest, cells, len(payload)),
    )
    records = (
        (workbook_id, index, row, Json(row_cells))
        for index, (_, _, rows) in enumerate(sheets)
        for row, row_cells in enumerate(rows, start=1)
    )
    execute_values(cursor, "INSERT INTO spreadsheet_rows VALUES %s", records, page_size=250)


def save_workbook(connect, filename, payload, workbook_id, brain_id):
    try:
        sheets = read_workbook(filename, payload)
    except WorkbookError as exc:
        raise HTTPException(422, str(exc)) from exc
    catalog = sheet_catalog(sheets)
    cells = cell_count(sheets)
    cost = workbook_cost(sheets)
    digest = hashlib.sha256(payload).hexdigest()
    with connect as conn, conn, conn.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(767947831)")
        cursor.execute(SCHEMA)
        existing = duplicated_workbook(cursor, digest, cells, cost)
        if existing:
            return existing
        enforce_caps(cursor, cells, len(payload))
        insert_workbook(cursor, workbook_id, filename, catalog, digest, cells, payload, sheets)
    remember_workbook(brain_id, workbook_id)
    return workbook_payload(workbook_id, filename, catalog, cells=cells, cost=cost)


def remember_workbook(brain_id, workbook_id):
    from src.core.saving.ingest_effects import remember

    remember(brain_id, workbook_id, {"workbook_id": workbook_id})


def lookup_workbook(brain_id, workbook_id):
    connect = postgres_connection()
    if connect is None:
        return None
    try:
        with connect(brain_id) as conn, conn, conn.cursor() as cursor:
            cursor.execute("SELECT id FROM spreadsheets WHERE id = %s", (str(workbook_id),))
            row = cursor.fetchone()
            return str(row[0]) if row else None
    except Exception:
        return None


def delete_workbook_data(brain_id, workbook_id):
    connect = postgres_connection()
    if connect is None:
        return
    with connect(brain_id) as conn, conn, conn.cursor() as cursor:
        cursor.execute("DELETE FROM spreadsheets WHERE id = %s", (str(workbook_id),))


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
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    return await run_in_threadpool(
        save_workbook, connection(brain_id), Path(file.filename or "").name, payload, workbook_id, brain_id,
    )


@router.get("/retrieve/spreadsheets", response_model=SpreadsheetWorkbookListResponse)
def list_workbooks(
    brain_id: str = Depends(get_brain_id),
    offset: int = Query(0, ge=0, le=250_000),
    limit: int = Query(PAGE_SIZE, ge=1, le=100),
):
    """
    List imported workbooks for the current brain.
    """
    with connection(brain_id) as conn, conn, conn.cursor() as cursor:
        try:
            cursor.execute("SELECT COUNT(*) FROM spreadsheets")
            total = cursor.fetchone()[0]
            cursor.execute(
                "SELECT id, filename, sheets FROM spreadsheets "
                "ORDER BY created_at DESC, id LIMIT %s OFFSET %s",
                (limit, offset),
            )
        except errors.UndefinedTable:
            conn.rollback()
            return {"workbooks": [], "next_offset": None}
        workbooks = [
            {"id": str(workbook), "filename": filename, "sheets": sheets}
            for workbook, filename, sheets in cursor
        ]
        next_offset = offset + len(workbooks)
        return {
            "workbooks": workbooks,
            "next_offset": next_offset if workbooks and next_offset < total else None,
        }


@router.delete("/retrieve/spreadsheets/{workbook_id}", status_code=204)
def delete_workbook(workbook_id: UUID, brain_id: str = Depends(get_brain_id)):
    """
    Delete one imported workbook and its rows for the current brain.
    """
    with connection(brain_id) as conn, conn, conn.cursor() as cursor:
        try:
            cursor.execute("DELETE FROM spreadsheets WHERE id = %s RETURNING id", (str(workbook_id),))
        except errors.UndefinedTable as exc:
            conn.rollback()
            raise HTTPException(404, "Workbook not found.") from exc
        if cursor.fetchone() is None:
            raise HTTPException(404, "Workbook not found.")
    return Response(status_code=204)


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

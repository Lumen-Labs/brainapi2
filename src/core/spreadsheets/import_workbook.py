"""Read spreadsheet values while retaining their original coordinates."""

import csv
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from io import BytesIO, StringIO
from itertools import zip_longest
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_EXPANDED_BYTES = 64 * 1024 * 1024
MAX_CELLS = 250_000
MAX_COLUMNS = 256
CSV_NUMBER = re.compile(r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")


class WorkbookError(ValueError):
    pass


def cell(value, *, kind=None, formula=None):
    if kind is None:
        if value is None:
            kind = "blank"
        elif isinstance(value, bool):
            kind = "boolean"
        elif isinstance(value, (int, float, Decimal)):
            kind, value = "number", str(Decimal(str(value)))
        elif isinstance(value, (date, datetime, time)):
            kind, value = "date", value.isoformat()
        elif isinstance(value, timedelta):
            kind, value = "duration", str(value)
        else:
            kind, value = "text", str(value)
    result = {"type": kind, "value": value}
    if formula is not None:
        result["formula"] = formula
    return result


def csv_cell(value):
    if value == "":
        return cell(None)
    if len(value) <= 100 and CSV_NUMBER.fullmatch(value):
        return cell(Decimal(value))
    return cell(value)


def xlsx_rows(original, cached):
    for source, saved in zip_longest(original.iter_rows(), cached.iter_rows(), fillvalue=()):
        if max(len(source), len(saved)) > MAX_COLUMNS:
            raise WorkbookError("Worksheet exceeds the column limit.")
        result = []
        for original_cell, cached_cell in zip_longest(source, saved):
            value = getattr(original_cell, "value", None)
            kind = getattr(original_cell, "data_type", None)
            if kind == "f":
                cached_value = getattr(cached_cell, "value", None)
                cached_kind = getattr(cached_cell, "data_type", None)
                formula = value if isinstance(value, str) else getattr(value, "text", "<array formula>")
                if cached_value is None or cached_kind == "e":
                    result.append(cell(None, kind="formula_unavailable", formula=formula))
                else:
                    result.append(cell(cached_value, formula=formula))
            elif kind == "e":
                result.append(cell(value, kind="error"))
            else:
                result.append(cell(value))
        yield result


def read_workbook(filename, payload):
    if len(payload) > MAX_UPLOAD_BYTES:
        raise WorkbookError("Spreadsheet exceeds the 10 MiB upload limit.")
    budget = MAX_CELLS

    def read_sheet(name, rows):
        nonlocal budget
        header = next(rows, [])
        while header and header[-1]["type"] == "blank":
            header.pop()
        if not header or len(header) > MAX_COLUMNS or any(
            item["type"] != "text" or not item["value"].strip() or item.get("formula")
            for item in header
        ):
            raise WorkbookError(f"{name} requires unique, nonblank text headers in row 1.")
        columns = [item["value"] for item in header]
        if len({name.strip().casefold() for name in columns}) != len(columns):
            raise WorkbookError(f"{name} has ambiguous column headers.")
        records = [header]
        budget -= len(columns)
        for row in rows:
            budget -= len(columns)
            if budget < 0:
                raise WorkbookError("Spreadsheet exceeds the 250,000-cell limit.")
            if any(item["type"] != "blank" for item in row[len(columns):]):
                raise WorkbookError(f"{name} has data outside its named columns.")
            records.append(row[:len(columns)] + [cell(None)] * max(0, len(columns) - len(row)))
        if budget < 0:
            raise WorkbookError("Spreadsheet exceeds the 250,000-cell limit.")
        if any(
            "\x00" in str(item["value"]) or "\x00" in item.get("formula", "")
            for row in records for item in row
        ):
            raise WorkbookError("Spreadsheet contains a null character.")
        return name, columns, records

    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        try:
            source = csv.reader(StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
            headers = next(source, [])
            rows = ([csv_cell(value) for value in row] for row in source)
            return [read_sheet("CSV", iter_with_header([cell(v) for v in headers], rows))]
        except (UnicodeError, csv.Error) as exc:
            raise WorkbookError("Expected a valid UTF-8 CSV.") from exc
    if suffix != ".xlsx":
        raise WorkbookError("Expected an .xlsx or .csv file.")

    from openpyxl import load_workbook
    from openpyxl.utils.exceptions import InvalidFileException
    from xml.etree.ElementTree import ParseError

    try:
        with ZipFile(BytesIO(payload)) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_EXPANDED_BYTES:
                raise WorkbookError("Workbook exceeds the expanded size limit.")
        original = load_workbook(BytesIO(payload), read_only=True, data_only=False, keep_links=False)
        try:
            cached = load_workbook(BytesIO(payload), read_only=True, data_only=True, keep_links=False)
            try:
                sheets = []
                for sheet in original.worksheets:
                    if (sheet.max_column or 0) > MAX_COLUMNS or (sheet.max_row or 0) > MAX_CELLS:
                        raise WorkbookError(f"{sheet.title} exceeds spreadsheet dimensions.")
                    saved = cached[sheet.title]
                    sheet.reset_dimensions()
                    saved.reset_dimensions()
                    sheets.append(read_sheet(sheet.title, xlsx_rows(sheet, saved)))
                if not sheets:
                    raise WorkbookError("Workbook contains no worksheets.")
                return sheets
            finally:
                cached.close()
        finally:
            original.close()
    except (BadZipFile, InvalidFileException, KeyError, OSError, ParseError) as exc:
        raise WorkbookError("Expected a valid XLSX workbook.") from exc


def iter_with_header(header, rows):
    yield header
    yield from rows

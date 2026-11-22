"""Read spreadsheet values while retaining their original coordinates."""

import csv
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from io import BytesIO, StringIO
from itertools import zip_longest
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_EXPANDED_BYTES = 64 * 1024 * 1024
MAX_CELLS = 250_000
MAX_COLUMNS = 256
OLE_MAGIC = b"\xd0\xcf\x11\xe0"
CSV_NUMBER = re.compile(r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
CSV_PERCENT = re.compile(r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?%")
CSV_THOUSANDS = re.compile(r"[+-]?(?:0|[1-9][0-9]{0,2}(?:,[0-9]{3})+)(?:\.[0-9]+)?")
CURRENCY_MARKS = "$£€¥"
CSV_ISO_DATE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}):(\d{2})(?:\.\d+)?)?$"
)
CSV_US_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
CSV_YMD_SLASH = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$")


class WorkbookError(ValueError):
    pass


def date_value(value):
    if isinstance(value, datetime):
        if value.tzinfo is None and value.time() == time.min:
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value.isoformat()


def cell(value, *, kind=None, formula=None, display=None):
    if kind is None:
        if value is None:
            kind = "blank"
        elif isinstance(value, bool):
            kind = "boolean"
        elif isinstance(value, (int, float, Decimal)):
            kind, value = "number", str(Decimal(str(value)))
        elif isinstance(value, (date, datetime, time)):
            kind, value = "date", date_value(value)
        elif isinstance(value, timedelta):
            kind, value = "duration", str(value)
        else:
            kind, value = "text", str(value)
    result = {"type": kind, "value": value}
    if formula is not None:
        result["formula"] = formula
    if display is not None:
        result["display"] = display
    return result


def _decimal(text):
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise WorkbookError("Expected a finite decimal.") from exc


def csv_date(value):
    match = CSV_ISO_DATE.fullmatch(value)
    if match:
        year, month, day, hour, minute, second = (int(part) if part else 0 for part in match.groups())
        if hour or minute or second:
            return datetime(year, month, day, hour, minute, second)
        return date(year, month, day)
    match = CSV_YMD_SLASH.fullmatch(value)
    if match:
        year, month, day = (int(part) for part in match.groups())
        return date(year, month, day)
    match = CSV_US_DATE.fullmatch(value)
    if not match:
        return None
    month, day, year = (int(part) for part in match.groups())
    return date(year, month, day)


def csv_currency(value):
    text = value.strip()
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()
    if not text or text[0] not in CURRENCY_MARKS:
        return None
    amount = text[1:].strip().replace(",", "")
    if amount.startswith("(") and amount.endswith(")"):
        negative = True
        amount = amount[1:-1]
    if amount.startswith("-"):
        negative = True
        amount = amount[1:]
    try:
        number = Decimal(amount)
    except InvalidOperation:
        return None
    return cell(-number if negative else number)


def csv_cell(value):
    if value == "":
        return cell(None)
    stripped = value.strip()
    lowered = stripped.casefold()
    if lowered in {"true", "false"}:
        return cell(lowered == "true")
    if len(stripped) <= 100 and CSV_PERCENT.fullmatch(stripped):
        return cell(_decimal(stripped[:-1]) / 100, display="percent")
    currency = csv_currency(stripped)
    if currency is not None:
        return currency
    if len(stripped) <= 100 and CSV_THOUSANDS.fullmatch(stripped):
        return cell(_decimal(stripped.replace(",", "")))
    if len(stripped) <= 100 and CSV_NUMBER.fullmatch(stripped):
        return cell(_decimal(stripped))
    try:
        parsed = csv_date(stripped)
    except ValueError:
        parsed = None
    if parsed is not None:
        return cell(parsed)
    return cell(value)


def percent_display(number_format):
    text = (number_format or "").lower()
    return "percent" if "%" in text else None


def xlsx_rows(original, cached):
    for source, saved in zip_longest(original.iter_rows(), cached.iter_rows(), fillvalue=()):
        if max(len(source), len(saved)) > MAX_COLUMNS:
            raise WorkbookError("Worksheet exceeds the column limit.")
        result = []
        for original_cell, cached_cell in zip_longest(source, saved):
            value = getattr(original_cell, "value", None)
            kind = getattr(original_cell, "data_type", None)
            display = percent_display(getattr(original_cell, "number_format", None))
            if kind == "f":
                cached_value = getattr(cached_cell, "value", None)
                cached_kind = getattr(cached_cell, "data_type", None)
                formula = value if isinstance(value, str) else getattr(value, "text", "<array formula>")
                cached_display = percent_display(getattr(cached_cell, "number_format", None)) or display
                if cached_value is None or cached_kind == "e":
                    result.append(cell(None, kind="formula_unavailable", formula=formula))
                else:
                    result.append(cell(cached_value, formula=formula, display=cached_display))
            elif kind == "e":
                result.append(cell(value, kind="error"))
            else:
                result.append(cell(value, display=display))
        yield result


def merge_bounds(sheet):
    ranges = getattr(getattr(sheet, "merged_cells", None), "ranges", ()) or ()
    return [
        (item.min_row - 1, item.min_col - 1, item.max_row - 1, item.max_col - 1)
        for item in ranges
    ]


def fill_merged(records, ranges):
    for row1, col1, row2, col2 in ranges:
        if row1 >= len(records) or col1 >= len(records[row1]):
            continue
        source = records[row1][col1]
        if source.get("type") == "blank":
            continue
        for row in range(row1, row2 + 1):
            while row >= len(records):
                records.append([])
            current = records[row]
            for col in range(col1, col2 + 1):
                while col >= len(current):
                    current.append(cell(None))
                if row == row1 and col == col1:
                    continue
                if current[col].get("type") == "blank":
                    current[col] = dict(source)


def read_sheet(name, rows, merges=None, budget=None):
    remaining = MAX_CELLS if budget is None else budget[0]
    records = [list(row) for row in rows]
    if merges:
        fill_merged(records, merges)
    if not records:
        raise WorkbookError(f"{name} requires unique, nonblank text headers in row 1.")
    header = records[0]
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
    remaining -= len(columns)
    padded = [header]
    for row in records[1:]:
        remaining -= len(columns)
        if remaining < 0:
            raise WorkbookError("Spreadsheet exceeds the 250,000-cell limit.")
        if any(item["type"] != "blank" for item in row[len(columns):]):
            raise WorkbookError(f"{name} has data outside its named columns.")
        padded.append(row[:len(columns)] + [cell(None)] * max(0, len(columns) - len(row)))
    if remaining < 0:
        raise WorkbookError("Spreadsheet exceeds the 250,000-cell limit.")
    if budget is not None:
        budget[0] = remaining
    if any(
        "\x00" in str(item["value"]) or "\x00" in item.get("formula", "")
        for row in padded for item in row
    ):
        raise WorkbookError("Spreadsheet contains a null character.")
    return name, columns, padded


def iter_with_header(header, rows):
    yield header
    yield from rows


def read_csv(filename, payload, budget):
    try:
        source = csv.reader(StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
        headers = next(source, [])
        if len(headers) > MAX_COLUMNS:
            raise WorkbookError("Worksheet exceeds the column limit.")

        def csv_rows():
            for row in source:
                if len(row) > MAX_COLUMNS:
                    raise WorkbookError("Worksheet exceeds the column limit.")
                yield [csv_cell(value) for value in row]

        return [read_sheet(
            "CSV",
            iter_with_header([cell(v) for v in headers], csv_rows()),
            budget=budget,
        )]
    except WorkbookError:
        raise
    except (UnicodeError, csv.Error) as exc:
        raise WorkbookError("Expected a valid UTF-8 CSV.") from exc


def read_xlsx(payload, budget):
    from openpyxl import load_workbook
    from openpyxl.utils.exceptions import InvalidFileException
    from xml.etree.ElementTree import ParseError

    try:
        with ZipFile(BytesIO(payload)) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_EXPANDED_BYTES:
                raise WorkbookError("Workbook exceeds the expanded size limit.")
        original = load_workbook(BytesIO(payload), read_only=False, data_only=False, keep_links=False)
        try:
            cached = load_workbook(BytesIO(payload), read_only=True, data_only=True, keep_links=False)
            try:
                sheets = []
                for sheet in original.worksheets:
                    if (sheet.max_column or 0) > MAX_COLUMNS or (sheet.max_row or 0) > MAX_CELLS:
                        raise WorkbookError(f"{sheet.title} exceeds spreadsheet dimensions.")
                    saved = cached[sheet.title]
                    if hasattr(sheet, "reset_dimensions"):
                        sheet.reset_dimensions()
                    if hasattr(saved, "reset_dimensions"):
                        saved.reset_dimensions()
                    sheets.append(read_sheet(
                        sheet.title,
                        xlsx_rows(sheet, saved),
                        merge_bounds(sheet),
                        budget=budget,
                    ))
                if not sheets:
                    raise WorkbookError("Workbook contains no worksheets.")
                return sheets
            finally:
                cached.close()
        finally:
            original.close()
    except WorkbookError:
        raise
    except (BadZipFile, InvalidFileException, KeyError, OSError, ParseError, ValueError) as exc:
        raise WorkbookError("Expected a valid XLSX workbook.") from exc


def read_workbook(filename, payload):
    if not payload:
        raise WorkbookError("Spreadsheet is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise WorkbookError("Spreadsheet exceeds the 10 MiB upload limit.")
    if payload.startswith(OLE_MAGIC):
        raise WorkbookError("Encrypted or legacy Excel files are not supported.")
    budget = [MAX_CELLS]
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return read_csv(filename, payload, budget)
    if suffix != ".xlsx":
        raise WorkbookError("Expected an .xlsx or .csv file.")
    return read_xlsx(payload, budget)

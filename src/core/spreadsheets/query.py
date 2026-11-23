"""Parameterized operations over a workbook's typed rows."""

import re
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .import_workbook import WorkbookError

PAGE_SIZE = 50
ADDRESS = re.compile(r"([A-Z]{1,3})([1-9][0-9]{0,6})", re.IGNORECASE)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Filter(Model):
    column: str
    op: Literal["eq", "ne", "lt", "lte", "gt", "gte", "blank", "not_blank"]
    value: str | bool | None = None
    type: Literal["text", "number", "boolean", "date"] = "text"


class Column(Model):
    column: str


class Cell(Model):
    sheet: str
    cell: str


class Number(Model):
    number: Annotated[Decimal, Field(max_digits=100, decimal_places=50)]


class Arithmetic(Model):
    op: Literal["add", "subtract", "multiply", "divide"]
    left: Column | Cell | Number
    right: Column | Cell | Number


class Calculation(Model):
    op: Literal["count", "sum", "avg", "min", "max"]
    column: str | None = None
    expression: Arithmetic | None = None


class WorkbookQuery(Model):
    workbook: UUID
    sheets: list[str] = Field(default_factory=list, max_length=100)
    range: str | None = None
    columns: list[str] = Field(default_factory=list, max_length=256)
    filters: list[Filter] = Field(default_factory=list, max_length=32)
    unique: bool = False
    expression: Arithmetic | None = None
    aggregate: Calculation | None = None
    offset: int = Field(default=0, ge=0, le=250_000)
    limit: int = Field(default=PAGE_SIZE, ge=1, le=100)


def address(value):
    match = ADDRESS.fullmatch(value)
    if match is None:
        raise WorkbookError("Use A1 cell coordinates.")
    col = 0
    for char in match[1].upper():
        col = col * 26 + ord(char) - ord("A") + 1
    return int(match[2]), col - 1


def column_letter(index):
    result = ""
    while index >= 0:
        index, letter = divmod(index, 26)
        result = chr(ord("A") + letter) + result
        index -= 1
    return result


def index_of(columns, name):
    if name not in columns:
        raise WorkbookError(f"Unknown column {name}.")
    return columns.index(name)


def ref(sheet, row, column):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", sheet):
        sheet = "'" + sheet.replace("'", "''") + "'"
    return f"{sheet}!{column_letter(column)}{row}"


def query_workbook(cursor, query, sheets):
    by_name = {sheet["name"]: sheet for sheet in sheets}
    selected = query.sheets or ([sheets[0]["name"]] if len(sheets) == 1 else [])
    if not selected:
        raise WorkbookError("Choose one or more worksheets.")
    if len(set(selected)) != len(selected) or any(name not in by_name for name in selected):
        raise WorkbookError("Worksheet selection contains an unknown or duplicate sheet.")
    active = [sheet for sheet in sheets if sheet["name"] in selected]
    columns = active[0]["columns"]
    if any(sheet["columns"] != columns for sheet in active):
        raise WorkbookError("Selected worksheets have incompatible headers.")
    if query.aggregate and (query.expression or query.columns):
        raise WorkbookError("An aggregate accepts its own column or expression.")
    if query.unique and query.aggregate:
        raise WorkbookError("A unique lookup cannot also aggregate.")
    if query.aggregate and query.aggregate.column and query.aggregate.expression:
        raise WorkbookError("Choose a calculation column or an expression.")
    if query.aggregate and query.aggregate.op == "count" and query.aggregate.expression:
        raise WorkbookError("COUNT accepts rows or a single column.")
    first_row, last_row = 2, max(sheet["rows"] for sheet in active)
    selection = list(range(len(columns)))
    if query.range:
        ends = query.range.split(":")
        if len(ends) > 2:
            raise WorkbookError("Use A1 or A1:B5 ranges.")
        first_row, first_col = address(ends[0])
        last_row, last_col = address(ends[-1])
        if first_row > last_row or first_col > last_col or last_col >= len(columns) or any(last_row > sheet["rows"] for sheet in active):
            raise WorkbookError("Range is outside the selected worksheets.")
        selection = list(range(first_col, last_col + 1))
    if query.columns:
        selection = [index_of(columns, name) for name in query.columns]
        if len(set(selection)) != len(selection) or query.range and any(
            index not in range(first_col, last_col + 1) for index in selection
        ):
            raise WorkbookError("Selected columns must be unique and inside the range.")

    args = [str(query.workbook), [sheet["index"] for sheet in active], first_row, last_row]
    conditions = ["workbook = %s AND sheet = ANY(%s) AND row_number BETWEEN %s AND %s"]
    if not query.range:
        conditions.append("EXISTS (SELECT 1 FROM jsonb_array_elements(cells) AS cell WHERE cell->>'type' <> 'blank')")
    for filt in query.filters:
        index = index_of(columns, filt.column)
        if filt.op in {"blank", "not_blank"}:
            conditions.append("(cells -> %s ->> 'type') " + ("=" if filt.op == "blank" else "<>") + " 'blank'")
            args.append(index)
            continue
        operators = {"eq": "=", "ne": "<>", "lt": "<", "lte": "<=", "gt": ">", "gte": ">="}
        if filt.value is None or (filt.type == "boolean") != isinstance(filt.value, bool):
            raise WorkbookError("Filter value must match its type.")
        if filt.type == "boolean" and filt.op not in {"eq", "ne"}:
            raise WorkbookError("Boolean filters support equality only.")
        value = str(filt.value).lower() if filt.type == "boolean" else filt.value
        lhs = "cells -> %s ->> 'value'"
        if filt.type == "number":
            try:
                value = Number(number=value).number
            except ValueError as exc:
                raise WorkbookError("Expected a finite decimal filter value.") from exc
            lhs = "CASE WHEN cells -> %s ->> 'type' = 'number' THEN (cells -> %s ->> 'value')::numeric END"
            conditions.append(f"({lhs}) {operators[filt.op]} %s")
            args.extend([index, index, value])
            continue
        conditions.append(f"(cells -> %s ->> 'type') = %s AND ({lhs}) {operators[filt.op]} %s")
        args.extend([index, filt.type, index, value])

    where = " AND ".join(conditions)
    table = "spreadsheet_rows"
    expression = query.aggregate.expression if query.aggregate else query.expression
    operands = []
    evidence = []
    numeric_columns = []

    def compile_operand(operand):
        if isinstance(operand, Number):
            operands.append(operand.number)
            return "%s::numeric"
        if isinstance(operand, Cell):
            if operand.sheet not in by_name:
                raise WorkbookError("Unknown operand worksheet.")
            sheet = by_name[operand.sheet]
            row, col = address(operand.cell)
            if row > sheet["rows"] or col >= len(sheet["columns"]):
                raise WorkbookError("Operand cell is outside its worksheet.")
            cursor.execute(
                f"SELECT cells -> %s FROM {table} WHERE workbook = %s AND sheet = %s AND row_number = %s",
                (col, str(query.workbook), sheet["index"], row),
            )
            value = cursor.fetchone()[0]
            if value["type"] != "number":
                raise WorkbookError("Operand cell must contain a number or a saved numeric formula result.")
            operands.append(Decimal(value["value"]))
            evidence.append(ref(operand.sheet, row, col))
            return "%s::numeric"
        index = index_of(columns, operand.column)
        numeric_columns.append(index)
        operands.extend([index, index])
        return "CASE WHEN cells -> %s ->> 'type' = 'number' THEN (cells -> %s ->> 'value')::numeric END"

    computed = None
    if expression:
        operators = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
        left = compile_operand(expression.left)
        right = compile_operand(expression.right)
        computed = f"(({left}) {operators[expression.op]} ({right}))"
    elif query.aggregate and query.aggregate.column and query.aggregate.op != "count":
        computed = compile_operand(Column(column=query.aggregate.column))

    for index in set(numeric_columns):
        cursor.execute(
            f"SELECT 1 FROM {table} WHERE {where} AND (cells -> %s ->> 'type') NOT IN ('number', 'blank') LIMIT 1",
            [*args, index],
        )
        if cursor.fetchone():
            raise WorkbookError(f"{columns[index]} contains a nonnumeric value or an unavailable formula result.")

    if query.expression and not numeric_columns and not query.range and not query.filters:
        cursor.execute(f"SELECT {computed}", operands)
        return {"matched": 1, "value": result_value(cursor.fetchone()[0]), "sources": evidence}

    cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", args)
    total = cursor.fetchone()[0]
    if query.unique and total > 1:
        raise WorkbookError("Multiple rows match. Specify a worksheet or a more specific key.")

    if query.aggregate:
        operation = query.aggregate.op
        if operation == "count":
            index = index_of(columns, query.aggregate.column) if query.aggregate.column else None
            if index is None:
                value = total
            else:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {where} AND (cells -> %s ->> 'type') <> 'blank'",
                    [*args, index],
                )
                value = cursor.fetchone()[0]
            used = ([] if query.filters else selection) if index is None else [index]
        else:
            if computed is None:
                raise WorkbookError("Choose a numeric column or expression for this aggregate.")
            cursor.execute(
                f"SELECT {operation}({computed}) FROM {table} WHERE {where}",
                [*operands, *args],
            )
            value = cursor.fetchone()[0]
            used = numeric_columns
        used = list(dict.fromkeys([*used, *(index_of(columns, filt.column) for filt in query.filters)]))
        spans = [
            f"{ref(sheet['name'], first_row, index)}:{column_letter(index)}{min(last_row, sheet['rows'])}"
            for sheet in active for index in used if first_row <= min(last_row, sheet["rows"])
        ]
        return {"matched": total, "value": result_value(value), "sources": list(dict.fromkeys([*evidence, *spans]))}

    cursor.execute(
        f"SELECT sheet, row_number, cells{', ' + computed if computed else ''} FROM {table} "
        f"WHERE {where} ORDER BY sheet, row_number LIMIT %s OFFSET %s",
        [*operands, *args, query.limit, query.offset],
    )
    names = {sheet["index"]: sheet["name"] for sheet in active}
    rows = []
    for sheet, row, cells, *calculation in cursor:
        values = [
            {"column": columns[index], "source": ref(names[sheet], row, index), **cells[index]}
            for index in selection
        ]
        result = {"sheet": names[sheet], "row": row, "cells": values}
        if calculation:
            result["calculation"] = {
                **result_value(calculation[0]),
                "sources": list(dict.fromkeys([*evidence, *(ref(names[sheet], row, i) for i in numeric_columns)])),
            }
        rows.append(result)
    offset = query.offset + len(rows)
    return {"matched": total, "rows": rows, "next_offset": offset if rows and offset < total else None}


def result_value(value):
    return {"type": "blank", "value": None} if value is None else {"type": "number", "value": str(value)}

"""Parameterized operations over a workbook's typed rows."""

import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .import_workbook import WorkbookError

PAGE_SIZE = 50
ADDRESS = re.compile(r"([A-Z]{1,3})([1-9][0-9]{0,6})", re.IGNORECASE)
TIMESTAMP_SQL = (
    "CASE WHEN cells -> %s ->> 'type' = 'date' "
    "THEN (cells -> %s ->> 'value')::timestamp END"
)


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
    group_by: list[str] = Field(default_factory=list, max_length=16)
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


def parse_filter_date(value):
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkbookError("Expected a date filter value.") from exc
    whole = parsed.tzinfo is None and parsed.time() == time.min
    start = parsed.replace(microsecond=0) if whole else parsed
    return start, start + timedelta(days=1) if whole else None, whole


def date_filter_sql(index, filt):
    start, end, whole = parse_filter_date(filt.value)
    ts = TIMESTAMP_SQL
    if whole:
        clauses = {
            "eq": (f"({ts}) >= %s AND ({ts}) < %s", [index, index, start, index, index, end]),
            "ne": (f"(({ts}) < %s OR ({ts}) >= %s)", [index, index, start, index, index, end]),
            "lt": (f"({ts}) < %s", [index, index, start]),
            "lte": (f"({ts}) < %s", [index, index, end]),
            "gt": (f"({ts}) >= %s", [index, index, end]),
            "gte": (f"({ts}) >= %s", [index, index, start]),
        }
    else:
        operator = {"eq": "=", "ne": "<>", "lt": "<", "lte": "<=", "gt": ">", "gte": ">="}[filt.op]
        clauses = {filt.op: (f"({ts}) {operator} %s", [index, index, start])}
    return clauses[filt.op]


def filter_sql(columns, filt):
    index = index_of(columns, filt.column)
    if filt.op in {"blank", "not_blank"}:
        operator = "=" if filt.op == "blank" else "<>"
        return f"(cells -> %s ->> 'type') {operator} 'blank'", [index]
    operators = {"eq": "=", "ne": "<>", "lt": "<", "lte": "<=", "gt": ">", "gte": ">="}
    if filt.value is None or (filt.type == "boolean") != isinstance(filt.value, bool):
        raise WorkbookError("Filter value must match its type.")
    if filt.type == "boolean" and filt.op not in {"eq", "ne"}:
        raise WorkbookError("Boolean filters support equality only.")
    if filt.type == "date":
        return date_filter_sql(index, filt)
    if filt.type == "number":
        try:
            value = Number(number=filt.value).number
        except ValueError as exc:
            raise WorkbookError("Expected a finite decimal filter value.") from exc
        lhs = "CASE WHEN cells -> %s ->> 'type' = 'number' THEN (cells -> %s ->> 'value')::numeric END"
        return f"({lhs}) {operators[filt.op]} %s", [index, index, value]
    value = str(filt.value).lower() if filt.type == "boolean" else filt.value
    return (
        f"(cells -> %s ->> 'type') = %s AND (cells -> %s ->> 'value') {operators[filt.op]} %s",
        [index, filt.type, index, value],
    )


def selected_sheets(query, sheets):
    by_name = {sheet["name"]: sheet for sheet in sheets}
    names = query.sheets or ([sheets[0]["name"]] if len(sheets) == 1 else [])
    if not names:
        raise WorkbookError("Choose one or more worksheets.")
    if len(set(names)) != len(names) or any(name not in by_name for name in names):
        raise WorkbookError("Worksheet selection contains an unknown or duplicate sheet.")
    active = [sheet for sheet in sheets if sheet["name"] in names]
    columns = active[0]["columns"]
    if any(sheet["columns"] != columns for sheet in active):
        raise WorkbookError("Selected worksheets have incompatible headers.")
    return by_name, active, columns


def row_window(query, active, columns):
    last_row = 1 + max(sheet["rows"] for sheet in active)
    first_row = 2
    selection = list(range(len(columns)))
    first_col = 0
    last_col = len(columns) - 1
    if query.range:
        ends = query.range.split(":")
        if len(ends) > 2:
            raise WorkbookError("Use A1 or A1:B5 ranges.")
        first_row, first_col = address(ends[0])
        last_row, last_col = address(ends[-1])
        if first_row > last_row or first_col > last_col or last_col >= len(columns) or any(
            last_row > 1 + sheet["rows"] for sheet in active
        ):
            raise WorkbookError("Range is outside the selected worksheets.")
        first_row = max(first_row, 2)
        selection = list(range(first_col, last_col + 1))
    if query.columns:
        selection = [index_of(columns, name) for name in query.columns]
        if len(set(selection)) != len(selection) or query.range and any(
            index not in range(first_col, last_col + 1) for index in selection
        ):
            raise WorkbookError("Selected columns must be unique and inside the range.")
    return first_row, last_row, selection


@dataclass
class QueryPlan:
    table: str
    by_name: dict
    active: list
    columns: list
    first_row: int
    last_row: int
    selection: list
    where: str
    args: list
    computed: str | None = None
    operands: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    numeric_columns: list = field(default_factory=list)


def query_scope(query, sheets):
    query_constraints(query)
    by_name, active, columns = selected_sheets(query, sheets)
    first_row, last_row, selection = row_window(query, active, columns)
    args = [str(query.workbook), [sheet["index"] for sheet in active], first_row, last_row]
    conditions = ["workbook = %s AND sheet = ANY(%s) AND row_number BETWEEN %s AND %s"]
    if not query.range:
        conditions.append(
            "EXISTS (SELECT 1 FROM jsonb_array_elements(cells) AS cell WHERE cell->>'type' <> 'blank')"
        )
    for filt in query.filters:
        clause, values = filter_sql(columns, filt)
        conditions.append(clause)
        args.extend(values)
    return QueryPlan(
        table="spreadsheet_rows",
        by_name=by_name,
        active=active,
        columns=columns,
        first_row=first_row,
        last_row=last_row,
        selection=selection,
        where=" AND ".join(conditions),
        args=args,
    )


def query_constraints(query):
    if query.aggregate and (query.expression or query.columns):
        raise WorkbookError("An aggregate accepts its own column or expression.")
    if query.unique and query.aggregate:
        raise WorkbookError("A unique lookup cannot also aggregate.")
    if query.unique and query.group_by:
        raise WorkbookError("A unique lookup cannot also group.")
    if query.group_by and query.expression:
        raise WorkbookError("group_by cannot be combined with a per-row expression.")
    if query.aggregate and query.aggregate.column and query.aggregate.expression:
        raise WorkbookError("Choose a calculation column or an expression.")
    if query.aggregate and query.aggregate.op == "count" and query.aggregate.expression:
        raise WorkbookError("COUNT accepts rows or a single column.")
    if query.group_by and not query.aggregate:
        query.aggregate = Calculation(op="count")


def result_value(value):
    return {"type": "blank", "value": None} if value is None else {"type": "number", "value": str(value)}


def compile_operand(cursor, table, query, by_name, columns, operand, operands, evidence, numeric_columns):
    if isinstance(operand, Number):
        operands.append(operand.number)
        return "%s::numeric"
    if isinstance(operand, Cell):
        if operand.sheet not in by_name:
            raise WorkbookError("Unknown operand worksheet.")
        sheet = by_name[operand.sheet]
        row, col = address(operand.cell)
        if row > 1 + sheet["rows"] or col >= len(sheet["columns"]):
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


def compiled_expression(cursor, query, plan):
    expression = query.aggregate.expression if query.aggregate else query.expression

    def bind(operand):
        return compile_operand(
            cursor, plan.table, query, plan.by_name, plan.columns,
            operand, plan.operands, plan.evidence, plan.numeric_columns,
        )

    if expression:
        operators = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
        plan.computed = f"(({bind(expression.left)}) {operators[expression.op]} ({bind(expression.right)}))"
    elif query.aggregate and query.aggregate.column and query.aggregate.op != "count":
        plan.computed = bind(Column(column=query.aggregate.column))


def reject_nonnumeric(cursor, plan):
    for index in set(plan.numeric_columns):
        cursor.execute(
            f"SELECT 1 FROM {plan.table} WHERE {plan.where} AND (cells -> %s ->> 'type') NOT IN ('number', 'blank') LIMIT 1",
            [*plan.args, index],
        )
        if cursor.fetchone():
            raise WorkbookError(
                f"{plan.columns[index]} contains a nonnumeric value or an unavailable formula result."
            )


def paged(query, rows, total):
    offset = query.offset + len(rows)
    return offset if rows and offset < total else None


def select_rows(cursor, query, plan, total):
    extra = f", {plan.computed}" if plan.computed else ""
    cursor.execute(
        f"SELECT sheet, row_number, cells{extra} FROM {plan.table} "
        f"WHERE {plan.where} ORDER BY sheet, row_number LIMIT %s OFFSET %s",
        [*plan.operands, *plan.args, query.limit, query.offset],
    )
    names = {sheet["index"]: sheet["name"] for sheet in plan.active}
    rows = [format_row(plan, names, sheet, row, cells, calculation)
            for sheet, row, cells, *calculation in cursor]
    return {"matched": total, "rows": rows, "next_offset": paged(query, rows, total)}


def format_row(plan, names, sheet, row, cells, calculation):
    result = {
        "sheet": names[sheet],
        "row": row,
        "cells": [
            {"column": plan.columns[index], "source": ref(names[sheet], row, index), **cells[index]}
            for index in plan.selection
        ],
    }
    if calculation:
        result["calculation"] = {
            **result_value(calculation[0]),
            "sources": list(dict.fromkeys(
                [*plan.evidence, *(ref(names[sheet], row, i) for i in plan.numeric_columns)]
            )),
        }
    return result


def query_workbook(cursor, query, sheets):
    plan = query_scope(query, sheets)
    compiled_expression(cursor, query, plan)
    reject_nonnumeric(cursor, plan)
    if query.expression and not plan.numeric_columns and not query.range and not query.filters:
        cursor.execute(f"SELECT {plan.computed}", plan.operands)
        return {"matched": 1, "value": result_value(cursor.fetchone()[0]), "sources": plan.evidence}
    if query.group_by:
        return grouped_result(cursor, query, plan)
    cursor.execute(f"SELECT COUNT(*) FROM {plan.table} WHERE {plan.where}", plan.args)
    total = cursor.fetchone()[0]
    if query.unique and total > 1:
        raise WorkbookError("Multiple rows match. Specify a worksheet or a more specific key.")
    if query.aggregate:
        return aggregate_result(cursor, query, plan, total)
    return select_rows(cursor, query, plan, total)


def group_value_sql(query, plan):
    operation = query.aggregate.op
    if operation == "count":
        index = index_of(plan.columns, query.aggregate.column) if query.aggregate.column else None
        if index is None:
            return "COUNT(*)", []
        return "COUNT(*) FILTER (WHERE (cells -> %s ->> 'type') <> 'blank')", [index]
    if plan.computed is None:
        raise WorkbookError("Choose a numeric column or expression for this aggregate.")
    return f"{operation}({plan.computed})", list(plan.operands)


def grouped_result(cursor, query, plan):
    keys = [index_of(plan.columns, name) for name in query.group_by]
    if len(set(keys)) != len(keys):
        raise WorkbookError("group_by columns must be unique.")
    key_sql = ", ".join(["cells -> %s"] * len(keys))
    value_sql, value_args = group_value_sql(query, plan)
    positions = ", ".join(str(position) for position in range(1, len(keys) + 1))
    cursor.execute(
        f"SELECT COUNT(*) FROM (SELECT 1 FROM {plan.table} WHERE {plan.where} GROUP BY {key_sql}) grouped",
        [*plan.args, *keys],
    )
    total = cursor.fetchone()[0]
    cursor.execute(
        f"SELECT {key_sql}, {value_sql} FROM {plan.table} WHERE {plan.where} "
        f"GROUP BY {positions} ORDER BY {positions} LIMIT %s OFFSET %s",
        [*keys, *value_args, *plan.args, query.limit, query.offset],
    )
    groups = [
        {
            "key": [
                {"column": plan.columns[index], **(item if isinstance(item, dict) else {"type": "blank", "value": None})}
                for index, item in zip(keys, row[:-1])
            ],
            "value": result_value(row[-1]),
            "sources": list(dict.fromkeys(plan.evidence)),
        }
        for row in cursor
    ]
    return {"matched": total, "groups": groups, "next_offset": paged(query, groups, total)}


def count_aggregate(cursor, query, plan, total):
    index = index_of(plan.columns, query.aggregate.column) if query.aggregate.column else None
    if index is None:
        return total, ([] if query.filters else plan.selection)
    cursor.execute(
        f"SELECT COUNT(*) FROM {plan.table} WHERE {plan.where} AND (cells -> %s ->> 'type') <> 'blank'",
        [*plan.args, index],
    )
    return cursor.fetchone()[0], [index]


def aggregate_result(cursor, query, plan, total):
    if query.aggregate.op == "count":
        value, used = count_aggregate(cursor, query, plan, total)
    elif plan.computed is None:
        raise WorkbookError("Choose a numeric column or expression for this aggregate.")
    else:
        cursor.execute(
            f"SELECT {query.aggregate.op}({plan.computed}) FROM {plan.table} WHERE {plan.where}",
            [*plan.operands, *plan.args],
        )
        value, used = cursor.fetchone()[0], plan.numeric_columns
    used = list(dict.fromkeys([*used, *(index_of(plan.columns, filt.column) for filt in query.filters)]))
    spans = [
        f"{ref(sheet['name'], plan.first_row, index)}:{column_letter(index)}{min(plan.last_row, 1 + sheet['rows'])}"
        for sheet in plan.active for index in used if plan.first_row <= min(plan.last_row, 1 + sheet["rows"])
    ]
    return {"matched": total, "value": result_value(value), "sources": list(dict.fromkeys([*plan.evidence, *spans]))}

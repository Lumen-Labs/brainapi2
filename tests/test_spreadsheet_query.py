from datetime import datetime

from src.core.spreadsheets.query import (
    Filter,
    WorkbookQuery,
    compiled_expression,
    date_filter_sql,
    filter_sql,
    grouped_result,
    parse_filter_date,
    query_scope,
    query_workbook,
    row_window,
)


def test_date_equals_covers_the_whole_day():
    start, end, whole = parse_filter_date("2026-01-31")
    assert whole is True
    assert start == datetime(2026, 1, 31)
    assert end == datetime(2026, 2, 1)
    sql, args = date_filter_sql(0, Filter(column="When", op="eq", value="2026-01-31", type="date"))
    assert ">=" in sql and "<" in sql
    assert args[3] == start
    assert args[-1] == end


def test_inclusive_end_date_includes_that_day():
    sql, args = date_filter_sql(0, Filter(column="When", op="lte", value="2026-01-31", type="date"))
    assert sql.count("<") == 1
    assert ">=" not in sql
    assert args[-1] == datetime(2026, 2, 1)


def test_timestamp_equals_is_exact():
    sql, args = date_filter_sql(
        0, Filter(column="When", op="eq", value="2026-01-31T12:30:00", type="date")
    )
    assert "=" in sql and ">=" not in sql
    assert args[-1] == datetime(2026, 1, 31, 12, 30, 0)


def test_date_filter_casts_stored_value_to_timestamp():
    sql, _args = filter_sql(
        ["When"], Filter(column="When", op="eq", value="2026-01-31", type="date")
    )
    assert "::timestamp" in sql
    assert "type' = 'date'" in sql
    assert "~" in sql


def test_date_filter_skips_time_only_values():
    sql, args = filter_sql(
        ["When"], Filter(column="When", op="eq", value="2026-01-31", type="date")
    )
    assert sql.count("%s") == len(args)
    assert "^[0-9]{4}-" in sql


def test_date_filter_on_time_column_returns_valid_result():
    query = WorkbookQuery(
        workbook="00000000-0000-0000-0000-000000000001",
        filters=[Filter(column="When", op="eq", value="2026-01-31", type="date")],
    )
    cursor = RecordingCursor([])
    result = query_workbook(cursor, query, _sheet())
    assert result["matched"] == 1
    assert result["rows"] == []
    assert any("^[0-9]{4}-" in sql for sql, _ in cursor.statements)


class RecordingCursor:
    def __init__(self, groups, total=1):
        self.statements = []
        self._groups = groups
        self._total = total

    def execute(self, sql, args=None):
        self.statements.append((sql, list(args or [])))

    def fetchone(self):
        return (self._total,)

    def __iter__(self):
        return iter(self._groups)


def _sheet():
    return [{"name": "Sales", "columns": ["When", "Amount"], "index": 0, "rows": 3}]


def test_group_by_uses_type_and_value_not_formula():
    query = WorkbookQuery(
        workbook="00000000-0000-0000-0000-000000000001",
        group_by=["When"],
        aggregate={"op": "sum", "column": "Amount"},
    )
    plan = query_scope(query, _sheet())
    compiled_expression(RecordingCursor([]), query, plan)
    cursor = RecordingCursor(
        [[{"type": "text", "value": "east"}, 10]],
        total=1,
    )
    result = grouped_result(cursor, query, plan)
    assert "jsonb_build_object" in cursor.statements[0][0]
    assert "cells -> %s ->> 'type'" in cursor.statements[0][0]
    assert "cells -> %s -> 'value'" in cursor.statements[0][0]
    assert "cells -> %s ->> 'value'" not in cursor.statements[0][0]
    assert "formula" not in cursor.statements[0][0]
    assert result["groups"][0]["sources"] == ["Sales!A2:A4", "Sales!B2:B4"]


def test_range_query_skips_the_header_row():
    query = WorkbookQuery(
        workbook="00000000-0000-0000-0000-000000000001",
        range="A1:B5",
    )
    first, last, selection = row_window(
        query,
        [{"name": "CSV", "columns": ["A", "B"], "rows": 10, "index": 0}],
        ["A", "B"],
    )
    assert first == 2
    assert last == 5
    assert selection == [0, 1]


def test_default_window_uses_data_row_count():
    query = WorkbookQuery(workbook="00000000-0000-0000-0000-000000000001")
    first, last, selection = row_window(
        query,
        [{"name": "CSV", "columns": ["A"], "rows": 98, "index": 0}],
        ["A"],
    )
    assert first == 2
    assert last == 99
    assert selection == [0]

from datetime import datetime

from src.core.spreadsheets.query import (
    WorkbookQuery,
    date_filter_sql,
    filter_sql,
    parse_filter_date,
    row_window,
)
from src.core.spreadsheets.query import Filter


def test_date_equals_covers_the_whole_day():
    start, end, whole = parse_filter_date("2026-01-31")
    assert whole is True
    assert start == datetime(2026, 1, 31)
    assert end == datetime(2026, 2, 1)
    sql, args = date_filter_sql(0, Filter(column="When", op="eq", value="2026-01-31", type="date"))
    assert ">=" in sql and "<" in sql
    assert args[2] == start
    assert args[5] == end


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
    assert "type' = 'date'" in sql.replace(" ", "") or "type' = 'date'" in sql


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

from __future__ import annotations

import re

MAX_READ_QUERY_ROWS = 100
READ_QUERY_TIMEOUT_MS = 10000

_READ_QUERY_START = re.compile(
    r"^\s*(SELECT|WITH|EXPLAIN|TABLE)\b",
    re.IGNORECASE | re.DOTALL,
)
_FORBIDDEN_SQL = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|"
    r"COPY|MERGE|CALL|DO|VACUUM|REINDEX|CLUSTER|"
    r"LISTEN|NOTIFY|LOAD|LOCK|DISCARD|RESET"
    r")\b",
    re.IGNORECASE,
)


class ReadQueryValidationError(Exception):
    pass


def validate_read_only_sql(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise ReadQueryValidationError("Empty query")
    parts = [part.strip() for part in normalized.split(";") if part.strip()]
    if len(parts) != 1:
        raise ReadQueryValidationError("Multiple statements are not allowed")
    normalized = parts[0]
    if _FORBIDDEN_SQL.search(normalized):
        raise ReadQueryValidationError("Only read-only SELECT queries are allowed")
    if not _READ_QUERY_START.match(normalized):
        raise ReadQueryValidationError(
            "Query must start with SELECT, WITH, EXPLAIN, or TABLE"
        )
    return normalized

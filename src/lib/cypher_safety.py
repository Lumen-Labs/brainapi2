"""Safe Cypher identifier handling for Neo4j query construction.

Values must be passed as query parameters. Labels, relationship types, property
names, and database names cannot be parameterized in Cypher, so they are
validated against a strict allowlist and backtick-quoted before interpolation.

This module is the only place that may interpolate identifier fragments into
Cypher. Callers assemble queries by concatenating return values from these
helpers with static Cypher text.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")
DATABASE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,62}$")
MAX_IDENTIFIER_LEN = 255
MAX_LIMIT = 1_000_000
MAX_PATH_DEPTH = 1_000
MAX_RAW_QUERY_ROWS = 100
RAW_QUERY_TIMEOUT_SECONDS = 10.0

_CROSS_DATABASE_RE = re.compile(
    r"\b("
    r"USE|"
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:COMPOSITE\s+)?DATABASE|"
    r"DROP\s+(?:COMPOSITE\s+)?DATABASE|"
    r"ALTER\s+DATABASE|"
    r"STOP\s+DATABASE|"
    r"START\s+DATABASE|"
    r"SHOW\s+DATABASES?"
    r")\b",
    re.IGNORECASE,
)
_ADMIN_PROCEDURE_RE = re.compile(
    r"\bCALL\s+(?:`?dbms`?\.|`?db`?\.`?admin)",
    re.IGNORECASE,
)
_ADMIN_PROCEDURE_COMPACT_RE = re.compile(r"call(?:dbms\.|db\.admin)")

_LABEL_TRANSLATION = str.maketrans(
    {
        " ": "_",
        "-": "_",
        ".": "_",
        ",": "_",
        ":": "_",
        ";": "_",
        "(": "_",
        ")": "_",
        "[": "_",
        "]": "_",
        "{": "_",
        "}": "_",
        "'": "_",
    }
)

_STATIC_PROPERTY_KEYS = frozenset(
    {
        "aliases",
        "amount",
        "deprecated",
        "description",
        "flow_key",
        "happened_at",
        "last_updated",
        "metadata",
        "name",
        "observations",
        "polarity",
        "properties",
        "source_chunk_ids",
        "superseded_by",
        "uuid",
        "v_id",
        "valid_from",
        "valid_to",
        "valid_at",
        "invalid_at",
    }
)


class InvalidCypherIdentifier(ValueError):
    """Raised when a Cypher identifier or bound integer fails validation."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "identifier",
        value: Any = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.value = value


class RawCypherRejected(ValueError):
    """Raised when a caller-supplied Cypher query is not allowed."""


def require_identifier(name: str, *, kind: str = "identifier") -> str:
    if not isinstance(name, str) or not name:
        raise InvalidCypherIdentifier(
            f"{kind} must be a non-empty string",
            kind=kind,
            value=name,
        )
    if len(name) > MAX_IDENTIFIER_LEN:
        raise InvalidCypherIdentifier(
            f"{kind} exceeds maximum length of {MAX_IDENTIFIER_LEN}",
            kind=kind,
            value=name,
        )
    if not IDENTIFIER_RE.fullmatch(name):
        raise InvalidCypherIdentifier(
            f"Invalid Cypher {kind}: {name!r}. "
            "Use letters, digits, and underscore only, starting with a letter or underscore.",
            kind=kind,
            value=name,
        )
    return name


def quote_identifier(name: str, *, kind: str = "identifier") -> str:
    ident = require_identifier(name, kind=kind)
    return "`" + ident.replace("`", "``") + "`"


def clean_label(label: str) -> str:
    if not isinstance(label, str) or not label.strip():
        raise InvalidCypherIdentifier(
            "Label must be a non-empty string",
            kind="label",
            value=label,
        )
    cleaned = label.translate(_LABEL_TRANSLATION).upper()
    return require_identifier(cleaned, kind="label")


def clean_labels(labels: Sequence[str]) -> list[str]:
    return [clean_label(label) for label in labels]


def quote_label(label: str) -> str:
    return quote_identifier(clean_label(label), kind="label")


def quote_labels(labels: Sequence[str]) -> str:
    quoted = [quote_label(label) for label in labels]
    if not quoted:
        raise InvalidCypherIdentifier(
            "At least one label is required",
            kind="label",
            value=labels,
        )
    return ":".join(quoted)


def optional_node_labels(labels: Sequence[str] | None) -> str:
    if not labels:
        return ""
    return ":" + quote_labels(labels)


def quote_rel_type(name: str) -> str:
    return quote_identifier(clean_label(name), kind="relationship type")


def quote_rel_types_pattern(types: Sequence[str]) -> str:
    if not types:
        raise InvalidCypherIdentifier(
            "At least one relationship type is required",
            kind="relationship type",
            value=types,
        )
    return "|".join(quote_rel_type(rel_type) for rel_type in types)


def clean_property_key(property_key: str) -> str:
    if not isinstance(property_key, str) or not property_key.strip():
        raise InvalidCypherIdentifier(
            "Property key must be a non-empty string",
            kind="property key",
            value=property_key,
        )
    sanitized = property_key.strip("`").strip()
    if not sanitized or "\x00" in sanitized:
        raise InvalidCypherIdentifier(
            "Property key must be a non-empty string",
            kind="property key",
            value=property_key,
        )
    words = sanitized.split()
    cleaned = words[0].lower() + "".join(word.capitalize() for word in words[1:])
    return cleaned


def quote_property_key(property_key: str) -> str:
    cleaned = clean_property_key(property_key)
    return quote_identifier(cleaned, kind="property key")


def static_property_key(key: str) -> str:
    if key not in _STATIC_PROPERTY_KEYS:
        raise InvalidCypherIdentifier(
            f"Refusing to interpolate non-allowlisted property key {key!r}",
            kind="property key",
            value=key,
        )
    return key


def require_database_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise InvalidCypherIdentifier(
            "Database name must be a non-empty string",
            kind="database",
            value=name,
        )
    if name.lower() == "system":
        raise InvalidCypherIdentifier(
            'Database name "system" is reserved',
            kind="database",
            value=name,
        )
    if not DATABASE_RE.fullmatch(name):
        raise InvalidCypherIdentifier(
            f"Invalid database name: {name!r}. "
            "Use a letter followed by letters, digits, dots, underscores, or dashes.",
            kind="database",
            value=name,
        )
    return name


def quote_database_name(name: str) -> str:
    ident = require_database_name(name)
    return "`" + ident.replace("`", "``") + "`"


def require_int(
    value: Any,
    *,
    name: str,
    minimum: int = 0,
    maximum: int = MAX_LIMIT,
) -> int:
    if isinstance(value, bool):
        raise InvalidCypherIdentifier(
            f"{name} must be an integer",
            kind="integer",
            value=value,
        )
    try:
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str) and value.strip().isdigit():
            parsed = int(value.strip())
        else:
            raise InvalidCypherIdentifier(
                f"{name} must be an integer",
                kind="integer",
                value=value,
            )
    except (ValueError, OverflowError) as exc:
        raise InvalidCypherIdentifier(
            f"{name} must be an integer",
            kind="integer",
            value=value,
        ) from exc
    if parsed < minimum or parsed > maximum:
        raise InvalidCypherIdentifier(
            f"{name} must be between {minimum} and {maximum}",
            kind="integer",
            value=value,
        )
    return parsed


def require_limit(value: Any, *, name: str = "limit") -> int:
    return require_int(value, name=name, minimum=0, maximum=MAX_LIMIT)


def require_path_depth(value: Any, *, name: str = "depth") -> int:
    return require_int(value, name=name, minimum=0, maximum=MAX_PATH_DEPTH)


def relationship_valid_at_clause(alias: str = "r", param: str = "as_of") -> str:
    """Cypher predicate: alias is valid at $param (ISO strings, half-open)."""
    rel = require_identifier(alias, kind="identifier")
    bound = require_identifier(param, kind="identifier")
    start = static_property_key("valid_from")
    end = static_property_key("valid_to")
    return (
        "(" + rel + "['" + start + "'] IS NULL OR " + rel + "['" + start + "'] <= $" + bound + ")"
        " AND "
        "(" + rel + "['" + end + "'] IS NULL OR " + rel + "['" + end + "'] > $" + bound + ")"
    )


def relationship_changed_in_window_clause(
    alias: str = "r",
    since_param: str = "since",
    until_param: str = "until",
) -> str:
    """Cypher predicate: alias has a validity stamp Python can window-filter.

    Do not compare stamp strings to $since/$until here. Historical
    ``invalid_at`` values are day-first (``DD/MM/YYYY``) and lose the
    window under lexicographic ISO compares. Callers pass since/until
    through ``change_kind`` after parse_instant.
    """
    rel = require_identifier(alias, kind="identifier")
    require_identifier(since_param, kind="identifier")
    require_identifier(until_param, kind="identifier")
    start = static_property_key("valid_from")
    end = static_property_key("valid_to")
    invalid = static_property_key("invalid_at")
    return (
        rel
        + "['"
        + start
        + "'] IS NOT NULL OR "
        + rel
        + "['"
        + end
        + "'] IS NOT NULL OR "
        + rel
        + "['"
        + invalid
        + "'] IS NOT NULL"
    )


def create_database_query(database: str) -> str:
    return "CREATE DATABASE " + quote_database_name(database)


def create_database_if_not_exists_query(database: str) -> str:
    return create_database_query(database) + " IF NOT EXISTS"


def show_database_status_query(database: str) -> str:
    return "SHOW DATABASE " + quote_database_name(database) + " YIELD currentStatus"


def _structural_cypher(query: str, *, blank_identifiers: bool = True) -> str:
    """Blank comments and quoted literals so keyword scans ignore them.

    String literals honor Cypher backslash escapes only. Backtick identifiers
    honor doubled backticks. Identifiers are blanked for database-admin scans
    so a quoted label cannot look like USE/CREATE DATABASE, and left intact
    for procedure scans so CALL `dbms`.components() is still visible.
    """
    out: list[str] = []
    i = 0
    n = len(query)
    while i < n:
        if query.startswith("//", i):
            while i < n and query[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if query.startswith("/*", i):
            out.append("  ")
            i += 2
            while i < n and not query.startswith("*/", i):
                out.append("\n" if query[i] == "\n" else " ")
                i += 1
            if i < n:
                out.append("  ")
                i += 2
            continue
        ch = query[i]
        if ch in {"'", '"'}:
            i = _skip_quoted(query, i, ch, out, blank=True, escape_backslash=True)
            continue
        if ch == "`":
            i = _skip_quoted(
                query,
                i,
                "`",
                out,
                blank=blank_identifiers,
                escape_backslash=False,
            )
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _skip_quoted(
    query: str,
    start: int,
    quote: str,
    out: list[str],
    *,
    blank: bool,
    escape_backslash: bool,
) -> int:
    n = len(query)
    i = start
    out.append(" " if blank else query[i])
    i += 1
    while i < n:
        if escape_backslash and query[i] == "\\" and i + 1 < n:
            if blank:
                out.append("  ")
            else:
                out.append(query[i : i + 2])
            i += 2
            continue
        if query[i] == quote:
            if not escape_backslash and i + 1 < n and query[i + 1] == quote:
                out.append("  " if blank else query[i : i + 2])
                i += 2
                continue
            out.append(" " if blank else query[i])
            i += 1
            break
        out.append("\n" if query[i] == "\n" else (" " if blank else query[i]))
        i += 1
    return i


def _cypher_statements(query: str) -> list[str]:
    structural = _structural_cypher(query)
    statements: list[str] = []
    current: list[str] = []
    for original, masked in zip(query, structural):
        if masked == ";":
            part = "".join(current).strip()
            if part:
                statements.append(part)
            current = []
        else:
            current.append(original)
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def validate_raw_cypher(query: str) -> str:
    """Reject empty, multi-statement, cross-database, and admin Cypher."""
    if not isinstance(query, str) or not query.strip():
        raise RawCypherRejected("Raw Cypher query must be a non-empty string")
    statements = _cypher_statements(query)
    if len(statements) != 1:
        raise RawCypherRejected("Multiple Cypher statements are not allowed")
    normalized = statements[0]
    structural = _structural_cypher(normalized)
    if _CROSS_DATABASE_RE.search(structural):
        raise RawCypherRejected(
            "Raw Cypher cannot switch databases or run database administration"
        )
    admin_structural = _structural_cypher(normalized, blank_identifiers=False)
    admin_compact = re.sub(r"[`\s]+", "", admin_structural).lower()
    if (
        _ADMIN_PROCEDURE_RE.search(admin_structural)
        or _ADMIN_PROCEDURE_COMPACT_RE.search(admin_compact)
    ):
        raise RawCypherRejected(
            "Raw Cypher cannot call dbms.* or db.admin procedures"
        )
    return normalized

"""Safe Cypher identifier handling for Neo4j query construction.

Values must be passed as query parameters. Labels, relationship types, property
names, and database names cannot be parameterized in Cypher, so they are
validated against a strict allowlist and backtick-quoted before interpolation.

This module is the only place that may interpolate identifier fragments into
Cypher. Callers assemble queries by concatenating return values from these
helpers with static Cypher text.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Sequence

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")
DATABASE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.-]{1,61}[A-Za-z0-9]$")
_PROCEDURE_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
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
_LOAD_CSV_RE = re.compile(r"\bLOAD\s+CSV\b", re.IGNORECASE)
_APOC_PROCEDURE_RE = re.compile(r"\bCALL\s+`?apoc`?", re.IGNORECASE)
_APOC_COMPACT_RE = re.compile(r"callapoc")

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
        "ingest_task_ids",
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
        "recorded_at",
        "inserted_at",
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


def _ascii_label(translated: str, *, original: str) -> str:
    decomposed = unicodedata.normalize("NFKD", translated)
    ascii_chars: list[str] = []
    leftover_non_ascii = False
    leftover_special = False
    for ch in decomposed:
        if unicodedata.combining(ch):
            continue
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            ascii_chars.append(ch)
        elif not ch.isascii():
            leftover_non_ascii = True
        else:
            leftover_special = True
    stem = "".join(ascii_chars)
    if leftover_special:
        raise InvalidCypherIdentifier(
            f"Invalid Cypher label: {original!r}. "
            "Use letters, digits, and underscore only, starting with a letter or underscore.",
            kind="label",
            value=original,
        )
    if leftover_non_ascii:
        digest = hashlib.sha256(
            unicodedata.normalize("NFKC", original.strip()).encode("utf-8")
        ).hexdigest()[:16]
        if stem:
            suffix = "_" + digest
            stem = stem[: max(1, MAX_IDENTIFIER_LEN - len(suffix))]
            return stem + suffix
        return "U" + digest
    if not stem:
        raise InvalidCypherIdentifier(
            f"Invalid Cypher label: {original!r}. "
            "Use letters, digits, and underscore only, starting with a letter or underscore.",
            kind="label",
            value=original,
        )
    if stem[0].isdigit():
        stem = "L_" + stem
    return stem[:MAX_IDENTIFIER_LEN]


def clean_label(label: str) -> str:
    if not isinstance(label, str) or not label.strip():
        raise InvalidCypherIdentifier(
            "Label must be a non-empty string",
            kind="label",
            value=label,
        )
    cleaned = label.translate(_LABEL_TRANSLATION).upper()
    return require_identifier(_ascii_label(cleaned, original=label), kind="label")


def original_type_properties(labels: Sequence[str]) -> dict[str, Any]:
    originals = [
        label
        for label in labels
        if isinstance(label, str) and any(ord(ch) > 127 for ch in label)
    ]
    if not originals:
        return {}
    properties: dict[str, Any] = {"original_type": originals[0]}
    if len(originals) > 1:
        properties["original_types"] = list(originals)
    return properties


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
            "Invalid database name. Use 3-63 letters, digits, and internal "
            "dots or dashes; start with a letter; do not end with a dot or dash.",
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
    """Cypher predicate: alias is valid at $param (ISO strings, half-open).

    Fail closed: a missing start (``valid_from`` / ``valid_at``) does not match.
    """
    rel = require_identifier(alias, kind="identifier")
    bound = require_identifier(param, kind="identifier")
    start = static_property_key("valid_from")
    legacy_start = static_property_key("valid_at")
    end = static_property_key("valid_to")
    start_ok = (
        "(("
        + rel
        + "['"
        + start
        + "'] IS NOT NULL AND "
        + rel
        + "['"
        + start
        + "'] <= $"
        + bound
        + ") OR ("
        + rel
        + "['"
        + start
        + "'] IS NULL AND "
        + rel
        + "['"
        + legacy_start
        + "'] IS NOT NULL AND "
        + rel
        + "['"
        + legacy_start
        + "'] <= $"
        + bound
        + "))"
    )
    end_ok = (
        "("
        + rel
        + "['"
        + end
        + "'] IS NULL OR "
        + rel
        + "['"
        + end
        + "'] > $"
        + bound
        + ")"
    )
    return start_ok + " AND " + end_ok


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
    keys = (
        "valid_from",
        "valid_to",
        "invalid_at",
        "recorded_at",
        "valid_at",
        "inserted_at",
    )
    return " OR ".join(
        rel + "['" + static_property_key(key) + "'] IS NOT NULL" for key in keys
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
            while i < n and query[i] not in "\r\n":
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
    current_original: list[str] = []
    current_masked: list[str] = []
    for original, masked in zip(query, structural):
        if masked == ";":
            _append_executable_statement(
                statements, current_original, current_masked
            )
            current_original = []
            current_masked = []
        else:
            current_original.append(original)
            current_masked.append(masked)
    _append_executable_statement(statements, current_original, current_masked)
    return statements


def _append_executable_statement(
    statements: list[str], original_chars: list[str], masked_chars: list[str]
) -> None:
    original = "".join(original_chars).strip()
    if original and "".join(masked_chars).strip():
        statements.append(original)


def _procedure_names_after_call(query: str) -> list[str]:
    """Find CALL keywords in identifier-blanked text, then parse names.

    Quoted labels such as ``call dbms.metrics`` therefore cannot look like
    CALL, while ``CALL `dbms`.components()`` is still visible.
    """
    blanked = _structural_cypher(query, blank_identifiers=True)
    unblanked = _structural_cypher(query, blank_identifiers=False)
    names: list[str] = []
    for match in re.finditer(r"\bCALL\b", blanked, re.IGNORECASE):
        i = match.end()
        n = len(unblanked)
        parts: list[str] = []
        while i < n:
            while i < n and unblanked[i].isspace():
                i += 1
            if i >= n:
                break
            if unblanked[i] == "`":
                i += 1
                start = i
                while i < n and unblanked[i] != "`":
                    i += 1
                parts.append(unblanked[start:i].replace("``", "`"))
                if i < n:
                    i += 1
            else:
                ident = _PROCEDURE_IDENT_RE.match(unblanked, i)
                if not ident:
                    break
                parts.append(ident.group(0))
                i = ident.end()
            while i < n and unblanked[i].isspace():
                i += 1
            if i < n and unblanked[i] == ".":
                i += 1
                continue
            break
        if parts:
            names.append(".".join(parts).lower())
    return names


def _is_admin_procedure(name: str) -> bool:
    return (
        name == "dbms"
        or name.startswith("dbms.")
        or name == "db.admin"
        or name.startswith("db.admin.")
    )


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
    if any(
        _is_admin_procedure(name)
        for name in _procedure_names_after_call(normalized)
    ):
        raise RawCypherRejected(
            "Raw Cypher cannot call dbms.* or db.admin procedures"
        )
    if _LOAD_CSV_RE.search(structural):
        raise RawCypherRejected("Raw Cypher cannot LOAD CSV")
    apoc_structural = _structural_cypher(normalized, blank_identifiers=False)
    apoc_compact = re.sub(r"[`\s]+", "", apoc_structural).lower()
    if _APOC_PROCEDURE_RE.search(apoc_structural) or _APOC_COMPACT_RE.search(
        apoc_compact
    ):
        raise RawCypherRejected("Raw Cypher cannot call apoc.* procedures")
    return normalized

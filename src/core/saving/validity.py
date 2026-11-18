"""Point-in-time validity for chunks and graph relationships.

Canonical stored timestamps are ISO-8601 UTC (``YYYY-MM-DDTHH:MM:SSZ``).
Readers still accept the historical ``invalid_at`` / ``superseded_at`` /
``valid_at`` / ``inserted_at`` stamps so existing brains keep answering
without a backfill. Default retrieve (no ``as_of``) still hides superseded
and invalidated copies unless ``include_history`` is on.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from src.utils.dates import parse_date_string, to_naive_utc

VALID_FROM = "valid_from"
VALID_TO = "valid_to"
SUPERSEDED_BY = "superseded_by"
VALIDITY_PROPERTY_KEYS = (VALID_FROM, VALID_TO, SUPERSEDED_BY)

_AS_OF_FORMATS = (
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%d",
)


class InvalidAsOf(ValueError):
    """Raised when an as-of / since / until value cannot be parsed."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def format_instant(value: datetime | None = None) -> str:
    instant = value or utc_now()
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    else:
        instant = instant.astimezone(timezone.utc)
    return instant.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_instant(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = to_naive_utc(value)
        return parsed.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z") and "+" not in text[1:] and text.count("-") >= 2:
        iso = text[:-1] + "+00:00"
        try:
            return to_naive_utc(datetime.fromisoformat(iso)).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(text)
        return to_naive_utc(parsed).replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in _AS_OF_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return to_naive_utc(parsed).replace(tzinfo=timezone.utc)
    parsed = parse_date_string(text)
    if parsed is None:
        return None
    return to_naive_utc(parsed).replace(tzinfo=timezone.utc)


def require_instant(value: Any, *, name: str = "as_of") -> datetime:
    parsed = parse_instant(value)
    if parsed is None:
        raise InvalidAsOf(f"{name} must be an ISO-8601 datetime")
    return parsed


def optional_instant(value: Any, *, name: str = "as_of") -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return require_instant(value, name=name)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def validity_props_from_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    props: dict[str, Any] = {}
    source = _mapping(mapping)
    for key in VALIDITY_PROPERTY_KEYS:
        if source.get(key) not in (None, ""):
            props[key] = source[key]
    return props


def stamp_valid_from(
    mapping: Mapping[str, Any] | None,
    source_timestamp: str | None = None,
) -> dict[str, Any]:
    props = dict(_mapping(mapping))
    if props.get(VALID_FROM):
        return props
    instant = parse_instant(source_timestamp) or parse_instant(props.get("valid_at"))
    props[VALID_FROM] = format_instant(instant)
    return props


def close_validity(
    mapping: Mapping[str, Any] | None,
    *,
    closed_at: str | None,
    superseded_by: str | None = None,
) -> dict[str, Any]:
    props = dict(_mapping(mapping))
    stamp = format_instant(parse_instant(closed_at) or utc_now())
    if not props.get(VALID_TO):
        props[VALID_TO] = stamp
    if superseded_by and not props.get(SUPERSEDED_BY):
        props[SUPERSEDED_BY] = superseded_by
    if not props.get("invalid_at"):
        props["invalid_at"] = stamp
    props["deprecated"] = True
    if not props.get("superseded_at"):
        props["superseded_at"] = stamp
    return props


def _valid_from_of(fields: Mapping[str, Any], inserted_at: Any = None) -> datetime | None:
    return (
        parse_instant(fields.get(VALID_FROM))
        or parse_instant(fields.get("valid_at"))
        or parse_instant(inserted_at)
    )


def _valid_to_of(fields: Mapping[str, Any]) -> datetime | None:
    return (
        parse_instant(fields.get(VALID_TO))
        or parse_instant(fields.get("invalid_at"))
        or parse_instant(fields.get("superseded_at"))
    )


def is_valid_at(
    fields: Mapping[str, Any] | None,
    as_of: datetime | None,
    *,
    inserted_at: Any = None,
    deprecated: bool = False,
) -> bool:
    """Half-open interval: valid_from <= as_of < valid_to.

    ``as_of is None`` is the live/current view: a copy is current when it has
    no end stamp and is not deprecated.
    """
    source = _mapping(fields)
    end = _valid_to_of(source)
    if as_of is None:
        if end is not None:
            return False
        if deprecated or source.get("deprecated") is True:
            return False
        return True
    start = _valid_from_of(source, inserted_at)
    if start is not None and as_of < start:
        return False
    if end is not None and as_of >= end:
        return False
    return True


def chunk_is_valid_at(chunk: Any, as_of: datetime | None) -> bool:
    meta = _mapping(getattr(chunk, "metadata", None))
    return is_valid_at(meta, as_of, inserted_at=getattr(chunk, "inserted_at", None))


def predicate_is_valid_at(predicate: Any, as_of: datetime | None) -> bool:
    props = _mapping(getattr(predicate, "properties", None))
    deprecated = bool(getattr(predicate, "deprecated", False) or props.get("deprecated"))
    return is_valid_at(props, as_of, deprecated=deprecated)


def interval_overlaps(
    fields: Mapping[str, Any] | None,
    since: datetime,
    until: datetime,
    *,
    inserted_at: Any = None,
) -> bool:
    source = _mapping(fields)
    start = _valid_from_of(source, inserted_at)
    end = _valid_to_of(source)
    if start is not None and since <= start < until:
        return True
    if end is not None and since <= end < until:
        return True
    return False


def change_kind(
    fields: Mapping[str, Any] | None,
    since: datetime,
    until: datetime,
    *,
    inserted_at: Any = None,
) -> str | None:
    source = _mapping(fields)
    start = _valid_from_of(source, inserted_at)
    end = _valid_to_of(source)
    added = start is not None and since <= start < until
    closed = end is not None and since <= end < until
    if closed and (source.get(SUPERSEDED_BY) or source.get("superseded_at")):
        return "superseded"
    if closed:
        return "invalidated"
    if added:
        return "added"
    return None


def event_timestamp(
    fields: Mapping[str, Any] | None,
    kind: str,
    *,
    inserted_at: Any = None,
) -> str:
    source = _mapping(fields)
    if kind == "added":
        instant = _valid_from_of(source, inserted_at)
    else:
        instant = _valid_to_of(source)
    return format_instant(instant or utc_now())

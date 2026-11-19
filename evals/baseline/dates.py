from __future__ import annotations

from datetime import date, datetime
from typing import Any

TIMESTAMP_FIELDS = ("source_timestamp", "valid_from", "valid_at")


def parse_date(value: str | None) -> date | None:
    raw = (value or "").strip()
    if not raw:
        return None
    raw = raw.replace("Z", "")
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw[:19] if "T" in raw else raw[:10], fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def in_force(doc: dict, as_of: str | None) -> bool:
    if not as_of:
        return True
    point = parse_date(as_of)
    if point is None:
        return True
    start = parse_date(str(doc.get("valid_from") or "") or None)
    end = parse_date(str(doc.get("valid_to") or "") or None)
    if start is None:
        return False
    if point < start:
        return False
    if end and point > end:
        return False
    return True


def filter_as_of(docs: list[dict], as_of: str | None) -> list[dict]:
    if not as_of:
        return list(docs)
    return [doc for doc in docs if in_force(doc, as_of)]


def document_source_timestamp(doc: dict[str, Any] | None) -> str | None:
    if not isinstance(doc, dict):
        return None
    for field in TIMESTAMP_FIELDS:
        raw = doc.get(field)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text
    return None


def is_versioned_document(doc: dict[str, Any] | None) -> bool:
    if not isinstance(doc, dict):
        return False
    if doc.get("clause_id") or doc.get("version"):
        return True
    return any(doc.get(field) not in (None, "") for field in TIMESTAMP_FIELDS)


def require_document_source_timestamp(
    doc: dict[str, Any] | None, path: str = ""
) -> str:
    loc = f"{path}: " if path else ""
    payload = doc if isinstance(doc, dict) else {}
    doc_id = payload.get("id")
    stamp = document_source_timestamp(payload)
    if not stamp:
        raise ValueError(
            f"{loc}document {doc_id} missing parseable "
            "source_timestamp/valid_from/valid_at"
        )
    if parse_date(stamp) is None:
        raise ValueError(
            f"{loc}document {doc_id} has unparseable source timestamp {stamp!r}"
        )
    return stamp

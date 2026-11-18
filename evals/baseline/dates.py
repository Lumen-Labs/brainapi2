from __future__ import annotations

from datetime import date, datetime


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

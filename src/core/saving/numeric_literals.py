"""Keep stated numeric literals on facts and distinguish counter-values."""

from __future__ import annotations

import math
import re
from typing import Any

_STATED_NUMBER_RE = re.compile(
    r"(?<![\d.])(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)(?![\d.])"
)


def parse_stated_number(text: str | None) -> float | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    match = _STATED_NUMBER_RE.search(raw)
    if not match:
        return None
    token = match.group("num").replace(",", "")
    try:
        number = float(token)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def coerce_amount(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return parse_stated_number(str(value))


def amount_identity_token(amount: Any) -> str:
    number = coerce_amount(amount)
    if number is None:
        return ""
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return f"{number:.6g}"


def looks_like_year(number: float) -> bool:
    return number == int(number) and 1000 <= int(number) <= 2100


def amounts_differ(left: Any, right: Any) -> bool:
    first = coerce_amount(left)
    second = coerce_amount(right)
    if first is None or second is None:
        return False
    if looks_like_year(first) or looks_like_year(second):
        return int(first) != int(second) if first == int(first) and second == int(second) else first != second
    return not math.isclose(first, second, rel_tol=0.0, abs_tol=1e-9)


def display_numeric_value(
    object_name: str | None,
    description: str | None,
    amount: Any,
) -> str:
    object_text = (object_name or "").strip()
    description_text = (description or "").strip()
    base = object_text or description_text
    token = amount_identity_token(amount)
    if not token:
        return base
    compact = base.replace(",", "")
    if token in compact:
        return base
    if not base:
        return token
    return f"{base} {token}".strip()


def _endpoint_is_event(endpoint: Any) -> bool:
    if endpoint is None:
        return False
    typ = str(getattr(endpoint, "type", None) or "").strip().lower()
    if typ == "event":
        return True
    labels = getattr(endpoint, "labels", None) or []
    return any(str(label).strip().lower() == "event" for label in labels)


def _text_has_token(text: str | None, token: str) -> bool:
    if not token:
        return True
    return token in (text or "").replace(",", "")


def retain_numeric_on_relationship(rel: Any) -> Any:
    amount = coerce_amount(getattr(rel, "amount", None))
    description = str(getattr(rel, "description", None) or "")
    name = str(getattr(rel, "name", None) or "")
    texts = [description, name]
    for endpoint in (getattr(rel, "tail", None), getattr(rel, "tip", None)):
        if endpoint is None:
            continue
        texts.append(str(getattr(endpoint, "name", None) or ""))
        texts.append(str(getattr(endpoint, "description", None) or ""))
    if amount is None:
        for text in texts:
            amount = parse_stated_number(text)
            if amount is not None:
                break
    if amount is None:
        return rel
    try:
        rel.amount = amount
    except Exception:
        pass
    token = amount_identity_token(amount)
    if token and not _text_has_token(description, token):
        rel.description = f"{description.rstrip()} ({token})" if description.strip() else token
    for endpoint in (getattr(rel, "tail", None), getattr(rel, "tip", None)):
        if not _endpoint_is_event(endpoint):
            continue
        end_name = str(getattr(endpoint, "name", None) or "").strip()
        if token and not _text_has_token(end_name, token):
            endpoint.name = f"{end_name} {token}".strip() if end_name else token
    props = dict(getattr(rel, "properties", None) or {})
    props["amount"] = amount
    try:
        rel.properties = props
    except Exception:
        pass
    return rel

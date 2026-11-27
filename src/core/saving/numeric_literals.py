"""Keep stated numeric literals on facts and distinguish counter-values."""

from __future__ import annotations

import math
import re
from typing import Any

_DATE_RE = re.compile(
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"(?:[T\s]\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?\b"
    r"|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b"
)
_STATED_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9.])"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)"
    r"(?:\s*(?P<suffix>[kKmMbB]\b|[Mm]illion|[Bb]illion|[Tt]housand))?"
    r"(?![A-Za-z0-9.])"
)
_VERSION_PREFIX_RE = re.compile(
    r"^(?:v|x|ver|version|rev|gen|generation|model|q[1-4])$",
    re.IGNORECASE,
)
_MIXED_CASE_RE = re.compile(r"[A-Z].*[a-z]|[a-z].*[A-Z]")
_UNIT_RE = re.compile(
    r"^(?:eur|usd|gbp|jpy|cny|chf|sek|nok|dkk|cad|aud|inr|kg|km|lb|oz|"
    r"%|percent|units?|people|employees?)$",
    re.IGNORECASE,
)
_TOKEN_SPLIT_RE = re.compile(r"\s+")
_SUFFIX_MULTIPLIER = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "m": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
}


def _suffix_multiplier(suffix: str | None) -> float:
    if not suffix:
        return 1.0
    return _SUFFIX_MULTIPLIER.get(suffix.strip().lower(), 1.0)


def _adjacent_token(text: str, index: int, *, forward: bool) -> str:
    if forward:
        remnant = text[index:].lstrip(" \t,;:()[]{}\"'")
    else:
        remnant = text[:index].rstrip(" \t,;:()[]{}\"'")
    if not remnant:
        return ""
    parts = _TOKEN_SPLIT_RE.split(remnant)
    if not parts:
        return ""
    return parts[0] if forward else parts[-1]


def _has_currency_or_unit(token: str) -> bool:
    stripped = (token or "").strip()
    if not stripped:
        return False
    if stripped[0] in "$€£¥":
        return True
    return bool(_UNIT_RE.match(stripped))


def _is_model_or_version_context(
    previous: str, number: float, token: str, suffix: str
) -> bool:
    if suffix or "," in token or "." in token:
        return False
    if number != int(number) or not 1 <= int(number) <= 99:
        return False
    if not previous:
        return False
    if _VERSION_PREFIX_RE.match(previous):
        return True
    return bool(_MIXED_CASE_RE.search(previous))


def _should_skip_match(
    text: str,
    match: re.Match[str],
    number: float,
    token: str,
    suffix: str,
) -> bool:
    previous = _adjacent_token(text, match.start(), forward=False)
    nxt = _adjacent_token(text, match.end(), forward=True)
    if _has_currency_or_unit(nxt):
        return False
    if _VERSION_PREFIX_RE.match(previous) and looks_like_year(number) and not suffix:
        return True
    return _is_model_or_version_context(previous, number, token, suffix)


def _match_rank(token: str, suffix: str, number: float) -> int:
    if suffix or "," in token:
        return 0
    if "." in token:
        return 1
    if looks_like_year(number):
        return 3
    return 2


def parse_stated_number(text: str | None) -> float | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    masked = _DATE_RE.sub(" ", raw)
    candidates: list[tuple[int, int, float]] = []
    for match in _STATED_NUMBER_RE.finditer(masked):
        raw_num = match.group("num")
        token = raw_num.replace(",", "")
        suffix = match.group("suffix") or ""
        try:
            number = float(token)
        except ValueError:
            continue
        if not math.isfinite(number):
            continue
        if _should_skip_match(masked, match, number, raw_num, suffix):
            continue
        scaled = number * _suffix_multiplier(suffix)
        if not math.isfinite(scaled):
            continue
        candidates.append((_match_rank(raw_num, suffix, number), match.start(), scaled))
    if not candidates:
        return None
    best_rank = min(item[0] for item in candidates)
    chosen = [item for item in candidates if item[0] == best_rank]
    return chosen[0][2]


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
        both_ints = first == int(first) and second == int(second)
        return int(first) != int(second) if both_ints else first != second
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


def _text_has_token(text: str | None, token: str) -> bool:
    if not token:
        return True
    return token in (text or "").replace(",", "")


def _source_span_text(rel: Any) -> str:
    explicit = getattr(rel, "source_span", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    props = getattr(rel, "properties", None) or {}
    if isinstance(props, dict):
        for key in ("source_span", "span", "evidence", "quote"):
            value = props.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def retain_numeric_on_relationship(rel: Any) -> Any:
    amount = coerce_amount(getattr(rel, "amount", None))
    description = str(getattr(rel, "description", None) or "")
    name = str(getattr(rel, "name", None) or "")
    if amount is None:
        for text in (_source_span_text(rel), description, name):
            if not text:
                continue
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
    props = dict(getattr(rel, "properties", None) or {})
    props["amount"] = amount
    try:
        rel.properties = props
    except Exception:
        pass
    return rel

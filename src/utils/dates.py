import calendar
from datetime import datetime, timedelta, timezone
from typing import Optional
import re

_DATE_INPUT_FORMATS = (
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B-%d-%Y",
    "%b-%d-%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H%M%S",
    "%Y-%m-%dT%H%M",
    "%Y%m%dT%H%M%S",
    "%Y%m%dT%H%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%I:%M %p on %d %B, %Y",
    "%I:%M %p on %d %b, %Y",
)

_RELATIVE_PATTERNS = (
    (re.compile(r"^yesterday$", re.I), lambda ref: ref - timedelta(days=1)),
    (re.compile(r"^tomorrow$", re.I), lambda ref: ref + timedelta(days=1)),
    (re.compile(r"^today$", re.I), lambda ref: ref),
    (re.compile(r"^last\s+week$", re.I), lambda ref: ref - timedelta(days=7)),
    (re.compile(r"^next\s+week$", re.I), lambda ref: ref + timedelta(days=7)),
    (re.compile(r"^last\s+month$", re.I), lambda ref: _shift_months(ref, -1)),
    (re.compile(r"^next\s+month$", re.I), lambda ref: _shift_months(ref, 1)),
    (
        re.compile(r"^(\d+)\s+days?\s+ago$", re.I),
        lambda ref, m: ref - timedelta(days=int(m.group(1))),
    ),
    (
        re.compile(r"^in\s+(\d+)\s+days?$", re.I),
        lambda ref, m: ref + timedelta(days=int(m.group(1))),
    ),
    (
        re.compile(r"^(\d+)\s+weeks?\s+ago$", re.I),
        lambda ref, m: ref - timedelta(weeks=int(m.group(1))),
    ),
)

_WEEKDAY_NAMES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _shift_months(ref: datetime, delta: int) -> datetime:
    month = ref.month - 1 + delta
    year = ref.year + month // 12
    month = month % 12 + 1
    day = min(ref.day, [31, 29 if calendar.isleap(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return ref.replace(year=year, month=month, day=day)


def to_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def parse_date_string(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    cleaned = value.strip()
    for fmt in _DATE_INPUT_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
        return to_naive_utc(parsed)
    return None


def normalize_date_string(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return value
    parsed = parse_date_string(value)
    if parsed is not None:
        return parsed.strftime("%d/%m/%Y")
    return value.strip()


def resolve_relative_date(
    value: Optional[str],
    reference_time: Optional[str | datetime] = None,
) -> Optional[str]:
    if not value or not isinstance(value, str):
        return value
    cleaned = value.strip()
    absolute = normalize_date_string(cleaned)
    if absolute != cleaned or parse_date_string(cleaned) is not None:
        return absolute

    if reference_time is None:
        return cleaned
    if isinstance(reference_time, datetime):
        ref = reference_time
    else:
        ref = parse_date_string(str(reference_time))
        if ref is None:
            return cleaned

    for pattern, resolver in _RELATIVE_PATTERNS:
        match = pattern.match(cleaned)
        if not match:
            continue
        if match.lastindex:
            resolved = resolver(ref, match)
        else:
            resolved = resolver(ref)
        return resolved.strftime("%d/%m/%Y")

    weekday_match = re.match(r"^(last|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", cleaned, re.I)
    if weekday_match:
        mode = weekday_match.group(1).lower()
        target = _WEEKDAY_NAMES[weekday_match.group(2).lower()]
        current = ref.weekday()
        if mode == "this":
            delta = (target - current) % 7
        elif mode == "next":
            delta = (target - current) % 7 or 7
        else:
            delta = -((current - target) % 7 or 7)
        return (ref + timedelta(days=delta)).strftime("%d/%m/%Y")

    return cleaned

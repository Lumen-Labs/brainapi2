from datetime import datetime
from typing import Optional

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
)


def normalize_date_string(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return value
    cleaned = value.strip()
    for fmt in _DATE_INPUT_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return value

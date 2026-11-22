"""Redact free-text query parameters from access and application logs."""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit, urlunsplit

CONTEXT_PATH = "/retrieve/context"
FREE_TEXT_PARAMS = frozenset({"text"})
REDACTED = "[redacted]"
_CONTEXT_QUERY_RE = re.compile(
    r"(?:https?://[^\s\"']+)?/retrieve/context\?[^\s\"']+",
    re.IGNORECASE,
)
_installed = False


def _is_context_path(path: str) -> bool:
    cleaned = (path or "").split("?", 1)[0].rstrip("/")
    return cleaned.endswith(CONTEXT_PATH)


def redact_query(query: str) -> str:
    parts: list[str] = []
    for key, value in parse_qsl(query, keep_blank_values=True):
        if key.lower() in FREE_TEXT_PARAMS:
            parts.append(f"{quote_plus(key)}={REDACTED}")
        else:
            parts.append(urlencode([(key, value)]))
    return "&".join(parts)


def redact_path(path: str) -> str:
    if not path or "?" not in path:
        return path
    parts = urlsplit(path)
    target_path = parts.path or path.split("?", 1)[0]
    if not _is_context_path(target_path):
        return path
    query = parts.query or path.split("?", 1)[1]
    redacted_query = redact_query(query)
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path or target_path,
            redacted_query,
            parts.fragment,
        )
    )


def redact_access_target(target: str) -> str:
    if not target or "/retrieve/context" not in target:
        return target
    if "?" not in target:
        return target
    match = _CONTEXT_QUERY_RE.search(target)
    if not match:
        return redact_path(target)
    return target[: match.start()] + redact_path(match.group(0)) + target[match.end() :]


class AccessLogPrivacyFilter(logging.Filter):
    """Rewrite GET /retrieve/context query strings so `text` is not logged."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            redacted = redact_access_target(args[2])
            if redacted != args[2]:
                record.args = args[:2] + (redacted,) + args[3:]
        elif isinstance(args, dict):
            for key, value in list(args.items()):
                if isinstance(value, str) and "/retrieve/context" in value:
                    args[key] = redact_access_target(value)
        if isinstance(record.msg, str) and "/retrieve/context?" in record.msg:
            record.msg = redact_access_target(record.msg)
        return True


def install_access_log_privacy() -> None:
    global _installed
    if _installed:
        return
    filt = AccessLogPrivacyFilter()
    for name in ("uvicorn.access", "uvicorn.error", "uvicorn"):
        logging.getLogger(name).addFilter(filt)
    logging.getLogger().addFilter(filt)
    _installed = True

"""Redact free-text query parameters from access and application logs."""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qsl, quote_plus, unquote, urlencode, urlsplit, urlunsplit

CONTEXT_PATH = "/retrieve/context"
FREE_TEXT_PARAMS = frozenset({"text"})
REDACTED = "[redacted]"
_CONTEXT_QUERY_RE = re.compile(
    r"(?:https?://[^\s\"]+)?/retrieve/(?:%63|c)ontext\?[^\s\"]*",
    re.IGNORECASE,
)
_installed = False
_original_factory = None


def _is_context_path(path: str) -> bool:
    cleaned = unquote((path or "").split("?", 1)[0]).rstrip("/").lower()
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
    parts = urlsplit(path, allow_fragments=False)
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
            "",
        )
    )


def redact_embedded_message(text: str) -> str:
    if not text or "?" not in text:
        return text
    return _CONTEXT_QUERY_RE.sub(lambda match: redact_path(match.group(0)), text)


def redact_access_target(target: str) -> str:
    if not target or "?" not in target:
        return target
    if "/retrieve/" not in target.lower() and "/retrieve/" not in unquote(target).lower():
        return target
    stripped = target.strip()
    if stripped.startswith("/") or stripped.lower().startswith("http"):
        redacted = redact_path(stripped)
        if redacted != stripped:
            start = target.find(stripped)
            if start >= 0:
                return target[:start] + redacted + target[start + len(stripped) :]
            return redacted
    return redact_embedded_message(target)


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
                if isinstance(value, str) and "/retrieve/" in value.lower():
                    args[key] = redact_access_target(value)
        try:
            rendered = record.getMessage()
        except (TypeError, ValueError):
            return True
        redacted_msg = redact_embedded_message(rendered)
        if redacted_msg == rendered:
            redacted_msg = redact_access_target(rendered)
        if redacted_msg != rendered:
            record.msg = redacted_msg
            record.args = ()
        return True


def _redacting_record_factory(*args, **kwargs):
    factory = _original_factory or logging.getLogRecordFactory()
    record = factory(*args, **kwargs)
    AccessLogPrivacyFilter().filter(record)
    return record


def install_access_log_privacy() -> None:
    global _installed, _original_factory
    if _installed:
        return
    filt = AccessLogPrivacyFilter()
    for name in ("uvicorn.access", "uvicorn.error", "uvicorn"):
        logging.getLogger(name).addFilter(filt)
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(filt)
    _original_factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(_redacting_record_factory)
    _installed = True

"""
File: /_naming.py
Project: postgresql

Helpers for converting a `brain_id` (which can be a UUID, name_key, or any
arbitrary string) into a deterministic Postgres-safe database identifier.

Postgres identifiers must be at most 63 bytes (NAMEDATALEN - 1) and, when
unquoted, can only contain `[a-z][a-z0-9_]*`. We always lowercase + sanitize
to ASCII `[a-z0-9_]`. If the resulting name exceeds the length budget we
truncate the human-readable tail and append a SHA-256 prefix so the mapping
stays stable and collision-resistant.
"""

from __future__ import annotations

import hashlib
import re

__all__ = ["brain_db_name", "BRAIN_DB_PREFIX"]


BRAIN_DB_PREFIX = "brain_"
_MAX_DBNAME_LEN = 63
_HASH_SUFFIX_LEN = 12

_SANITIZE_RE = re.compile(r"[^a-z0-9_]+")


def brain_db_name(brain_id: str) -> str:
    """Return the Postgres database name that hosts `brain_id`.

    Always lowercase, ASCII, and stable across runs for the same `brain_id`.
    """
    if not brain_id:
        raise ValueError("brain_id must be a non-empty string")

    sanitized = _SANITIZE_RE.sub("_", brain_id.lower()).strip("_")
    if not sanitized:
        sanitized = "x"

    candidate = f"{BRAIN_DB_PREFIX}{sanitized}"
    if len(candidate) <= _MAX_DBNAME_LEN:
        return candidate

    digest = hashlib.sha256(brain_id.encode("utf-8")).hexdigest()[:_HASH_SUFFIX_LEN]
    head_budget = _MAX_DBNAME_LEN - len(BRAIN_DB_PREFIX) - len(digest) - 1
    head = sanitized[:head_budget].rstrip("_")
    return f"{BRAIN_DB_PREFIX}{head}_{digest}"

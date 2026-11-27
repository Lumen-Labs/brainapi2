"""When ingest may close an older outgoing edge.

Same-type outgoing non-event edges are **not** closed by default. Only:

- relationship types on the single-valued allowlist (subject-level
  singletons such as ``HEADQUARTERED_IN``), or
- edges whose ``source_chunk_ids`` belong to chunks this ingest just
  superseded (source-revision-driven replacement).

``SINGLE_VALUED_RELATIONSHIP_TYPES`` is the source of truth for those
singletons. Conflict scan uses the same set for deterministic identity
conflicts. Ingest-time invalidation passes ``require_same_source=True``
so a single-valued type is closed only when the incoming edge shares
``source_key`` with the existing one, or when either side has no
``source_key`` (legacy / unknown source). Intent remember/correct still
close functional types across sources.

Multi-valued types such as ``WORKS_AT``, ``LOCATED_IN``, ``FOUNDED``,
and ``OPERATES`` accumulate. ``REQUIRES`` is multi-valued by default (a
rule can require several things). Operators may extend the allowlist
with ``VALIDITY_SINGLE_VALUED_REL_TYPES`` (comma-separated). This module
does not call an LLM.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping

# Subject-level singletons: one current value at a time.
# Shared with conflict scan for deterministic identity conflicts.
SINGLE_VALUED_RELATIONSHIP_TYPES = frozenset(
    {
        "BORN_IN",
        "CEO",
        "ESTABLISHED",
        "ESTABLISHED_IN",
        "FOUNDED_IN",
        "FOUNDED_YEAR",
        "HAS_CEO",
        "HAS_STATUS",
        "HEADQUARTERED_IN",
        "LED_BY",
        "LIVES_AT",
        "LIVES_IN",
        "MARRIED_TO",
        "STATUS",
    }
)

_ENV = "VALIDITY_SINGLE_VALUED_REL_TYPES"


def single_valued_relationship_types() -> frozenset[str]:
    names = set(SINGLE_VALUED_RELATIONSHIP_TYPES)
    extra = os.getenv(_ENV, "") or ""
    for part in extra.split(","):
        token = part.strip().upper()
        if token:
            names.add(token)
    return frozenset(names)


def relationship_type_is_single_valued(name: str | None) -> bool:
    token = (name or "").strip().upper()
    return bool(token) and token in single_valued_relationship_types()


def source_chunk_ids_from_mapping(mapping: Mapping[str, Any] | None) -> set[str]:
    source = dict(mapping or {})
    values: list[Any] = []
    raw = source.get("source_chunk_ids")
    if isinstance(raw, (list, tuple, set)):
        values.extend(raw)
    elif raw:
        values.append(raw)
    if source.get("source_chunk_id"):
        values.append(source.get("source_chunk_id"))
    nested = source.get("properties")
    if isinstance(nested, Mapping):
        values.extend(source_chunk_ids_from_mapping(nested))
    return {str(item) for item in values if item not in (None, "")}


def existing_edge_from_superseded_source(
    mapping: Mapping[str, Any] | None,
    superseded_chunk_ids: Iterable[str] | None,
) -> bool:
    superseded = {str(item) for item in (superseded_chunk_ids or []) if item}
    if not superseded:
        return False
    return bool(source_chunk_ids_from_mapping(mapping) & superseded)


def source_key_from_mapping(mapping: Mapping[str, Any] | None) -> str:
    source = dict(mapping or {})
    value = source.get("source_key")
    if value is None:
        nested = source.get("properties")
        if isinstance(nested, Mapping):
            value = nested.get("source_key")
    text = str(value or "").strip()
    return text


def should_close_outgoing_edge(
    *,
    relationship_name: str | None,
    existing_properties: Mapping[str, Any] | None,
    superseded_chunk_ids: Iterable[str] | None = None,
    incoming_source_key: str | None = None,
    require_same_source: bool = False,
) -> bool:
    if existing_edge_from_superseded_source(
        existing_properties, superseded_chunk_ids
    ):
        return True
    if not relationship_type_is_single_valued(relationship_name):
        return False
    if not require_same_source:
        return True
    incoming = str(incoming_source_key or "").strip()
    existing = source_key_from_mapping(existing_properties)
    if not incoming or not existing:
        return True
    return incoming == existing

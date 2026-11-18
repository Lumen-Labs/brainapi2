"""When ingest may close an older outgoing edge.

Same-type outgoing non-event edges are **not** closed by default. Only:

- relationship types on the single-valued allowlist (functional attributes
  such as ``LIVES_IN``), or
- edges whose ``source_chunk_ids`` belong to chunks this ingest just
  superseded (source-revision-driven replacement).

Multi-valued types such as ``WORKS_WITH`` and ``APPLIES_TO`` accumulate.
Operators may extend the allowlist with ``VALIDITY_SINGLE_VALUED_REL_TYPES``
(comma-separated). This module does not call an LLM.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping

# High-confidence functional (single-valued) outgoing predicates.
SINGLE_VALUED_RELATIONSHIP_TYPES = frozenset(
    {
        "BORN_IN",
        "EMPLOYED_BY",
        "HAS_STATUS",
        "HAS_TITLE",
        "HEADQUARTERED_IN",
        "LIVES_AT",
        "LIVES_IN",
        "LOCATED_IN",
        "MARRIED_TO",
        "REQUIRES",
        "WORKS_AT",
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


def should_close_outgoing_edge(
    *,
    relationship_name: str | None,
    existing_properties: Mapping[str, Any] | None,
    superseded_chunk_ids: Iterable[str] | None = None,
) -> bool:
    if existing_edge_from_superseded_source(
        existing_properties, superseded_chunk_ids
    ):
        return True
    return relationship_type_is_single_valued(relationship_name)

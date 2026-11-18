from __future__ import annotations

from typing import Any, Iterable, Mapping

from src.core.saving.conflicts import normalize_predicate
from src.core.saving.validity import close_validity
from src.core.saving.validity_cardinality import (
    existing_edge_from_superseded_source,
    relationship_type_is_single_valued,
    should_close_outgoing_edge,
)


class RetireLookupError(RuntimeError):
    """Raised when neighbors cannot be loaded after a replacement was written."""


def same_scope(subject_uuid: str, predicate: str, other_subject: str, other_predicate: str) -> bool:
    return (
        str(subject_uuid or "") == str(other_subject or "")
        and normalize_predicate(predicate) == normalize_predicate(other_predicate)
    )


def _is_event(entity: Any) -> bool:
    labels = [str(label).strip().lower() for label in (getattr(entity, "labels", None) or [])]
    if "event" in labels:
        return True
    return (getattr(entity, "type", None) or "").strip().lower() == "event"


def _closure_reason(
    props: Mapping[str, Any],
    superseded_chunk_ids: Iterable[str] | None,
    fallback: str,
) -> str:
    if existing_edge_from_superseded_source(props, superseded_chunk_ids):
        return "source_key rewrite"
    if relationship_type_is_single_valued(str(props.get("name") or "")):
        return "single-valued predicate replaced"
    return fallback


def retire_outgoing_matches(
    graph: Any,
    *,
    brain_id: str,
    subject_uuid: str,
    predicate_name: str,
    keep_uuid: str | None,
    reason: str,
    closed_at: str,
    superseded_by: str | None,
    superseded_chunk_ids: Iterable[str] | None = None,
    require_functional_or_source: bool = True,
    always_uuids: Iterable[str] | None = None,
) -> list[str]:
    reason_text = (reason or "").strip()
    if not reason_text:
        raise ValueError("retirements need a reason")
    if not subject_uuid or not predicate_name:
        return []
    forced = {str(item) for item in (always_uuids or []) if item}
    try:
        neighbors = graph.get_neighbors([subject_uuid], brain_id=brain_id)
    except Exception as exc:
        raise RetireLookupError("Could not load neighbors to retire prior facts") from exc
    closed: list[str] = []
    for predicate, neighbor in (neighbors or {}).get(subject_uuid) or []:
        if not predicate or not neighbor:
            continue
        if (getattr(predicate, "direction", None) or "").strip().lower() not in {"", "out", "neutral"}:
            if (getattr(predicate, "direction", None) or "").strip().lower() == "in":
                continue
        if _is_event(neighbor):
            continue
        if not same_scope(subject_uuid, predicate_name, subject_uuid, getattr(predicate, "name", "")):
            continue
        fact_id = str(getattr(predicate, "uuid", "") or "")
        if keep_uuid and fact_id == keep_uuid:
            continue
        props = getattr(predicate, "properties", None) or {}
        if not isinstance(props, Mapping):
            props = {}
        if props.get("invalid_at") or props.get("valid_to"):
            continue
        force_this = fact_id in forced
        if (
            require_functional_or_source
            and not force_this
            and not should_close_outgoing_edge(
                relationship_name=predicate_name,
                existing_properties=props,
                superseded_chunk_ids=superseded_chunk_ids,
            )
        ):
            continue
        edge_reason = reason_text if force_this else _closure_reason(
            {**dict(props), "name": predicate_name},
            superseded_chunk_ids,
            reason_text,
        )
        closed_props = close_validity(
            props,
            closed_at=closed_at,
            superseded_by=superseded_by or keep_uuid,
            reason=edge_reason,
        )
        graph.update_properties(
            fact_id,
            "relationship",
            brain_id=brain_id,
            new_properties={
                "invalid_at": closed_props.get("invalid_at") or closed_at,
                "deprecated": True,
                "valid_to": closed_props.get("valid_to") or closed_at,
                "superseded_by": closed_props.get("superseded_by") or superseded_by or keep_uuid,
                "retired_reason": closed_props.get("retired_reason") or edge_reason,
            },
        )
        closed.append(fact_id)
    return closed

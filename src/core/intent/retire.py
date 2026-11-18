from __future__ import annotations

from typing import Any, Iterable, Mapping

from src.core.saving.conflicts import normalize_predicate
from src.core.saving.validity import close_validity
from src.core.saving.validity_cardinality import should_close_outgoing_edge


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
) -> list[str]:
    reason_text = (reason or "").strip()
    if not reason_text:
        raise ValueError("retirements need a reason")
    if not subject_uuid or not predicate_name:
        return []
    try:
        neighbors = graph.get_neighbors([subject_uuid], brain_id=brain_id)
    except Exception:
        return []
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
        if require_functional_or_source and not should_close_outgoing_edge(
            relationship_name=predicate_name,
            existing_properties=props,
            superseded_chunk_ids=superseded_chunk_ids,
        ):
            continue
        closed_props = close_validity(
            props,
            closed_at=closed_at,
            superseded_by=superseded_by or keep_uuid,
            reason=reason_text,
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
                "retired_reason": closed_props.get("retired_reason") or reason_text,
            },
        )
        closed.append(fact_id)
    return closed

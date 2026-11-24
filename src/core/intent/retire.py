from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

from src.core.saving.conflicts import normalize_predicate
from src.core.saving.validity import (
    VALID_FROM,
    VALID_TO,
    close_validity,
    format_instant,
    parse_instant,
)
from src.core.saving.validity_cardinality import (
    existing_edge_from_superseded_source,
    relationship_type_is_single_valued,
    should_close_outgoing_edge,
)


class RetireLookupError(RuntimeError):
    """Raised when neighbors cannot be loaded after a replacement was written."""


@dataclass
class RetirementResult:
    closed: list[str] = field(default_factory=list)
    successor_start: str | None = None
    successor_uuid: str | None = None


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


def _props_map(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        nested = value.get("properties")
        if isinstance(nested, Mapping):
            merged = dict(nested)
            merged.update({k: v for k, v in value.items() if k != "properties"})
            return merged
        return dict(value)
    props = getattr(value, "properties", None)
    return dict(props) if isinstance(props, Mapping) else {}


def _start_of(props: Mapping[str, Any]) -> datetime | None:
    return parse_instant(props.get(VALID_FROM)) or parse_instant(props.get("valid_at"))


def _is_open(props: Mapping[str, Any]) -> bool:
    return not (props.get("invalid_at") or props.get(VALID_TO) or props.get("valid_to"))


def apply_open_update(
    graph: Any,
    fact_id: str,
    brain_id: str,
    new_properties: dict[str, Any],
) -> bool:
    """Update a relationship only when it is still open (no valid_to / invalid_at)."""
    getter = getattr(graph, "get_fact_record", None)
    if callable(getter):
        try:
            row = getter(fact_id, brain_id)
        except TypeError:
            row = getter(fact_id)
        except Exception:
            row = None
        if row is not None and not _is_open(_props_map(row)):
            return False
    updater = getattr(graph, "update_properties", None)
    if not callable(updater):
        return False
    try:
        updated = updater(
            fact_id,
            "relationship",
            brain_id=brain_id,
            new_properties=new_properties,
            expected_open=True,
        )
    except TypeError:
        updated = updater(
            fact_id,
            "relationship",
            brain_id=brain_id,
            new_properties=new_properties,
        )
    return updated is not None


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
) -> RetirementResult:
    reason_text = (reason or "").strip()
    if not reason_text:
        raise ValueError("retirements need a reason")
    if not subject_uuid or not predicate_name:
        return RetirementResult()
    forced = {str(item) for item in (always_uuids or []) if item}
    incoming_start = parse_instant(closed_at)
    try:
        neighbors = graph.get_neighbors([subject_uuid], brain_id=brain_id)
    except Exception as exc:
        raise RetireLookupError("Could not load neighbors to retire prior facts") from exc
    closed: list[str] = []
    successor_start: datetime | None = None
    successor_uuid: str | None = None
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
        props = _props_map(predicate)
        if not _is_open(props):
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
        existing_start = _start_of(props)
        if (
            not force_this
            and incoming_start is not None
            and existing_start is not None
            and existing_start > incoming_start
        ):
            if successor_start is None or existing_start < successor_start:
                successor_start = existing_start
                successor_uuid = fact_id
            continue
        close_stamp = closed_at
        if existing_start is not None:
            close_dt = parse_instant(closed_at) or incoming_start
            if close_dt is not None and close_dt < existing_start:
                continue
            if close_dt is not None:
                close_stamp = format_instant(close_dt)
        edge_reason = reason_text if force_this else _closure_reason(
            {**dict(props), "name": predicate_name},
            superseded_chunk_ids,
            reason_text,
        )
        closed_props = close_validity(
            props,
            closed_at=close_stamp,
            superseded_by=superseded_by or keep_uuid,
            reason=edge_reason,
        )
        valid_to = closed_props.get("valid_to") or close_stamp
        if existing_start is not None:
            end_dt = parse_instant(valid_to)
            if end_dt is not None and end_dt < existing_start:
                continue
        applied = apply_open_update(
            graph,
            fact_id,
            brain_id,
            {
                "invalid_at": closed_props.get("invalid_at") or close_stamp,
                "deprecated": True,
                "valid_to": valid_to,
                "superseded_by": closed_props.get("superseded_by") or superseded_by or keep_uuid,
                "retired_reason": closed_props.get("retired_reason") or edge_reason,
            },
        )
        if applied:
            closed.append(fact_id)
    return RetirementResult(
        closed=closed,
        successor_start=format_instant(successor_start) if successor_start else None,
        successor_uuid=successor_uuid,
    )

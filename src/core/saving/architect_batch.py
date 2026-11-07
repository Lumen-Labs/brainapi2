from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field

from src.constants.agents import RelationshipAmount
from src.core.saving.grounding import (
    align_span,
    endpoints_well_formed,
    find_span_offsets,
    realign_span_offsets,
)


class BatchEndpoint(BaseModel):
    uuid: Optional[str] = None
    name: str
    type: str
    description: Optional[str] = None
    happened_at: Optional[str] = None
    polarity: Optional[str] = "neutral"
    properties: Optional[dict] = Field(default_factory=dict)


class BatchRelationship(BaseModel):
    tail: BatchEndpoint
    tip: BatchEndpoint
    name: str
    description: Optional[str] = None
    amount: RelationshipAmount = None
    source_span: str = ""
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    happened_at: Optional[str] = None
    properties: Optional[dict] = Field(default_factory=dict)


class BatchExtractResponse(BaseModel):
    new_nodes: list[BatchEndpoint] = Field(default_factory=list)
    relationships: list[BatchRelationship] = Field(default_factory=list)


@dataclass
class BatchValidationIssue:
    index: int
    reason: str
    relationship: Optional[BatchRelationship] = None


@dataclass
class BatchValidationResult:
    accepted: list[BatchRelationship] = field(default_factory=list)
    rejected: list[BatchValidationIssue] = field(default_factory=list)
    new_nodes: list[BatchEndpoint] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.accepted) and not self.rejected

    @property
    def usable(self) -> bool:
        """True when at least one relationship survived validation."""
        return bool(self.accepted)

    @property
    def reasons(self) -> list[str]:
        return [f"[{i.index}] {i.reason}" for i in self.rejected]


def _is_type_named_placeholder(endpoint: Any) -> bool:
    name = (getattr(endpoint, "name", None) or "").strip()
    typ = (getattr(endpoint, "type", None) or "").strip()
    if not name or not typ:
        return True
    return name.lower() == typ.lower()


def _is_event_endpoint(endpoint: Any) -> bool:
    return (getattr(endpoint, "type", None) or "").strip().lower() == "event"


def _endpoint_exists(
    endpoint: BatchEndpoint,
    *,
    known_uuids: set[str],
    known_keys: set[tuple[str, str]],
) -> bool:
    if endpoint.uuid and endpoint.uuid in known_uuids:
        return True
    key = (
        (endpoint.name or "").strip().lower(),
        (endpoint.type or "").strip().lower(),
    )
    return key in known_keys


def _scout_lookup(
    scout_entities: Sequence[Any],
) -> tuple[dict[str, Any], dict[tuple[str, str], Any]]:
    by_uuid: dict[str, Any] = {}
    by_key: dict[tuple[str, str], Any] = {}
    for entity in scout_entities:
        uuid = getattr(entity, "uuid", None) or (
            entity.get("uuid") if isinstance(entity, dict) else None
        )
        name = getattr(entity, "name", None) or (
            entity.get("name") if isinstance(entity, dict) else None
        )
        typ = getattr(entity, "type", None) or (
            entity.get("type") if isinstance(entity, dict) else None
        )
        if uuid:
            by_uuid[str(uuid)] = entity
        if name and typ:
            by_key[(str(name).strip().lower(), str(typ).strip().lower())] = entity
    return by_uuid, by_key


def _entity_description(entity: Any) -> str:
    if entity is None:
        return ""
    if isinstance(entity, dict):
        return (entity.get("description") or "").strip()
    return (getattr(entity, "description", None) or "").strip()


def _fill_event_description(
    endpoint: BatchEndpoint,
    *,
    rel: BatchRelationship,
    scout_by_uuid: dict[str, Any],
    scout_by_key: dict[tuple[str, str], Any],
) -> bool:
    """
    Ensure EVENT endpoint has a non-empty description.
    Prefers Scout description, then relationship description, then source_span,
    then a non-placeholder endpoint name.
    Returns True if description is non-empty after fill.
    """
    if not _is_event_endpoint(endpoint):
        return True
    if (endpoint.description or "").strip():
        return True

    scout = None
    if endpoint.uuid and endpoint.uuid in scout_by_uuid:
        scout = scout_by_uuid[endpoint.uuid]
    else:
        key = (
            (endpoint.name or "").strip().lower(),
            (endpoint.type or "").strip().lower(),
        )
        scout = scout_by_key.get(key)

    name = (endpoint.name or "").strip()
    for candidate in (
        _entity_description(scout),
        (rel.description or "").strip(),
        (rel.source_span or "").strip(),
        name if name and not _is_type_named_placeholder(endpoint) else "",
    ):
        if candidate:
            endpoint.description = candidate
            return True
    return False


def _ensure_endpoint_uuid(
    endpoint: BatchEndpoint,
    *,
    scout_by_uuid: dict[str, Any],
    scout_by_key: dict[tuple[str, str], Any],
) -> None:
    """Fill missing endpoint uuid from scout match or mint a fresh one."""
    if endpoint.uuid:
        return
    key = (
        (endpoint.name or "").strip().lower(),
        (endpoint.type or "").strip().lower(),
    )
    scout = scout_by_key.get(key)
    if scout is not None:
        scout_uuid = getattr(scout, "uuid", None) or (
            scout.get("uuid") if isinstance(scout, dict) else None
        )
        if scout_uuid:
            endpoint.uuid = str(scout_uuid)
            return
    import uuid as _uuid

    endpoint.uuid = str(_uuid.uuid4())


def _admit_endpoint(
    endpoint: BatchEndpoint,
    *,
    known_uuids: set[str],
    known_keys: set[tuple[str, str]],
    admitted: list[BatchEndpoint],
    scout_by_uuid: dict[str, Any] | None = None,
    scout_by_key: dict[tuple[str, str], Any] | None = None,
) -> bool:
    """
    Soft-register a well-formed unknown relationship endpoint so dense extracts
    stay on the schema path instead of all-reject → tooler escalate.
    """
    if _is_type_named_placeholder(endpoint) or not (endpoint.name or "").strip():
        return False
    typ = (endpoint.type or "").strip()
    if not typ:
        return False
    key = (endpoint.name.strip().lower(), typ.lower())
    _ensure_endpoint_uuid(
        endpoint,
        scout_by_uuid=scout_by_uuid or {},
        scout_by_key=scout_by_key or {},
    )
    if endpoint.uuid:
        known_uuids.add(endpoint.uuid)
    if key not in known_keys:
        known_keys.add(key)
        admitted.append(endpoint)
    return True


def ensure_event_descriptions(
    rel: BatchRelationship,
    *,
    scout_by_uuid: dict[str, Any],
    scout_by_key: dict[tuple[str, str], Any],
) -> tuple[bool, str]:
    for endpoint in (rel.tail, rel.tip):
        if not _is_event_endpoint(endpoint):
            continue
        if not _fill_event_description(
            endpoint,
            rel=rel,
            scout_by_uuid=scout_by_uuid,
            scout_by_key=scout_by_key,
        ):
            return False, "missing_event_description"
    return True, ""


def _span_in_bounds(
    rel: BatchRelationship,
    source_text: str,
) -> tuple[bool, str]:
    text = source_text or ""
    span = (rel.source_span or "").strip()
    if not span:
        return False, "missing_source_span"

    ok, new_start, new_end, reason = realign_span_offsets(
        span=span,
        source_text=text,
        span_start=rel.span_start,
        span_end=rel.span_end,
    )
    if not ok:
        return False, reason or "source_span_not_grounded"

    rel.span_start = new_start
    rel.span_end = new_end
    # Prefer exact quote from source when offsets were repaired.
    if new_start is not None and new_end is not None:
        sliced = text[new_start:new_end]
        if sliced.strip():
            rel.source_span = sliced
    return True, ""


def normalize_loose(value: str) -> str:
    return " ".join((value or "").split()).strip().lower()


def validate_batch_extract(
    payload: BatchExtractResponse | dict[str, Any],
    *,
    source_text: str,
    scout_entities: Sequence[Any],
) -> BatchValidationResult:
    if isinstance(payload, dict):
        payload = BatchExtractResponse.model_validate(payload)

    known_uuids: set[str] = set()
    known_keys: set[tuple[str, str]] = set()
    scout_by_uuid, scout_by_key = _scout_lookup(scout_entities)
    for entity in scout_entities:
        uuid = getattr(entity, "uuid", None) or (entity.get("uuid") if isinstance(entity, dict) else None)
        name = getattr(entity, "name", None) or (entity.get("name") if isinstance(entity, dict) else None)
        typ = getattr(entity, "type", None) or (entity.get("type") if isinstance(entity, dict) else None)
        if uuid:
            known_uuids.add(str(uuid))
        if name and typ:
            known_keys.add((str(name).strip().lower(), str(typ).strip().lower()))

    kept_nodes: list[BatchEndpoint] = []
    result = BatchValidationResult(new_nodes=[])
    for node in payload.new_nodes or []:
        if _is_type_named_placeholder(node):
            result.rejected.append(
                BatchValidationIssue(
                    index=-1,
                    reason=f"new_node_type_named_placeholder:{node.name}",
                )
            )
            continue
        if _is_event_endpoint(node) and not (node.description or "").strip():
            scout = None
            if node.uuid and node.uuid in scout_by_uuid:
                scout = scout_by_uuid[node.uuid]
            else:
                scout = scout_by_key.get(
                    (
                        (node.name or "").strip().lower(),
                        (node.type or "").strip().lower(),
                    )
                )
            filled = _entity_description(scout)
            if filled:
                node.description = filled
            else:
                result.rejected.append(
                    BatchValidationIssue(
                        index=-1,
                        reason=f"new_node_missing_event_description:{node.name}",
                    )
                )
                continue
        if not node.uuid:
            import uuid as _uuid

            node.uuid = str(_uuid.uuid4())
        kept_nodes.append(node)
        if node.uuid:
            known_uuids.add(node.uuid)
        known_keys.add(
            ((node.name or "").strip().lower(), (node.type or "").strip().lower())
        )
    result.new_nodes = kept_nodes

    for idx, rel in enumerate(payload.relationships or []):
        if _is_type_named_placeholder(rel.tail) or _is_type_named_placeholder(rel.tip):
            result.rejected.append(
                BatchValidationIssue(
                    index=idx,
                    reason="type_named_placeholder_endpoint",
                    relationship=rel,
                )
            )
            continue
        if not endpoints_well_formed(rel):
            result.rejected.append(
                BatchValidationIssue(
                    index=idx,
                    reason="malformed_endpoints",
                    relationship=rel,
                )
            )
            continue
        if not (rel.name or "").strip():
            result.rejected.append(
                BatchValidationIssue(
                    index=idx,
                    reason="missing_predicate",
                    relationship=rel,
                )
            )
            continue
        if not _endpoint_exists(rel.tail, known_uuids=known_uuids, known_keys=known_keys):
            if not _admit_endpoint(
                rel.tail,
                known_uuids=known_uuids,
                known_keys=known_keys,
                admitted=kept_nodes,
                scout_by_uuid=scout_by_uuid,
                scout_by_key=scout_by_key,
            ):
                result.rejected.append(
                    BatchValidationIssue(
                        index=idx,
                        reason="unknown_tail_endpoint",
                        relationship=rel,
                    )
                )
                continue
        if not _endpoint_exists(rel.tip, known_uuids=known_uuids, known_keys=known_keys):
            if not _admit_endpoint(
                rel.tip,
                known_uuids=known_uuids,
                known_keys=known_keys,
                admitted=kept_nodes,
                scout_by_uuid=scout_by_uuid,
                scout_by_key=scout_by_key,
            ):
                result.rejected.append(
                    BatchValidationIssue(
                        index=idx,
                        reason="unknown_tip_endpoint",
                        relationship=rel,
                    )
                )
                continue
        ok_span, span_reason = _span_in_bounds(rel, source_text)
        if not ok_span:
            result.rejected.append(
                BatchValidationIssue(
                    index=idx,
                    reason=span_reason,
                    relationship=rel,
                )
            )
            continue
        ok_event, event_reason = ensure_event_descriptions(
            rel,
            scout_by_uuid=scout_by_uuid,
            scout_by_key=scout_by_key,
        )
        if not ok_event:
            result.rejected.append(
                BatchValidationIssue(
                    index=idx,
                    reason=event_reason,
                    relationship=rel,
                )
            )
            continue
        _ensure_endpoint_uuid(
            rel.tail, scout_by_uuid=scout_by_uuid, scout_by_key=scout_by_key
        )
        _ensure_endpoint_uuid(
            rel.tip, scout_by_uuid=scout_by_uuid, scout_by_key=scout_by_key
        )
        result.accepted.append(rel)

    return result


def event_leg_incomplete(
    relationships: Sequence[BatchRelationship],
    scout_entities: Sequence[Any],
) -> bool:
    event_uuids = set()
    for entity in scout_entities:
        typ = getattr(entity, "type", None) or (entity.get("type") if isinstance(entity, dict) else None)
        uuid = getattr(entity, "uuid", None) or (entity.get("uuid") if isinstance(entity, dict) else None)
        if typ and str(typ).strip().lower() == "event" and uuid:
            event_uuids.add(str(uuid))
    if not event_uuids:
        return False
    touched: set[str] = set()
    for rel in relationships:
        for endpoint in (rel.tail, rel.tip):
            if endpoint.uuid and endpoint.uuid in event_uuids:
                touched.add(endpoint.uuid)
            elif (endpoint.type or "").strip().lower() == "event":
                for entity in scout_entities:
                    name = getattr(entity, "name", None) or (
                        entity.get("name") if isinstance(entity, dict) else None
                    )
                    uuid = getattr(entity, "uuid", None) or (
                        entity.get("uuid") if isinstance(entity, dict) else None
                    )
                    if (
                        uuid
                        and name
                        and normalize_loose(name) == normalize_loose(endpoint.name)
                    ):
                        touched.add(str(uuid))
    return bool(event_uuids - touched)


# Re-export for callers/tests that previously imported align helpers here.
__all__ = [
    "BatchEndpoint",
    "BatchRelationship",
    "BatchExtractResponse",
    "BatchValidationIssue",
    "BatchValidationResult",
    "validate_batch_extract",
    "event_leg_incomplete",
    "ensure_event_descriptions",
    "normalize_loose",
    "find_span_offsets",
    "align_span",
]

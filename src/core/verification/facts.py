"""Expert verification state for graph facts.

Verification is orthogonal to temporal validity. Do not read or write
valid_from, valid_to, superseded_by, invalid_at, or other temporal fields here.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Optional, Sequence
from uuid import uuid4

from pydantic import BaseModel, Field

VerificationState = Literal["unverified", "verified", "rejected"]

UNVERIFIED: VerificationState = "unverified"
VERIFIED: VerificationState = "verified"
REJECTED: VerificationState = "rejected"
VERIFICATION_STATES: tuple[VerificationState, ...] = (UNVERIFIED, VERIFIED, REJECTED)

STATE_FIELD = "verification_state"
BY_FIELD = "verified_by"
AT_FIELD = "verified_at"
SOURCE_FIELD = "verification_source"
NOTE_FIELD = "verification_note"
HISTORY_FIELD = "verification_history"

VERIFICATION_PROPERTY_FIELDS = (
    STATE_FIELD,
    BY_FIELD,
    AT_FIELD,
    SOURCE_FIELD,
    NOTE_FIELD,
    HISTORY_FIELD,
)

DEFAULT_VERIFIED_RANK_BOOST = 0.15

_include_rejected: ContextVar[bool] = ContextVar(
    "fact_include_rejected", default=False
)


class FactVerification(BaseModel):
    state: VerificationState = UNVERIFIED
    verified_by: Optional[str] = None
    verified_at: Optional[str] = None
    verification_source: Optional[str] = None
    note: Optional[str] = None
    fact_id: Optional[str] = None
    kind: Optional[Literal["relationship", "node"]] = None


class FactHistoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    from_state: VerificationState
    to_state: VerificationState
    verified_by: Optional[str] = None
    verified_at: str
    verification_source: Optional[str] = None
    note: Optional[str] = None


class FactRecord(BaseModel):
    kind: Literal["relationship", "node"]
    fact_id: str
    name: str = ""
    description: str = ""
    verification: FactVerification
    history: list[FactHistoryEntry] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    subject_uuid: Optional[str] = None
    subject_name: Optional[str] = None
    subject_labels: list[str] = Field(default_factory=list)
    object_uuid: Optional[str] = None
    object_name: Optional[str] = None
    object_labels: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


@contextmanager
def verification_scope(*, include_rejected: bool = False):
    token = _include_rejected.set(bool(include_rejected))
    try:
        yield
    finally:
        _include_rejected.reset(token)


def include_rejected_enabled() -> bool:
    return bool(_include_rejected.get())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_state(value: Any) -> VerificationState:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in VERIFICATION_STATES:
            return lowered  # type: ignore[return-value]
    return UNVERIFIED


def _mapping(entity: Any) -> dict[str, Any]:
    if entity is None:
        return {}
    if isinstance(entity, dict):
        props = dict(entity.get("properties") or {})
        merged = dict(entity)
        merged.update(props)
        return merged
    props = dict(getattr(entity, "properties", None) or {})
    merged = dict(props)
    for key in VERIFICATION_PROPERTY_FIELDS + ("uuid", "name", "description"):
        value = getattr(entity, key, None)
        if value is not None and key not in merged:
            merged[key] = value
    return merged


def extract_verification(entity: Any) -> FactVerification:
    data = _mapping(entity)
    return FactVerification(
        state=normalize_state(data.get(STATE_FIELD)),
        verified_by=_optional_str(data.get(BY_FIELD)),
        verified_at=_optional_str(data.get(AT_FIELD)),
        verification_source=_optional_str(data.get(SOURCE_FIELD)),
        note=_optional_str(data.get(NOTE_FIELD)),
        fact_id=_optional_str(data.get("uuid")),
    )


def verification_extras(entity: Any) -> dict[str, Any]:
    verification = extract_verification(entity)
    payload: dict[str, Any] = {STATE_FIELD: verification.state}
    if verification.verified_by:
        payload[BY_FIELD] = verification.verified_by
    if verification.verified_at:
        payload[AT_FIELD] = verification.verified_at
    if verification.verification_source:
        payload[SOURCE_FIELD] = verification.verification_source
    if verification.note:
        payload[NOTE_FIELD] = verification.note
    return payload


def entity_is_rejected(entity: Any) -> bool:
    return extract_verification(entity).state == REJECTED


def entity_is_verified(entity: Any) -> bool:
    return extract_verification(entity).state == VERIFIED


def fact_tuple_is_rejected(*parts: Any) -> bool:
    for part in parts:
        if part is not None and entity_is_rejected(part):
            return True
    return False


def fact_tuple_is_verified(*parts: Any) -> bool:
    for part in parts:
        if part is not None and entity_is_verified(part):
            return True
    return False


def fact_pair_allowed(left: Any, right: Any | None = None) -> bool:
    if include_rejected_enabled():
        return True
    return not fact_tuple_is_rejected(left, right)


def combined_verification(*parts: Any) -> FactVerification:
    extracted = [extract_verification(part) for part in parts if part is not None]
    if not extracted:
        return FactVerification()
    rejected = next((item for item in extracted if item.state == REJECTED), None)
    if rejected:
        return rejected
    verified = next((item for item in extracted if item.state == VERIFIED), None)
    if verified:
        return verified
    return extracted[0]


def parse_history(raw: Any) -> list[FactHistoryEntry]:
    entries: list[Any]
    if raw is None or raw == "":
        entries = []
    elif isinstance(raw, list):
        entries = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = [raw]
        entries = parsed if isinstance(parsed, list) else [parsed]
    else:
        entries = [raw]

    history: list[FactHistoryEntry] = []
    for item in entries:
        payload = item
        if isinstance(item, str):
            try:
                payload = json.loads(item)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        if not isinstance(payload, dict):
            continue
        try:
            history.append(
                FactHistoryEntry(
                    id=str(payload.get("id") or uuid4()),
                    from_state=normalize_state(payload.get("from_state")),
                    to_state=normalize_state(payload.get("to_state")),
                    verified_by=_optional_str(payload.get("verified_by")),
                    verified_at=str(
                        payload.get("verified_at") or utc_now_iso()
                    ),
                    verification_source=_optional_str(
                        payload.get("verification_source")
                    ),
                    note=_optional_str(payload.get("note")),
                )
            )
        except Exception:
            continue
    return history


def serialize_history(history: Sequence[FactHistoryEntry]) -> list[str]:
    return [
        json.dumps(entry.model_dump(mode="json"), separators=(",", ":"))
        for entry in history
    ]


def append_history_entry(
    history: Sequence[FactHistoryEntry],
    *,
    from_state: VerificationState,
    to_state: VerificationState,
    verified_by: str | None,
    verified_at: str,
    verification_source: str | None,
    note: str | None,
) -> list[FactHistoryEntry]:
    updated = list(history)
    updated.append(
        FactHistoryEntry(
            from_state=from_state,
            to_state=to_state,
            verified_by=verified_by,
            verified_at=verified_at,
            verification_source=verification_source,
            note=note,
        )
    )
    return updated


def verification_write_properties(
    *,
    state: VerificationState,
    verified_by: str | None,
    verified_at: str | None,
    verification_source: str | None,
    note: str | None,
    history: Sequence[FactHistoryEntry],
) -> dict[str, Any]:
    return {
        STATE_FIELD: state,
        BY_FIELD: verified_by,
        AT_FIELD: verified_at,
        SOURCE_FIELD: verification_source,
        NOTE_FIELD: note,
        HISTORY_FIELD: serialize_history(history),
    }


def hydrate_entity_verification(entity: Any) -> Any:
    verification = extract_verification(entity)
    if hasattr(entity, STATE_FIELD):
        setattr(entity, STATE_FIELD, verification.state)
        setattr(entity, BY_FIELD, verification.verified_by)
        setattr(entity, AT_FIELD, verification.verified_at)
        setattr(entity, SOURCE_FIELD, verification.verification_source)
        setattr(entity, NOTE_FIELD, verification.note)
    return entity


def predicate_from_properties(
    *,
    name: str,
    description: str = "",
    properties: dict[str, Any] | None = None,
    direction: str = "neutral",
    **kwargs: Any,
):
    from src.constants.kg import Predicate

    props = dict(properties or {})
    verification = extract_verification(props)
    uuid = str(kwargs.pop("uuid", None) or props.get("uuid") or "")
    return Predicate(
        uuid=uuid,
        name=name,
        description=description or "",
        direction=direction,  # type: ignore[arg-type]
        properties=props,
        deprecated=bool(props.get("deprecated", False) or kwargs.get("deprecated", False)),
        verification_state=verification.state,
        verified_by=verification.verified_by,
        verified_at=verification.verified_at,
        verification_source=verification.verification_source,
        verification_note=verification.note,
        **kwargs,
    )


def verified_rank_boost() -> float:
    try:
        from src.config import config

        value = getattr(config, "verified_fact_rank_boost", DEFAULT_VERIFIED_RANK_BOOST)
        return max(0.0, float(value))
    except Exception:
        return DEFAULT_VERIFIED_RANK_BOOST


def apply_verified_boost(score: float, entity: Any) -> float:
    if entity_is_verified(entity):
        return float(score) + verified_rank_boost()
    return float(score)


def filter_relationship_results(results: Iterable[Any]) -> list[Any]:
    if include_rejected_enabled():
        return list(results)
    kept: list[Any] = []
    for item in results:
        predicate = getattr(item, "predicate", None)
        if predicate is None and isinstance(item, (tuple, list)) and len(item) >= 2:
            predicate = item[1]
        if predicate is not None and entity_is_rejected(predicate):
            continue
        kept.append(item)
    return kept


def record_from_graph_row(row: dict[str, Any]) -> FactRecord:
    kind = "relationship" if row.get("kind") == "relationship" else "node"
    props = dict(row.get("properties") or {})
    fact_id = str(row.get("uuid") or props.get("uuid") or "")
    verification = extract_verification({**props, "uuid": fact_id})
    verification.fact_id = fact_id
    verification.kind = kind
    return FactRecord(
        kind=kind,
        fact_id=fact_id,
        name=str(row.get("name") or props.get("name") or ""),
        description=str(row.get("description") or props.get("description") or ""),
        verification=verification,
        history=parse_history(props.get(HISTORY_FIELD)),
        properties=props,
        subject_uuid=_optional_str(row.get("subject_uuid")),
        subject_name=_optional_str(row.get("subject_name")),
        subject_labels=list(row.get("subject_labels") or []),
        object_uuid=_optional_str(row.get("object_uuid")),
        object_name=_optional_str(row.get("object_name")),
        object_labels=list(row.get("object_labels") or []),
        labels=list(row.get("labels") or []),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

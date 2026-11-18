from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from src.core.intent.models import FactSource, FactStatus, IntentFact
from src.core.intent.parse import fact_statement
from src.core.saving.validity import SUPERSEDED_BY, VALID_FROM, VALID_TO, is_valid_at, parse_instant
from src.core.verification.facts import REJECTED, UNVERIFIED, VERIFIED, extract_verification, normalize_state


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    props = getattr(value, "properties", None)
    if isinstance(props, Mapping):
        merged = dict(props)
        nested = props.get("properties")
        if isinstance(nested, Mapping):
            merged = {**nested, **{k: v for k, v in props.items() if k != "properties"}}
        return merged
    return {}


def _optional(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def fact_status(
    fields: Mapping[str, Any] | None,
    *,
    verification_state: str | None = None,
    as_of: datetime | None = None,
    deprecated: bool = False,
) -> FactStatus:
    source = _mapping(fields)
    state = normalize_state(verification_state or source.get("verification_state"))
    if state == REJECTED:
        return "rejected"
    closed = bool(
        source.get("superseded_by")
        or source.get("superseded_at")
        or source.get(VALID_TO)
        or source.get("invalid_at")
        or deprecated
        or source.get("deprecated") is True
    )
    if closed:
        return "superseded"
    if not is_valid_at(source, as_of, deprecated=False):
        return "superseded"
    if state == VERIFIED:
        return "verified"
    return "unverified"


def source_from_fields(fields: Mapping[str, Any] | None, writer: str | None = None) -> FactSource:
    source = _mapping(fields)
    return FactSource(
        doc=_optional(source.get("source_doc") or source.get("doc") or source.get("source")),
        url=_optional(source.get("source_url") or source.get("url")),
        source_key=_optional(source.get("source_key")),
        writer=_optional(writer or source.get("writer")),
    )


def intent_fact_from_row(
    row: Mapping[str, Any] | None,
    *,
    as_of: datetime | None = None,
    contradicted_by: list[str] | None = None,
) -> IntentFact:
    data = dict(row or {})
    props = _mapping(data.get("properties") or data)
    verification = extract_verification({**props, "uuid": data.get("uuid") or props.get("uuid")})
    deprecated = bool(data.get("deprecated") or props.get("deprecated"))
    status = fact_status(
        props,
        verification_state=verification.state,
        as_of=as_of,
        deprecated=deprecated,
    )
    subject = str(data.get("subject_name") or props.get("subject_name") or "")
    predicate = str(data.get("name") or props.get("rel_type") or props.get("name") or "")
    obj = str(data.get("object_name") or props.get("object_name") or "")
    text = str(props.get("fact_text") or data.get("description") or props.get("description") or "")
    if not text:
        text = fact_statement(subject, predicate, obj)
    writer = _optional(props.get("writer"))
    retired = _optional(props.get("retired_reason"))
    if status == "rejected":
        retired = retired or _optional(verification.note)
    valid_at = _optional(props.get(VALID_FROM) or props.get("valid_at"))
    invalid_at = _optional(props.get(VALID_TO) or props.get("invalid_at") or props.get("superseded_at"))
    if status in {"verified", "unverified"}:
        retired = None
    return IntentFact(
        fact_id=str(data.get("uuid") or props.get("uuid") or ""),
        text=text,
        status=status,
        subject=subject,
        predicate=predicate,
        object=obj,
        contradicted_by=list(contradicted_by or []),
        valid_at=valid_at,
        invalid_at=invalid_at,
        superseded_by=_optional(props.get(SUPERSEDED_BY)),
        source=source_from_fields(props, writer),
        writer=writer,
        retired_reason=retired,
        verification_state=verification.state or UNVERIFIED,
        kind="relationship" if data.get("kind") != "node" else "node",
    )


def intent_fact_from_chunk(
    chunk: Any,
    *,
    as_of: datetime | None = None,
) -> IntentFact:
    meta = _mapping(getattr(chunk, "metadata", None))
    status = fact_status(meta, verification_state=meta.get("verification_state"), as_of=as_of)
    writer = _optional(meta.get("writer"))
    retired = _optional(meta.get("retired_reason"))
    if status in {"verified", "unverified"}:
        retired = None
    return IntentFact(
        fact_id=str(getattr(chunk, "id", "") or meta.get("fact_id") or ""),
        text=str(getattr(chunk, "text", "") or ""),
        status=status,
        subject=str(meta.get("subject") or ""),
        predicate=str(meta.get("predicate") or ""),
        object=str(meta.get("object") or ""),
        contradicted_by=[],
        valid_at=_optional(meta.get(VALID_FROM) or meta.get("valid_at")),
        invalid_at=_optional(meta.get(VALID_TO) or meta.get("invalid_at") or meta.get("superseded_at")),
        superseded_by=_optional(meta.get(SUPERSEDED_BY)),
        source=source_from_fields(meta, writer),
        writer=writer,
        retired_reason=retired,
        verification_state=normalize_state(meta.get("verification_state")),
        kind="passage",
    )


def current_at(fact: IntentFact, as_of: datetime | None) -> bool:
    fields = {
        VALID_FROM: fact.valid_at,
        VALID_TO: fact.invalid_at,
        SUPERSEDED_BY: fact.superseded_by,
        "retired_reason": fact.retired_reason,
        "deprecated": fact.status == "superseded",
        "verification_state": fact.verification_state,
    }
    if fact.status == "rejected":
        return False
    return is_valid_at(fields, as_of, deprecated=fact.status == "superseded")

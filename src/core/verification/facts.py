"""Expert verification state for graph facts.

Verification is orthogonal to temporal validity. Do not read or write
valid_from, valid_to, superseded_by, invalid_at, or other temporal fields here.

Rejected-content filtering (default `include_rejected=false`):

- Filtered: graph retrieve/search/traverse, event-centric neighbors, MCP
  `search_memory` rows whose serialized graph entities have
  `verification_state=rejected`, and chunk/observation hits whose *only*
  graph support (via `source_chunk_ids` / `resource_id`) is rejected
  *verifiable* facts — EVENT nodes or relationships. Ordinary PERSON/ORG
  nodes are not counted as standalone support.
- Not filtered: raw `search_memory` aggregations with no graph entities in
  the row, chunks/observations that cite no graph facts, vector hits whose
  metadata has no resolvable source chunk, and chunks that still have a
  non-rejected EVENT or relationship citation. Pass `include_rejected=true`
  to keep rejected support and leave verification fields visible.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Optional, Sequence
from uuid import uuid4

from pydantic import BaseModel, Field

from src.utils.serialization.data import always_dict

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


def is_reserved_verification_key(key: Any) -> bool:
    text = str(key or "").strip()
    if not text:
        return False
    if text in VERIFICATION_PROPERTY_FIELDS or text.lower() in VERIFICATION_PROPERTY_FIELDS:
        return True
    words = text.split()
    cleaned = words[0].lower() + "".join(word.capitalize() for word in words[1:])
    return cleaned in VERIFICATION_PROPERTY_FIELDS


def without_reserved_verification_keys(data: dict[str, Any] | None) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in dict(data or {}).items():
        if is_reserved_verification_key(key):
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                nested_key: nested_value
                for nested_key, nested_value in value.items()
                if not is_reserved_verification_key(nested_key)
            }
        else:
            cleaned[key] = value
    return cleaned

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
    interview_id: Optional[str] = None
    topic_id: Optional[str] = None
    source_key: Optional[str] = None
    source_chunk_id: Optional[str] = None
    source: Optional[str] = None


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


def node_is_verifiable_event(labels: Any) -> bool:
    return any(str(label).strip().upper() == "EVENT" for label in (labels or []))


def _property_map(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        coerced = always_dict(raw)
        return coerced if isinstance(coerced, dict) else {}
    return {}


def _entity_labels(entity: Any) -> list[Any]:
    if entity is None:
        return []
    if isinstance(entity, dict):
        return list(entity.get("labels") or [])
    return list(getattr(entity, "labels", None) or [])


def _flatten_properties(data: dict[str, Any]) -> dict[str, Any]:
    merged = dict(data)
    nested = _property_map(merged.get("properties"))
    if not nested:
        return merged
    for key, value in nested.items():
        if is_reserved_verification_key(key) and merged.get(key) not in (None, ""):
            continue
        merged[key] = value
    return merged


def _citation_layers(entity: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if entity is None:
        return {}, {}
    if isinstance(entity, dict):
        outer = dict(entity)
    else:
        outer = _property_map(getattr(entity, "properties", None))
        for key in ("source_chunk_ids", "source_chunk_id"):
            value = getattr(entity, key, None)
            if value not in (None, "", []) and outer.get(key) in (None, "", []):
                outer[key] = value
    return outer, _property_map(outer.get("properties"))


def _mapping(entity: Any) -> dict[str, Any]:
    if entity is None:
        return {}
    if isinstance(entity, dict):
        return _flatten_properties(entity)
    props = _property_map(getattr(entity, "properties", None))
    merged = _flatten_properties(props)
    for key in VERIFICATION_PROPERTY_FIELDS + ("uuid", "name", "description"):
        if merged.get(key) not in (None, ""):
            continue
        value = getattr(entity, key, None)
        if value is not None:
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


def coerce_history_entry(entry: Any) -> FactHistoryEntry | None:
    if entry is None or entry == "":
        return None
    if isinstance(entry, FactHistoryEntry):
        return entry
    if isinstance(entry, str):
        try:
            entry = json.loads(entry)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if isinstance(entry, dict):
        parsed = parse_history([entry])
        return parsed[0] if parsed else None
    return None


def serialize_history_entry(entry: Any) -> str | None:
    coerced = coerce_history_entry(entry)
    if coerced is None:
        return None
    return serialize_history([coerced])[0]


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


def _relationship_parts(item: Any) -> tuple[Any, Any, Any]:
    subject = getattr(item, "subject", None)
    predicate = getattr(item, "predicate", None)
    obj = getattr(item, "object", None)
    if predicate is None and isinstance(item, (tuple, list)) and len(item) >= 2:
        subject = item[0] if len(item) >= 1 else None
        predicate = item[1]
        obj = item[2] if len(item) >= 3 else None
    return subject, predicate, obj


def filter_relationship_results(results: Iterable[Any]) -> list[Any]:
    if include_rejected_enabled():
        return list(results)
    kept: list[Any] = []
    for item in results:
        if fact_tuple_is_rejected(*_relationship_parts(item)):
            continue
        kept.append(item)
    return kept


def _iter_relationship_values(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def neighborhood_row_is_rejected(row: Any) -> bool:
    if include_rejected_enabled():
        return False
    if not isinstance(row, dict):
        return entity_is_rejected(row)
    if entity_is_rejected(row.get("node")) or entity_is_rejected(row.get("related_nodes")):
        return True
    return any(
        entity_is_rejected(rel)
        for rel in _iter_relationship_values(row.get("relationships"))
    )


def normalize_listing_filter(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def extract_fact_listing_fields(row: dict[str, Any]) -> dict[str, Optional[str]]:
    props = _flatten_properties(_property_map(row.get("properties")))
    interview_id = _optional_str(
        row.get("interview_id")
        or props.get("interview_id")
        or props.get("interviewId")
    )
    topic_id = _optional_str(
        row.get("topic_id") or props.get("topic_id") or props.get("topicId")
    )
    source_key = _optional_str(
        props.get("source_key") or row.get("source_key") or props.get("source")
    )
    source_chunk_id = _optional_str(
        props.get("source_chunk_id") or row.get("source_chunk_id")
    )
    if not source_chunk_id:
        ids = props.get("source_chunk_ids") or row.get("source_chunk_ids") or []
        if isinstance(ids, (list, tuple)) and ids:
            source_chunk_id = _optional_str(ids[-1])
    return {
        "interview_id": interview_id,
        "topic_id": topic_id,
        "source_key": source_key,
        "source_chunk_id": source_chunk_id,
        "source": source_key,
    }


def fact_row_matches_listing_filters(
    row: dict[str, Any],
    *,
    interview_id: str | None = None,
    topic_id: str | None = None,
    entity_id: str | None = None,
    source: str | None = None,
) -> bool:
    interview_id = normalize_listing_filter(interview_id)
    topic_id = normalize_listing_filter(topic_id)
    entity_id = normalize_listing_filter(entity_id)
    source = normalize_listing_filter(source)
    fields = extract_fact_listing_fields(row)
    if interview_id and fields["interview_id"] != interview_id:
        return False
    if topic_id and fields["topic_id"] != topic_id:
        return False
    if source:
        candidates = {
            fields["source_key"],
            fields["source_chunk_id"],
            fields["source"],
        }
        if source not in {item for item in candidates if item}:
            return False
    if entity_id:
        props = _flatten_properties(_property_map(row.get("properties")))
        ids = {
            _optional_str(row.get("uuid")),
            _optional_str(row.get("subject_uuid")),
            _optional_str(row.get("object_uuid")),
            _optional_str(props.get("entity_id")),
        }
        if entity_id not in {item for item in ids if item}:
            return False
    return True


def record_from_graph_row(row: dict[str, Any]) -> FactRecord:
    kind = "relationship" if row.get("kind") == "relationship" else "node"
    props = dict(row.get("properties") or {})
    fact_id = str(row.get("uuid") or props.get("uuid") or "")
    verification = extract_verification({**props, "uuid": fact_id})
    verification.fact_id = fact_id
    verification.kind = kind
    listing = extract_fact_listing_fields({**row, "properties": props})
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
        interview_id=listing["interview_id"],
        topic_id=listing["topic_id"],
        source_key=listing["source_key"],
        source_chunk_id=listing["source_chunk_id"],
        source=listing["source"],
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _record_mapping(record: Any) -> Any:
    if record is None:
        return None
    if isinstance(record, dict):
        return record
    data = getattr(record, "data", None)
    if callable(data):
        try:
            return data()
        except Exception:
            pass
    try:
        return dict(record)
    except Exception:
        return record


def mapping_contains_rejected(value: Any) -> bool:
    if value is None or include_rejected_enabled():
        return False
    if isinstance(value, dict):
        if normalize_state(value.get(STATE_FIELD)) == REJECTED:
            return True
        nested = value.get("properties")
        if isinstance(nested, dict) and normalize_state(nested.get(STATE_FIELD)) == REJECTED:
            return True
        return any(mapping_contains_rejected(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(mapping_contains_rejected(item) for item in value)
    return entity_is_rejected(value) if not isinstance(value, (str, bytes, int, float, bool)) else False


class _FilteredRawResult:
    def __init__(self, records: list[Any], keys: Any, truncated: bool):
        self.records = records
        self._keys = keys
        self.truncated = truncated

    def keys(self):
        keys = self._keys
        if callable(keys):
            return keys()
        if keys is None:
            return []
        return keys


def filter_raw_operation_result(result: Any, *, include_rejected: bool = False) -> Any:
    if include_rejected or result is None:
        return result
    if hasattr(result, "records"):
        kept = [
            record
            for record in (result.records or [])
            if not mapping_contains_rejected(_record_mapping(record))
        ]
        keys = getattr(result, "keys", None)
        truncated = bool(getattr(result, "truncated", False))
        return _FilteredRawResult(kept, keys, truncated)
    if isinstance(result, list):
        return [row for row in result if not mapping_contains_rejected(row)]
    if isinstance(result, dict):
        records = result.get("records")
        if isinstance(records, list):
            filtered = dict(result)
            filtered["records"] = [
                row for row in records if not mapping_contains_rejected(row)
            ]
            return filtered
        if mapping_contains_rejected(result):
            return {}
    return result


def _chunk_ids_from_entity(entity: Any) -> list[str]:
    outer, nested = _citation_layers(entity)
    citations = list(outer.get("source_chunk_ids") or []) + list(
        nested.get("source_chunk_ids") or []
    )
    ids: list[str] = []
    for value in citations + [outer.get("source_chunk_id"), nested.get("source_chunk_id")]:
        chunk_id = str(value).strip() if value not in (None, "") else ""
        if chunk_id and chunk_id not in ids:
            ids.append(chunk_id)
    return ids


def _is_verifiable_support(entity: Any) -> bool:
    labels = _entity_labels(entity)
    if not labels:
        return True
    return node_is_verifiable_event(labels)


def _record_chunk_support(
    support: dict[str, list[bool]],
    entity: Any,
    rejected: bool,
) -> None:
    for chunk_id in _chunk_ids_from_entity(entity):
        support.setdefault(chunk_id, []).append(rejected)


def chunk_ids_with_only_rejected_support(
    nodes: Sequence[Any] | None,
    triples: Sequence[Any] | None,
) -> set[str]:
    support: dict[str, list[bool]] = {}
    for node in nodes or []:
        if not _is_verifiable_support(node):
            continue
        _record_chunk_support(support, node, entity_is_rejected(node))
    for triple in triples or []:
        parts = triple if isinstance(triple, (list, tuple)) else (triple,)
        for part in parts:
            if not _is_verifiable_support(part):
                continue
            _record_chunk_support(support, part, entity_is_rejected(part))
    return {
        chunk_id
        for chunk_id, flags in support.items()
        if flags and all(flags)
    }


def resource_id_of(item: Any) -> str | None:
    if item is None:
        return None
    if isinstance(item, dict):
        value = item.get("resource_id") or item.get("id")
    else:
        value = getattr(item, "resource_id", None) or getattr(item, "id", None)
    text = str(value).strip() if value not in (None, "") else ""
    return text or None


def filter_items_with_rejected_support(
    items: Sequence[Any],
    rejected_chunk_ids: set[str],
) -> list[Any]:
    if include_rejected_enabled() or not rejected_chunk_ids:
        return list(items)
    kept: list[Any] = []
    for item in items:
        resource_id = resource_id_of(item)
        if resource_id and resource_id in rejected_chunk_ids:
            continue
        metadata = None
        if isinstance(item, dict):
            metadata = item.get("metadata")
        else:
            metadata = getattr(item, "metadata", None)
        if isinstance(metadata, dict) and normalize_state(metadata.get(STATE_FIELD)) == REJECTED:
            continue
        kept.append(item)
    return kept

import hashlib
from typing import Any, Optional

from src.utils.dates import normalize_date_string


def stable_uuid(*parts: Optional[str]) -> str:
    material = "|".join((part or "").strip().lower() for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def stable_node_id(
    name: Optional[str],
    entity_type: Optional[str],
    happened_at: Optional[str] = None,
    supplied_uuid: Optional[str] = None,
) -> str:
    if supplied_uuid and str(supplied_uuid).strip():
        return str(supplied_uuid).strip()
    normalized_type = (entity_type or "").strip().lower()
    parts = [(name or "").strip().lower(), normalized_type]
    if normalized_type == "event":
        parts.append(normalize_date_string(happened_at) or "")
    return stable_uuid("node", *parts)


def stable_relationship_id(
    tail_uuid: str,
    predicate: str,
    tip_uuid: str,
    flow_key: Optional[str] = None,
    amount: Optional[Any] = None,
) -> str:
    from src.core.saving.numeric_literals import amount_identity_token

    parts = [
        "rel",
        tail_uuid or "",
        (predicate or "").strip().upper(),
        tip_uuid or "",
    ]
    if flow_key:
        parts.append(flow_key)
    token = amount_identity_token(amount)
    if token:
        parts.append(token)
    return stable_uuid(*parts)


def stable_flow_key(
    event_uuid: Optional[str] = None,
    event_name: Optional[str] = None,
    event_type: Optional[str] = None,
    happened_at: Optional[str] = None,
    supplied: Optional[str] = None,
) -> str:
    if supplied and str(supplied).strip():
        return str(supplied).strip()
    if event_uuid and str(event_uuid).strip():
        return str(event_uuid).strip()
    return stable_uuid(
        "flow",
        event_name,
        event_type,
        normalize_date_string(happened_at) if happened_at else None,
    )


def merge_source_chunk_ids(*sources: Any) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for source in sources:
        if source is None:
            continue
        if isinstance(source, str):
            values = [source]
        elif isinstance(source, (list, tuple, set)):
            values = list(source)
        else:
            values = [source]
        for value in values:
            chunk_id = str(value).strip() if value is not None else ""
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(chunk_id)
    return merged


def stamp_provenance(
    properties: Optional[dict],
    *,
    source_chunk_id: Optional[str] = None,
    source_timestamp: Optional[str] = None,
    source_key: Optional[str] = None,
    existing_properties: Optional[dict] = None,
) -> dict:
    props = dict(properties or {})
    existing = existing_properties or {}
    chunk_ids = merge_source_chunk_ids(
        existing.get("source_chunk_ids"),
        existing.get("source_chunk_id"),
        props.get("source_chunk_ids"),
        props.get("source_chunk_id"),
        source_chunk_id,
    )
    if chunk_ids:
        props["source_chunk_ids"] = chunk_ids
    props.pop("source_chunk_id", None)
    timestamp = source_timestamp or props.get("source_timestamp") or existing.get(
        "source_timestamp"
    )
    if timestamp:
        props["source_timestamp"] = timestamp
    if source_key and str(source_key).strip():
        key = str(source_key).strip()
    else:
        key = str(props.get("source_key") or existing.get("source_key") or "").strip()
    if key:
        props["source_key"] = key
    return props

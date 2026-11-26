from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

from src.core.saving.validity import close_validity


def cited_chunk_ids(props: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(props, Mapping):
        return set()
    raw = props.get("source_chunk_ids") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(item) for item in raw if item}


def chunk_is_current(chunk: Any) -> bool:
    meta = getattr(chunk, "metadata", None) or {}
    if not isinstance(meta, Mapping):
        return True
    return not (meta.get("superseded_at") or meta.get("valid_to"))


def live_chunk_ids(data: Any, brain_id: str, chunk_ids: Iterable[str]) -> set[str]:
    wanted = {str(item) for item in chunk_ids if item}
    if not wanted:
        return set()
    getter = getattr(data, "get_text_chunks_by_ids", None) if data is not None else None
    if not callable(getter):
        return wanted
    try:
        chunks, _observations = getter(list(wanted), False, brain_id)
    except TypeError:
        chunks, _observations = getter(
            ids=list(wanted), with_observations=False, brain_id=brain_id
        )
    live: set[str] = set()
    for chunk in chunks or []:
        chunk_id = str(getattr(chunk, "id", "") or "")
        if chunk_id in wanted and chunk_is_current(chunk):
            live.add(chunk_id)
    return live


def fact_has_remaining_live_sources(
    props: Mapping[str, Any] | None,
    closing_chunk_ids: Iterable[str],
    data: Any,
    brain_id: str,
) -> bool:
    cited = cited_chunk_ids(props)
    leftover = cited - {str(item) for item in closing_chunk_ids if item}
    if not leftover:
        return False
    return bool(live_chunk_ids(data, brain_id, leftover))


def props_map(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        nested = value.get("properties")
        if isinstance(nested, Mapping):
            merged = dict(nested)
            merged.update({k: v for k, v in value.items() if k != "properties"})
            return merged
        return dict(value)
    props = getattr(value, "properties", None)
    return dict(props) if isinstance(props, Mapping) else {}


def entity_id(entity: Any) -> str:
    if entity is None:
        return ""
    props = props_map(entity)
    return str(getattr(entity, "uuid", "") or props.get("uuid") or "")


def close_orphaned_citing_facts(
    graph: Any,
    brain_id: str,
    chunk_ids: list[str],
    closed_at: str,
    *,
    data: Any | None = None,
    reason: str = "source document deleted",
) -> list[str]:
    if not graph or not chunk_ids:
        return []
    citing = getattr(graph, "nodes_and_edges_citing", None)
    if not callable(citing):
        return []
    closing = {str(item) for item in chunk_ids if item}
    nodes, triples = citing(brain_id, list(closing), [])
    closed: list[str] = []
    for triple in triples or []:
        predicate = triple[1] if len(triple) > 1 else None
        if fact_has_remaining_live_sources(props_map(predicate), closing, data, brain_id):
            continue
        fact_id = entity_id(predicate)
        if _close_open_entity(
            graph, brain_id, fact_id, predicate, "relationship", closed_at, reason
        ):
            closed.append(fact_id)
    from src.core.verification.facts import node_is_verifiable_event

    for node in nodes or []:
        if not node_is_verifiable_event(getattr(node, "labels", None)):
            continue
        if fact_has_remaining_live_sources(props_map(node), closing, data, brain_id):
            continue
        fact_id = entity_id(node)
        if _close_open_entity(graph, brain_id, fact_id, node, "node", closed_at, reason):
            closed.append(fact_id)
    return closed


def _is_open(props: Mapping[str, Any]) -> bool:
    return not (props.get("invalid_at") or props.get("valid_to"))


def _close_open_entity(
    graph: Any,
    brain_id: str,
    fact_id: str,
    entity: Any,
    updating: str,
    closed_at: str,
    reason: str,
) -> bool:
    if not fact_id:
        return False
    props = props_map(entity)
    if not _is_open(props):
        return False
    closed_props = close_validity(props, closed_at=closed_at, reason=reason)
    patch = {
        "invalid_at": closed_props.get("invalid_at") or closed_at,
        "deprecated": True,
        "valid_to": closed_props.get("valid_to") or closed_at,
        "retired_reason": closed_props.get("retired_reason") or reason,
    }
    if updating == "relationship":
        from src.core.intent.retire import apply_open_update

        return apply_open_update(graph, fact_id, brain_id, patch)
    updater = getattr(graph, "update_properties", None)
    if not callable(updater):
        return False
    try:
        updated = updater(
            fact_id,
            "node",
            brain_id=brain_id,
            new_properties=patch,
        )
    except TypeError:
        updated = updater(fact_id, "node", brain_id, patch, [])
    return updated is not None

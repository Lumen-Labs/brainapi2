from __future__ import annotations

from typing import Any, Literal

from src.constants.kg import Node, Predicate

MAX_TRAVERSE_DEPTH = 5
MAX_TRAVERSE_HOPS = 100


def _entity_field(entity: Node | Predicate | dict, field: str, default: Any = None) -> Any:
    if isinstance(entity, (Node, Predicate)):
        return getattr(entity, field, default)
    return entity.get(field, default)


def _serialize_node(node: Node | dict) -> dict:
    if isinstance(node, Node):
        return node.model_dump(mode="json")
    return dict(node)


def _serialize_predicate(predicate: Predicate | dict) -> dict:
    if isinstance(predicate, Predicate):
        return predicate.model_dump(mode="json")
    return dict(predicate)


def _matches_rel_types(predicate: Predicate | dict, rel_types: list[str] | None) -> bool:
    if not rel_types:
        return True
    name = _entity_field(predicate, "name", "")
    return name in rel_types


def _matches_node_labels(node: Node | dict, filter_labels: list[str] | None) -> bool:
    if not filter_labels:
        return True
    labels = _entity_field(node, "labels", []) or []
    return bool(set(labels) & set(filter_labels))


def _matches_direction(predicate: Predicate | dict, filter_direction: str) -> bool:
    if filter_direction == "both":
        return True
    direction = _entity_field(predicate, "direction", "neutral")
    return direction == filter_direction


def flatten_neighborhood(
    neighbors: list[dict],
    *,
    rel_types: list[str] | None,
    node_labels: list[str] | None,
    direction: Literal["in", "out", "both"],
    limit: int,
    current_depth: int = 1,
    via_uuid: str | None = None,
) -> tuple[list[dict], bool]:
    hops: list[dict] = []
    truncated = False

    for entry in neighbors:
        if len(hops) >= limit:
            return hops, True

        predicate = entry["predicate"]
        node = entry["node"]

        if not _matches_rel_types(predicate, rel_types):
            continue
        if not _matches_node_labels(node, node_labels):
            continue
        if not _matches_direction(predicate, direction):
            continue

        hop: dict[str, Any] = {
            "depth": current_depth,
            "predicate": _serialize_predicate(predicate),
            "direction": _entity_field(predicate, "direction", "neutral"),
            "node": _serialize_node(node),
        }
        if via_uuid is not None:
            hop["via_uuid"] = via_uuid
        hops.append(hop)

        nested = entry.get("neighbors") or []
        if nested and len(hops) < limit:
            node_uuid = _entity_field(node, "uuid", "")
            nested_hops, nested_truncated = flatten_neighborhood(
                nested,
                rel_types=rel_types,
                node_labels=node_labels,
                direction=direction,
                limit=limit - len(hops),
                current_depth=current_depth + 1,
                via_uuid=node_uuid,
            )
            hops.extend(nested_hops)
            if nested_truncated:
                truncated = True
                break

        if len(hops) >= limit:
            truncated = True
            break

    return hops, truncated

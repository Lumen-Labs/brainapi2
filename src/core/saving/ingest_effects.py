"""What one ingestion task wrote, and how to remove only that."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


class DigestMismatch(Exception):
    pass


class PurgeNotFound(Exception):
    pass


@dataclass
class EffectTouch:
    uuid: str
    created: bool
    v_id: Optional[str] = None
    name: str = ""


@dataclass
class IngestEffect:
    task_id: str
    created_chunk_ids: list[str] = field(default_factory=list)
    reused_chunk_ids: list[str] = field(default_factory=list)
    chunk_vector_ids: dict[str, str] = field(default_factory=dict)
    structured_data_id: Optional[str] = None
    observations: list[dict[str, str]] = field(default_factory=list)
    nodes: list[EffectTouch] = field(default_factory=list)
    edges: list[EffectTouch] = field(default_factory=list)
    purged_at: Optional[str] = None
    revision: str = ""

    def chunk_ids(self) -> list[str]:
        return _unique([*self.created_chunk_ids, *self.reused_chunk_ids])


@dataclass
class ChunkSnapshot:
    ingest_task_ids: list[str] = field(default_factory=list)


@dataclass
class GraphSnapshot:
    source_chunk_ids: list[str] = field(default_factory=list)
    ingest_task_ids: list[str] = field(default_factory=list)
    v_id: Optional[str] = None
    name: str = ""


@dataclass
class PurgeAction:
    kind: str
    id: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class PurgePlan:
    actions: list[PurgeAction]
    digest: str
    reused: int = 0


def stamp_task_ids(existing: Any, task_id: Optional[str]) -> list[str]:
    from src.core.saving.identity import merge_source_chunk_ids

    if not task_id or not str(task_id).strip():
        return merge_source_chunk_ids(existing)
    return merge_source_chunk_ids(existing, str(task_id).strip())


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value).strip() if value is not None else ""
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _without(values: Iterable[Any], blocked: set[str]) -> list[str]:
    return [item for item in _unique(values) if item not in blocked]


def provenance_lists(properties: Optional[dict]) -> tuple[list[str], list[str], Optional[str]]:
    props = properties or {}
    nested = props.get("properties") if isinstance(props.get("properties"), dict) else {}
    chunks = _unique([*(props.get("source_chunk_ids") or []), *(nested.get("source_chunk_ids") or [])])
    tasks = _unique([*(props.get("ingest_task_ids") or []), *(nested.get("ingest_task_ids") or [])])
    vector = props.get("v_id") or nested.get("v_id")
    return chunks, tasks, str(vector) if vector else None


def merge_effect(current: Optional[IngestEffect], task_id: str, delta: dict) -> IngestEffect:
    base = current or IngestEffect(task_id=task_id)
    nodes = {item.uuid: item for item in base.nodes}
    for item in delta.get("nodes") or []:
        touch = item if isinstance(item, EffectTouch) else EffectTouch(**item)
        previous = nodes.get(touch.uuid)
        if previous is None:
            nodes[touch.uuid] = touch
        else:
            previous.created = previous.created or touch.created
            previous.v_id = previous.v_id or touch.v_id
            previous.name = previous.name or touch.name
    edges = {item.uuid: item for item in base.edges}
    for item in delta.get("edges") or []:
        touch = item if isinstance(item, EffectTouch) else EffectTouch(**item)
        previous = edges.get(touch.uuid)
        if previous is None:
            edges[touch.uuid] = touch
        else:
            previous.created = previous.created or touch.created
            previous.v_id = previous.v_id or touch.v_id
            previous.name = previous.name or touch.name
    observations: dict[str, dict[str, str]] = {}
    for item in [*base.observations, *(delta.get("observations") or [])]:
        obs_id = str(item.get("id") or "").strip()
        if obs_id:
            observations[obs_id] = {
                "id": obs_id,
                "resource_id": str(item.get("resource_id") or ""),
            }
    vectors = dict(base.chunk_vector_ids)
    vectors.update({str(key): str(value) for key, value in (delta.get("chunk_vector_ids") or {}).items() if key and value})
    structured = delta.get("structured_data_id") or base.structured_data_id
    return IngestEffect(
        task_id=task_id,
        created_chunk_ids=_unique([*base.created_chunk_ids, *(delta.get("created_chunk_ids") or [])]),
        reused_chunk_ids=_unique([*base.reused_chunk_ids, *(delta.get("reused_chunk_ids") or [])]),
        chunk_vector_ids=vectors,
        structured_data_id=str(structured) if structured else None,
        observations=list(observations.values()),
        nodes=list(nodes.values()),
        edges=list(edges.values()),
        purged_at=base.purged_at,
        revision=uuid.uuid4().hex,
    )


def plan_purge(
    effect: IngestEffect,
    *,
    chunks: dict[str, Optional[ChunkSnapshot]],
    nodes: dict[str, Optional[GraphSnapshot]],
    edges: dict[str, Optional[GraphSnapshot]],
    claimed_chunks: set[str],
) -> PurgePlan:
    if effect.purged_at:
        return PurgePlan(actions=[], digest=_digest([]), reused=len(effect.reused_chunk_ids))

    owned = set(effect.created_chunk_ids)
    actions: list[PurgeAction] = []
    deleted_chunks: set[str] = set()
    task_id = effect.task_id

    for chunk_id in effect.created_chunk_ids:
        snap = chunks.get(chunk_id)
        other_tasks = []
        if snap is not None:
            other_tasks = [item for item in snap.ingest_task_ids if item != task_id]
        if snap is None:
            continue
        vector_id = effect.chunk_vector_ids.get(chunk_id)
        if chunk_id in claimed_chunks or other_tasks:
            actions.append(
                PurgeAction(
                    "unlink_chunk",
                    chunk_id,
                    {"ingest_task_ids": other_tasks, **({"vector_id": vector_id} if vector_id else {})},
                )
            )
            continue
        deleted_chunks.add(chunk_id)
        actions.append(
            PurgeAction(
                "delete_chunk",
                chunk_id,
                {"vector_id": vector_id} if vector_id else {},
            )
        )

    for chunk_id in effect.reused_chunk_ids:
        if chunk_id in owned:
            continue
        snap = chunks.get(chunk_id)
        if snap is None or task_id not in snap.ingest_task_ids:
            continue
        actions.append(
            PurgeAction(
                "unlink_chunk",
                chunk_id,
                {"ingest_task_ids": [item for item in snap.ingest_task_ids if item != task_id]},
            )
        )

    for observation in effect.observations:
        resource = observation.get("resource_id") or ""
        if resource and resource not in deleted_chunks:
            continue
        actions.append(PurgeAction("delete_observation", observation["id"], {"resource_id": resource}))

    if effect.structured_data_id:
        actions.append(PurgeAction("delete_structured", effect.structured_data_id))

    deleted_nodes: list[str] = []
    for touch in effect.nodes:
        _classify_graph(actions, deleted_nodes, "node", touch, nodes.get(touch.uuid), owned, task_id)
    for touch in effect.edges:
        _classify_graph(actions, deleted_nodes, "edge", touch, edges.get(touch.uuid), owned, task_id)
    for node_id in deleted_nodes:
        actions.append(PurgeAction("delete_hub", node_id))

    actions.sort(key=lambda action: (action.kind, action.id))
    reused = len([chunk_id for chunk_id in effect.reused_chunk_ids if chunk_id not in deleted_chunks])
    return PurgePlan(actions=actions, digest=_digest(actions), reused=reused)


def _classify_graph(
    actions: list[PurgeAction],
    deleted_nodes: list[str],
    kind: str,
    touch: EffectTouch,
    snap: Optional[GraphSnapshot],
    owned: set[str],
    task_id: str,
) -> None:
    if snap is None:
        return
    remaining_chunks = _without(snap.source_chunk_ids, owned)
    remaining_tasks = [item for item in _unique(snap.ingest_task_ids) if item != task_id]
    name = touch.name or snap.name
    vector_id = touch.v_id or snap.v_id
    if remaining_chunks or remaining_tasks:
        actions.append(
            PurgeAction(
                f"unlink_{kind}",
                touch.uuid,
                {
                    "source_chunk_ids": remaining_chunks,
                    "ingest_task_ids": remaining_tasks,
                    "name": name,
                },
            )
        )
        return
    if touch.created or snap.source_chunk_ids:
        detail: dict[str, Any] = {"name": name}
        if vector_id:
            detail["vector_id"] = vector_id
        actions.append(PurgeAction(f"delete_{kind}", touch.uuid, detail))
        if kind == "node":
            deleted_nodes.append(touch.uuid)


def _digest(actions: list[PurgeAction]) -> str:
    payload = [
        {
            "detail": {key: value for key, value in action.detail.items() if key != "name"},
            "id": action.id,
            "kind": action.kind,
        }
        for action in actions
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def summarize(plan: PurgePlan) -> dict[str, Any]:
    def names(kind: str) -> list[str]:
        return [action.detail.get("name") or action.id for action in plan.actions if action.kind == kind]

    return {
        "digest": plan.digest,
        "executed": False,
        "passages": {
            "delete": sum(1 for action in plan.actions if action.kind == "delete_chunk"),
            "unlink": sum(1 for action in plan.actions if action.kind == "unlink_chunk"),
            "reused": plan.reused,
        },
        "records": sum(1 for action in plan.actions if action.kind == "delete_structured"),
        "entities": {"delete": names("delete_node"), "unlink": names("unlink_node")},
        "connections": {
            "delete": sum(1 for action in plan.actions if action.kind == "delete_edge"),
            "unlink": sum(1 for action in plan.actions if action.kind == "unlink_edge"),
        },
        "unresolved": sum(1 for action in plan.actions if action.kind == "unresolved"),
    }


def confirm_digest(plan: PurgePlan, digest: str) -> None:
    if not digest or digest != plan.digest:
        raise DigestMismatch


SCREENSHOT_IMPORTS = (
    {"label": "Slack message", "chars": 26},
    {"contains": "6612-WALRUS", "chars": 40},
    {"contains": "joined the channel", "chars": 37},
    {"contains": "2204-MINK", "chars": 34},
    {"label": "Connect your tools", "chars": 270},
    {"label": "Connect your tools", "chars": 1273},
    {"label": "Sync probe cedar", "chars": 66},
)

TARGET_BRAIN = "6aeae5d478a5d35c3d03a00e"


def matches_screenshot_import(label: str, chars: Optional[int]) -> bool:
    text = (label or "").strip()
    for spec in SCREENSHOT_IMPORTS:
        if chars != spec["chars"]:
            continue
        exact = spec.get("label")
        if exact is not None and text == exact:
            return True
        needle = spec.get("contains")
        if needle and needle in text:
            return True
    return False


def select_screenshot_jobs(jobs: Iterable[dict]) -> list[dict]:
    chosen: list[dict] = []
    for job in jobs:
        if str(job.get("brain_id") or job.get("brainRef") or "") != TARGET_BRAIN:
            continue
        chars = job.get("chars")
        if chars is None:
            size = job.get("size") or {}
            chars = size.get("chars")
        if matches_screenshot_import(str(job.get("label") or ""), chars if chars is None else int(chars)):
            chosen.append(job)
    return chosen


def reconstruct_effect(
    task_id: str,
    *,
    chunks: list[dict],
    nodes: list[dict],
    edges: list[dict],
    structured_data_id: Optional[str] = None,
    relationship_ids: Optional[Iterable[str]] = None,
) -> IngestEffect:
    """Build an effect for an import that predates the ledger. Does not delete."""

    chunk_ids = _unique(chunk.get("id") for chunk in chunks)
    owned = set(chunk_ids)
    known_edges = set(_unique(relationship_ids or []))
    effect_nodes: list[EffectTouch] = []
    seen_nodes: set[str] = set()
    for node in nodes:
        node_id = str(node.get("uuid") or "")
        if not node_id or node_id in seen_nodes:
            continue
        cited = set(_unique(node.get("source_chunk_ids") or []))
        endpoint = bool(node.get("endpoint"))
        if not (cited & owned) and not (endpoint and not (cited - owned)):
            continue
        seen_nodes.add(node_id)
        effect_nodes.append(
            EffectTouch(
                uuid=node_id,
                created=not bool(cited - owned),
                v_id=node.get("v_id"),
                name=str(node.get("name") or ""),
            )
        )
    effect_edges: list[EffectTouch] = []
    for edge in edges:
        edge_id = str(edge.get("uuid") or "")
        if not edge_id:
            continue
        cited = set(_unique(edge.get("source_chunk_ids") or []))
        if edge_id not in known_edges and not (cited & owned):
            continue
        effect_edges.append(
            EffectTouch(
                uuid=edge_id,
                created=edge_id in known_edges and not (cited - owned) or (bool(cited) and not (cited - owned)),
                v_id=edge.get("v_id"),
                name=str(edge.get("name") or ""),
            )
        )
    return IngestEffect(
        task_id=task_id,
        created_chunk_ids=chunk_ids,
        structured_data_id=structured_data_id,
        nodes=effect_nodes,
        edges=effect_edges,
    )


def relationship_ids_from_structured(payload: dict) -> list[str]:
    triples = (payload or {}).get("triples") or []
    if not triples:
        return []
    from src.services.api.constants.requests import IngestionTripleSet
    from src.core.agents.architect_agent import ingestion_triples_to_relationships

    current = []
    partial = []
    for raw in triples:
        parsed = IngestionTripleSet.model_validate(raw)
        if parsed.subject and parsed.subj_event:
            current.append(parsed)
        else:
            partial.append(parsed)
    relationships, _entities = ingestion_triples_to_relationships(current, partial)
    return _unique(rel.uuid for rel in relationships)


def effect_from_document(document: dict) -> IngestEffect:
    return IngestEffect(
        task_id=str(document.get("task_id") or document.get("_id")),
        created_chunk_ids=_unique(document.get("created_chunk_ids") or []),
        reused_chunk_ids=_unique(document.get("reused_chunk_ids") or []),
        chunk_vector_ids={str(key): str(value) for key, value in (document.get("chunk_vector_ids") or {}).items()},
        structured_data_id=document.get("structured_data_id"),
        observations=list(document.get("observations") or []),
        nodes=[EffectTouch(**item) for item in document.get("nodes") or []],
        edges=[EffectTouch(**item) for item in document.get("edges") or []],
        purged_at=document.get("purged_at"),
        revision=str(document.get("revision") or ""),
    )


def effect_document(effect: IngestEffect) -> dict:
    def touch(item: EffectTouch) -> dict:
        row = {"uuid": item.uuid, "created": item.created, "name": item.name}
        if item.v_id:
            row["v_id"] = item.v_id
        return row

    return {
        "_id": effect.task_id,
        "task_id": effect.task_id,
        "created_chunk_ids": effect.created_chunk_ids,
        "reused_chunk_ids": effect.reused_chunk_ids,
        "chunk_vector_ids": effect.chunk_vector_ids,
        "structured_data_id": effect.structured_data_id,
        "observations": effect.observations,
        "nodes": [touch(item) for item in effect.nodes],
        "edges": [touch(item) for item in effect.edges],
        "purged_at": effect.purged_at,
        "revision": effect.revision,
    }


def _effects(brain_id: str):
    from src.core.saving.pipeline_store import collection

    return collection(brain_id, "ingest_effects")


def load_effect(brain_id: str, task_id: str) -> Optional[IngestEffect]:
    document = _effects(brain_id).find_one({"_id": task_id})
    if not document:
        return None
    return effect_from_document(document)


def list_effects(brain_id: str) -> list[IngestEffect]:
    return [effect_from_document(document) for document in _effects(brain_id).find({})]


def remember(brain_id: str, task_id: str, delta: Optional[dict] = None) -> None:
    if not task_id:
        return
    try:
        _remember(brain_id, task_id, delta or {})
    except Exception as exc:
        print(f"[!] ingest effect was not recorded for {task_id}: {exc}")


def _remember(brain_id: str, task_id: str, delta: dict) -> None:
    store = _effects(brain_id)
    for _attempt in range(5):
        current = load_effect(brain_id, task_id)
        if current and current.purged_at:
            return
        merged = merge_effect(current, task_id, delta)
        document = effect_document(merged)
        if current is None:
            try:
                store.insert_one(document)
                return
            except Exception:
                continue
        result = store.update_one(
            {"_id": task_id, "revision": current.revision},
            {"$set": document},
        )
        if getattr(result, "modified_count", 0):
            return
    print(f"[!] ingest effect revision conflict for {task_id}")


def claimed_chunk_ids(brain_id: str, task_id: str) -> set[str]:
    claimed: set[str] = set()
    for effect in list_effects(brain_id):
        if effect.task_id == task_id or effect.purged_at:
            continue
        claimed.update(effect.created_chunk_ids)
        claimed.update(effect.reused_chunk_ids)
    return claimed


def apply_actions(plan: PurgePlan, sink: Any) -> None:
    chunk_deletes = [action.id for action in plan.actions if action.kind == "delete_chunk"]
    chunk_vectors = [
        action.detail["vector_id"]
        for action in plan.actions
        if action.kind == "delete_chunk" and action.detail.get("vector_id")
    ]
    if chunk_deletes:
        sink.delete_chunks(chunk_deletes)
        sink.delete_passage_vectors(chunk_deletes, chunk_vectors)
    for action in plan.actions:
        if action.kind == "unlink_chunk":
            sink.unlink_chunk(action.id, action.detail.get("ingest_task_ids") or [])
        elif action.kind == "delete_observation":
            sink.delete_observations([action.id])
        elif action.kind == "delete_structured":
            sink.delete_structured(action.id)
        elif action.kind == "unlink_node":
            sink.unlink_graph("node", action.id, action.detail)
        elif action.kind == "unlink_edge":
            sink.unlink_graph("edge", action.id, action.detail)
        elif action.kind == "delete_node":
            sink.delete_nodes([action.id])
            if action.detail.get("vector_id"):
                sink.delete_graph_vectors("nodes", [action.detail["vector_id"]])
        elif action.kind == "delete_edge":
            sink.delete_edges([action.id])
            if action.detail.get("vector_id"):
                sink.delete_graph_vectors("relationships", [action.detail["vector_id"]])
    hubs = [action.id for action in plan.actions if action.kind == "delete_hub"]
    if hubs:
        sink.delete_hubs(hubs)


def build_plan(
    effect: IngestEffect,
    *,
    chunks: dict[str, Optional[ChunkSnapshot]],
    nodes: dict[str, Optional[GraphSnapshot]],
    edges: dict[str, Optional[GraphSnapshot]],
    claimed_chunks: Optional[set[str]] = None,
) -> PurgePlan:
    return plan_purge(
        effect,
        chunks=chunks,
        nodes=nodes,
        edges=edges,
        claimed_chunks=claimed_chunks or set(),
    )


class BrainSink:
    def __init__(self, brain_id: str):
        self.brain_id = brain_id

    def delete_chunks(self, ids):
        from src.services.data.main import data_adapter

        data_adapter.delete_text_chunks(ids, self.brain_id)
        data_adapter.delete_observations([], self.brain_id, ids)

    def delete_passage_vectors(self, chunk_ids, vector_ids):
        from src.services.kg_agent.main import vector_store_adapter

        vector_store_adapter.remove_vectors_by_resource_id(chunk_ids, "data", self.brain_id)
        if vector_ids:
            vector_store_adapter.remove_vectors([str(item) for item in vector_ids], "data", self.brain_id)

    def unlink_chunk(self, chunk_id, task_ids):
        from src.services.data.main import data_adapter

        data_adapter.set_text_chunk_task_ids(chunk_id, task_ids, self.brain_id)

    def delete_observations(self, ids):
        from src.services.data.main import data_adapter

        data_adapter.delete_observations(ids, self.brain_id)

    def delete_structured(self, structured_id):
        from src.services.data.main import data_adapter

        data_adapter.delete_structured_data(structured_id, self.brain_id)

    def unlink_graph(self, kind, uuid, detail):
        from src.services.kg_agent.main import graph_adapter

        new_properties = {
            "source_chunk_ids": detail.get("source_chunk_ids") or [],
            "ingest_task_ids": detail.get("ingest_task_ids") or [],
        }
        graph_adapter.update_properties(
            uuid,
            "node" if kind == "node" else "relationship",
            self.brain_id,
            new_properties,
            [],
        )

    def delete_nodes(self, ids):
        from src.services.kg_agent.main import graph_adapter

        graph_adapter.remove_nodes(ids, self.brain_id)

    def delete_edges(self, ids):
        from src.services.kg_agent.main import graph_adapter

        graph_adapter.remove_relationships(
            [({"uuid": ""}, {"uuid": edge_id, "name": ""}, {"uuid": ""}) for edge_id in ids],
            self.brain_id,
        )

    def delete_graph_vectors(self, store, ids):
        from src.services.kg_agent.main import vector_store_adapter

        vector_store_adapter.remove_vectors([str(item) for item in ids], store, self.brain_id)

    def delete_hubs(self, ids):
        from src.services.kg_agent.main import graph_adapter

        graph_adapter.refresh_hub_bridges_for_entities(ids, self.brain_id)


def _chunk_snapshot(chunk: Any) -> ChunkSnapshot:
    metadata = getattr(chunk, "metadata", None) or {}
    return ChunkSnapshot(ingest_task_ids=_unique(metadata.get("ingest_task_ids") or []))


def _graph_snapshot(properties: Optional[dict], name: str = "") -> GraphSnapshot:
    chunks, tasks, vector = provenance_lists(properties)
    return GraphSnapshot(
        source_chunk_ids=chunks,
        ingest_task_ids=tasks,
        v_id=vector,
        name=name,
    )


def _chunks_for_source(brain_id: str, source_key: str) -> list[Any]:
    from src.services.data.main import data_adapter

    found: list[Any] = []
    skip = 0
    while skip < 500:
        rows, total = data_adapter.get_text_chunks(
            brain_id=brain_id,
            limit=100,
            skip=skip,
            metadata_eq={"source_key": source_key},
        )
        if not rows:
            break
        found.extend(rows)
        skip += len(rows)
        if total is not None and skip >= int(total):
            break
    return found


def _structured_payload(brain_id: str, task_id: str) -> Optional[dict]:
    from src.services.data.main import data_adapter

    try:
        row = data_adapter.get_structured_data_by_id(task_id, brain_id)
    except Exception:
        return None
    if row is None:
        return None
    data = getattr(row, "data", None) or {}
    return data if isinstance(data, dict) else None


def reconstruct_live(brain_id: str, task_id: str, source_key: Optional[str] = None) -> Optional[IngestEffect]:
    chunks = _chunks_for_source(brain_id, source_key) if source_key else []
    payload = _structured_payload(brain_id, task_id)
    relationship_ids: list[str] = []
    if payload:
        try:
            relationship_ids = relationship_ids_from_structured(payload)
        except Exception:
            relationship_ids = []
    if not chunks and not relationship_ids and not payload:
        return None
    from src.services.kg_agent.main import graph_adapter

    graph_nodes, graph_edges = graph_adapter.nodes_and_edges_citing(
        brain_id,
        [str(getattr(chunk, "id", "")) for chunk in chunks],
        relationship_ids,
    )
    if relationship_ids:
        try:
            extra = graph_adapter.get_triples_by_uuid(relationship_ids, brain_id)
        except Exception:
            extra = []
        seen = {
            str((getattr(edge[1], "properties", None) or {}).get("uuid") or getattr(edge[1], "uuid", ""))
            for edge in graph_edges
        }
        for triple in extra or []:
            predicate = triple[1]
            edge_id = str((getattr(predicate, "properties", None) or {}).get("uuid") or getattr(predicate, "uuid", ""))
            if edge_id and edge_id not in seen:
                graph_edges.append(triple)
                seen.add(edge_id)
    node_rows = []
    for node in graph_nodes:
        cited, _tasks, vector = provenance_lists(getattr(node, "properties", None))
        node_rows.append(
            {
                "uuid": node.uuid,
                "name": node.name,
                "source_chunk_ids": cited,
                "v_id": vector,
            }
        )
    edge_rows = []
    endpoint_ids: set[str] = set()
    for tail, predicate, tip in graph_edges:
        cited, _tasks, vector = provenance_lists(getattr(predicate, "properties", None))
        edge_id = str((getattr(predicate, "properties", None) or {}).get("uuid") or getattr(predicate, "uuid", ""))
        edge_rows.append(
            {
                "uuid": edge_id,
                "name": getattr(predicate, "name", "") or "",
                "source_chunk_ids": cited,
                "v_id": vector,
            }
        )
        if edge_id in set(relationship_ids):
            endpoint_ids.add(str(getattr(tail, "uuid", "") or ""))
            endpoint_ids.add(str(getattr(tip, "uuid", "") or ""))
    known = {row["uuid"] for row in node_rows}
    for endpoint in endpoint_ids:
        if not endpoint or endpoint in known:
            continue
        try:
            node = graph_adapter.get_by_uuid(endpoint, brain_id)
        except Exception:
            node = None
        if node is None:
            continue
        cited, _tasks, vector = provenance_lists(getattr(node, "properties", None))
        node_rows.append(
            {
                "uuid": endpoint,
                "name": getattr(node, "name", "") or "",
                "source_chunk_ids": cited,
                "v_id": vector,
                "endpoint": True,
            }
        )
    return reconstruct_effect(
        task_id,
        chunks=[{"id": str(getattr(chunk, "id", ""))} for chunk in chunks if getattr(chunk, "id", None)],
        nodes=node_rows,
        edges=edge_rows,
        structured_data_id=task_id if payload else None,
        relationship_ids=relationship_ids,
    )


def snapshots_for(brain_id: str, effect: IngestEffect):
    from src.services.data.main import data_adapter
    from src.services.kg_agent.main import graph_adapter

    chunks: dict[str, Optional[ChunkSnapshot]] = {}
    ids = effect.chunk_ids()
    if ids:
        loaded, _observations = data_adapter.get_text_chunks_by_ids(ids, False, brain_id)
        found = {str(chunk.id): chunk for chunk in loaded}
        for chunk_id in ids:
            chunk = found.get(chunk_id)
            chunks[chunk_id] = _chunk_snapshot(chunk) if chunk is not None else None
    nodes: dict[str, Optional[GraphSnapshot]] = {}
    node_ids = [item.uuid for item in effect.nodes]
    loaded_nodes = graph_adapter.get_by_uuids(node_ids, brain_id) if node_ids else []
    by_node = {node.uuid: node for node in loaded_nodes}
    for touch in effect.nodes:
        node = by_node.get(touch.uuid)
        nodes[touch.uuid] = _graph_snapshot(node.properties, node.name) if node is not None else None
    edges: dict[str, Optional[GraphSnapshot]] = {}
    edge_ids = [item.uuid for item in effect.edges]
    loaded_edges = graph_adapter.get_triples_by_uuid(edge_ids, brain_id) if edge_ids else []
    by_edge: dict[str, Any] = {}
    for triple in loaded_edges or []:
        predicate = triple[1]
        edge_id = str((getattr(predicate, "properties", None) or {}).get("uuid") or getattr(predicate, "uuid", ""))
        if edge_id:
            by_edge[edge_id] = predicate
    for touch in effect.edges:
        predicate = by_edge.get(touch.uuid)
        edges[touch.uuid] = (
            _graph_snapshot(getattr(predicate, "properties", None), getattr(predicate, "name", "") or "")
            if predicate is not None
            else None
        )
    return chunks, nodes, edges


def mark_purged(brain_id: str, effect: IngestEffect) -> None:
    from datetime import datetime, timezone

    effect.purged_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    effect.revision = uuid.uuid4().hex
    document = effect_document(effect)
    store = _effects(brain_id)
    if load_effect(brain_id, effect.task_id) is None:
        store.insert_one(document)
        return
    store.update_one({"_id": effect.task_id}, {"$set": document})


def purge_task(
    brain_id: str,
    task_id: str,
    *,
    digest: Optional[str] = None,
    source_key: Optional[str] = None,
) -> dict[str, Any]:
    effect = load_effect(brain_id, task_id)
    if effect is None:
        effect = reconstruct_live(brain_id, task_id, source_key)
    if effect is None:
        raise PurgeNotFound
    chunks, nodes, edges = snapshots_for(brain_id, effect)
    plan = build_plan(
        effect,
        chunks=chunks,
        nodes=nodes,
        edges=edges,
        claimed_chunks=claimed_chunk_ids(brain_id, task_id),
    )
    summary = summarize(plan)
    if not digest:
        return summary
    confirm_digest(plan, digest)
    apply_actions(plan, BrainSink(brain_id))
    mark_purged(brain_id, effect)
    summary["executed"] = True
    return summary


def handle_purge(
    effect: IngestEffect,
    *,
    chunks: dict[str, Optional[ChunkSnapshot]],
    nodes: dict[str, Optional[GraphSnapshot]],
    edges: dict[str, Optional[GraphSnapshot]],
    claimed_chunks: Optional[set[str]] = None,
    digest: Optional[str] = None,
    sink: Any = None,
) -> dict[str, Any]:
    plan = build_plan(effect, chunks=chunks, nodes=nodes, edges=edges, claimed_chunks=claimed_chunks)
    summary = summarize(plan)
    if not digest:
        return summary
    confirm_digest(plan, digest)
    if sink is not None:
        apply_actions(plan, sink)
    summary["executed"] = True
    return summary

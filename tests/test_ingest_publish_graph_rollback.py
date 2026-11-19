import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.constants.kg import Node, Predicate
from src.core.saving import ingest_cancellation as cancellation


class _MemoryCollection:
    def __init__(self, documents):
        self.documents = documents

    def find_one(self, query):
        row = self.documents.get(query.get("_id"))
        if row is None:
            return None
        if any(row.get(key) != value for key, value in query.items() if key != "_id"):
            return None
        return row.copy()

    def update_one(self, query, update, upsert=False):
        task_id = query["_id"]
        if upsert:
            self.documents.setdefault(task_id, {"_id": task_id, **update.get("$setOnInsert", {})})
        row = self.documents[task_id]
        if "revision" in query and row.get("revision") != query["revision"]:
            return SimpleNamespace(modified_count=0)
        row.update(update.get("$set", {}))
        return SimpleNamespace(modified_count=1)

    def find(self, query):
        return [row.copy() for row in self.documents.values()]


@pytest.fixture
def run(monkeypatch):
    documents = {}
    store = _MemoryCollection(documents)
    monkeypatch.setenv("INGESTION_CANCELLATION_ENABLED", "1")
    monkeypatch.setattr(
        cancellation, "_runs",
        lambda brain_id: store if brain_id == "brain" else Mock(
            find_one=Mock(return_value=None), find=Mock(return_value=[]),
        ),
    )
    cancellation.register("brain", "root")
    return documents


class FakeGraph:
    """In-memory graph with Neo4j MERGE/DELETE semantics and JSON-string observations."""

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.remove_nodes_calls = 0
        self.fail_remove_nodes_on = None

    def _shape_observations(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, default=str)

    def _store_node(self, node):
        payload = node.model_dump(mode="json") if hasattr(node, "model_dump") else dict(node)
        uuid = payload["uuid"]
        observations = self._shape_observations(payload.get("observations"))
        properties = dict(payload.get("properties") or {})
        if observations is not None:
            properties["observations"] = observations
        stored = {
            "uuid": uuid,
            "name": payload.get("name") or "",
            "labels": list(payload.get("labels") or []),
            "description": payload.get("description") or "",
            "properties": properties,
            "polarity": payload.get("polarity") or "neutral",
            "metadata": payload.get("metadata"),
            "happened_at": payload.get("happened_at"),
            "last_updated": payload.get("last_updated"),
            "observations": observations,
        }
        existing = self.nodes.get(uuid)
        if existing is None:
            self.nodes[uuid] = stored
            return stored
        merged_props = dict(existing.get("properties") or {})
        merged_props.update(stored["properties"])
        for key in ("source_chunk_ids", "aliases", "ingest_task_ids"):
            previous = existing.get("properties", {}).get(key) or existing.get(key)
            incoming = stored["properties"].get(key)
            if isinstance(previous, list) or isinstance(incoming, list):
                merged_props[key] = list(dict.fromkeys([*(previous or []), *(incoming or [])]))
        stored["properties"] = merged_props
        if stored.get("description") and existing.get("description") and stored["description"] != existing["description"]:
            stored["description"] = existing["description"] + " | " + stored["description"]
        elif not stored.get("description"):
            stored["description"] = existing.get("description") or ""
        for field in ("name", "polarity", "metadata", "happened_at", "last_updated", "observations"):
            if stored.get(field) in (None, "", []):
                stored[field] = existing.get(field)
        self.nodes[uuid] = stored
        return stored

    def add_nodes(self, nodes, brain_id, identification_params=None, metadata=None):
        return [self._store_node(node) and node for node in nodes]

    def add_relationship(self, subject, predicate, to_object, brain_id):
        pred = predicate.model_dump(mode="json") if hasattr(predicate, "model_dump") else dict(predicate)
        uuid = pred["uuid"]
        stored = {
            "uuid": uuid,
            "name": pred.get("name") or "",
            "description": pred.get("description") or "",
            "properties": dict(pred.get("properties") or {}),
            "subject": getattr(subject, "uuid", None) or subject.get("uuid"),
            "object": getattr(to_object, "uuid", None) or to_object.get("uuid"),
        }
        existing = self.edges.get(uuid)
        if existing:
            props = dict(existing.get("properties") or {})
            props.update(stored["properties"])
            stored["properties"] = props
            stored["subject"] = existing["subject"]
            stored["object"] = existing["object"]
            if not stored.get("description"):
                stored["description"] = existing.get("description") or ""
        self.edges[uuid] = stored
        return "ok"

    def get_by_uuid(self, uuid, brain_id):
        nodes = self.get_by_uuids([uuid], brain_id)
        return nodes[0] if nodes else None

    def get_by_uuids(self, uuids, brain_id):
        found = []
        for uuid in uuids:
            stored = self.nodes.get(uuid)
            if stored is None:
                continue
            observations = stored.get("observations")
            if isinstance(observations, str):
                observations = json.loads(observations)
            kwargs = {
                "uuid": stored["uuid"],
                "name": stored["name"],
                "labels": stored["labels"],
                "description": stored.get("description") or "",
                "properties": dict(stored.get("properties") or {}),
                "polarity": stored.get("polarity") or "neutral",
            }
            if observations is not None:
                kwargs["observations"] = observations
            if stored.get("metadata") is not None:
                kwargs["metadata"] = stored["metadata"]
            found.append(Node(**kwargs))
        return found

    def get_triples_by_uuid(self, uuids, brain_id):
        triples = []
        for uuid in uuids:
            edge = self.edges.get(uuid)
            if edge is None:
                continue
            subject = self.get_by_uuid(edge["subject"], brain_id) or Node(
                uuid=edge["subject"], name="", labels=["ENTITY"],
            )
            obj = self.get_by_uuid(edge["object"], brain_id) or Node(
                uuid=edge["object"], name="", labels=["ENTITY"],
            )
            triples.append((
                subject,
                Predicate(
                    uuid=edge["uuid"],
                    name=edge["name"],
                    description=edge.get("description") or "",
                    properties=dict(edge.get("properties") or {}),
                ),
                obj,
            ))
        return triples

    def list_incident_relationship_uuids(self, uuids, brain_id):
        found = {uuid: [] for uuid in uuids}
        for edge_id, edge in self.edges.items():
            for uuid in uuids:
                if edge["subject"] == uuid or edge["object"] == uuid:
                    found[uuid].append(edge_id)
        return found

    def remove_nodes(self, uuids, brain_id):
        self.remove_nodes_calls += 1
        if self.fail_remove_nodes_on == self.remove_nodes_calls:
            raise RuntimeError("injected remove_nodes failure")
        removed = []
        for uuid in uuids:
            if uuid not in self.nodes:
                continue
            if self.list_incident_relationship_uuids([uuid], brain_id).get(uuid):
                raise ValueError(f"Cannot delete node {uuid} with remaining relationships")
            removed.append(self.nodes.pop(uuid))
        return removed

    def remove_relationships(self, relationships, brain_id):
        deleted = []
        for _subject, pred, _obj in relationships:
            uuid = pred.get("uuid") if isinstance(pred, dict) else getattr(pred, "uuid", None)
            if uuid and uuid in self.edges:
                deleted.append(self.edges.pop(uuid))
        return deleted

    def update_properties(self, uuid, updating, brain_id, new_properties, properties_to_remove):
        target = self.nodes if updating == "node" else self.edges
        stored = target.get(uuid)
        if stored is None:
            return None
        props = dict(stored.get("properties") or {})
        for key, value in (new_properties or {}).items():
            if key in ("name", "description", "polarity", "metadata", "observations", "happened_at", "last_updated"):
                stored[key] = value
                if key == "observations":
                    stored["observations"] = self._shape_observations(value)
                    props["observations"] = stored["observations"]
            else:
                props[key] = value
        for key in properties_to_remove or []:
            if key in ("name", "description", "polarity", "metadata", "observations", "happened_at", "last_updated"):
                stored[key] = "" if key in {"name", "description"} else None
            props.pop(key, None)
        stored["properties"] = props
        return stored


def _person(uuid, name, **kwargs):
    return Node(uuid=uuid, name=name, labels=["PERSON"], **kwargs)


def _rel(uuid, name="KNOWS", **kwargs):
    kwargs.setdefault("description", "")
    return Predicate(uuid=uuid, name=name, **kwargs)


def _node_op(*nodes):
    return {
        "adapter": "graph",
        "method": "add_nodes",
        "args": {
            "nodes": [node.model_dump(mode="json") for node in nodes],
            "identification_params": None,
            "metadata": None,
        },
    }


def _rel_op(subject, predicate, obj):
    return {
        "adapter": "graph",
        "method": "add_relationship",
        "args": {
            "subject": subject.model_dump(mode="json"),
            "predicate": predicate.model_dump(mode="json"),
            "to_object": obj.model_dump(mode="json"),
        },
    }


def _publish_graph(run, monkeypatch, tmp_path, operations, graph):
    from src.core.saving import ingest_access_gate as gate, ingest_publication, ingest_staging as staging

    monkeypatch.setenv("INGEST_PUBLICATION_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(gate, "_gates", cancellation._runs)
    monkeypatch.setattr(ingest_publication, "load_output", lambda brain, root, ref: operations)
    monkeypatch.setattr(ingest_publication, "finish_publication", lambda *args, **kwargs: None)
    staging.create_run("brain", "staged")
    unit = staging.register_unit("brain", "staged", "unit-0", {"payload": {}})
    claim = staging.claim_unit("brain", "staged", unit)
    assert staging.complete_unit("brain", "staged", unit, claim, "graph-ops")
    staging.seal_units("brain", "staged")
    assert staging.close_staging("brain", "staged")["status"] == "publishing"
    ingest_publication.publish_staged_run("brain", "staged", gate.publication_gate, Mock(), Mock(), graph)
    cancellation._runs("brain").update_one(
        {"_id": "staged"},
        {"$set": {"status": "publish_partial"}},
    )
    return graph


def test_rollback_created_only_nodes_and_edges(run, monkeypatch, tmp_path):
    from src.core.saving import ingest_publication, ingest_staging as staging

    alice = _person("alice-1", "Alice")
    bob = _person("bob-1", "Bob")
    knows = _rel("rel-1", properties={"note": "new"})
    graph = FakeGraph()
    _publish_graph(run, monkeypatch, tmp_path, [_node_op(alice, bob), _rel_op(alice, knows, bob)], graph)
    assert set(graph.nodes) == {"alice-1", "bob-1"}
    assert set(graph.edges) == {"rel-1"}
    ledger = staging.graph_touches("brain", "staged")
    assert any(touch["created"] for touches in ledger.values() for touch in touches)

    ingest_publication.rollback_published_run("brain", "staged", Mock(), Mock(), graph)
    staging.finish_publish_rollback("brain", "staged")
    assert graph.nodes == {}
    assert graph.edges == {}
    stored = cancellation._runs("brain").find_one({"_id": "staged"})
    assert stored["status"] == "failed"
    assert stored["pending_cleanup"] is True


def test_rollback_reused_node_restores_prior_properties(run, monkeypatch, tmp_path):
    from src.core.saving import ingest_publication, ingest_staging as staging

    graph = FakeGraph()
    graph.add_nodes([_person("alice-1", "Alice", properties={"note": "old", "keep": "yes"})], "brain")
    updated = _person("alice-1", "Alice", properties={"note": "new", "keep": "yes", "extra": "added"})
    _publish_graph(run, monkeypatch, tmp_path, [_node_op(updated)], graph)
    assert graph.nodes["alice-1"]["properties"]["note"] == "new"
    assert graph.nodes["alice-1"]["properties"]["extra"] == "added"
    touches = [touch for touches in staging.graph_touches("brain", "staged").values() for touch in touches]
    assert touches[0]["created"] is False
    assert touches[0]["prior"]["properties"]["note"] == "old"

    ingest_publication.rollback_published_run("brain", "staged", Mock(), Mock(), graph)
    staging.finish_publish_rollback("brain", "staged")
    assert "alice-1" in graph.nodes
    assert graph.nodes["alice-1"]["properties"]["note"] == "old"
    assert graph.nodes["alice-1"]["properties"]["keep"] == "yes"
    assert "extra" not in graph.nodes["alice-1"]["properties"]


def test_rollback_reused_node_keeps_foreign_edges(run, monkeypatch, tmp_path):
    from src.core.saving import ingest_publication, ingest_staging as staging

    graph = FakeGraph()
    alice = _person("alice-1", "Alice", properties={"note": "old"})
    bob = _person("bob-1", "Bob")
    carol = _person("carol-1", "Carol")
    graph.add_nodes([alice, bob, carol], "brain")
    graph.add_relationship(alice, _rel("foreign-1", "KNOWS", properties={"owned": "other"}), carol, "brain")
    published_alice = _person("alice-1", "Alice", properties={"note": "new"})
    our_rel = _rel("rel-1", "KNOWS", properties={"owned": "this-run"})
    _publish_graph(
        run, monkeypatch, tmp_path,
        [_node_op(published_alice), _rel_op(alice, our_rel, bob)],
        graph,
    )
    assert set(graph.edges) == {"foreign-1", "rel-1"}
    assert graph.nodes["alice-1"]["properties"]["note"] == "new"

    ingest_publication.rollback_published_run("brain", "staged", Mock(), Mock(), graph)
    staging.finish_publish_rollback("brain", "staged")
    assert "alice-1" in graph.nodes
    assert "bob-1" in graph.nodes
    assert "carol-1" in graph.nodes
    assert "rel-1" not in graph.edges
    assert "foreign-1" in graph.edges
    assert graph.nodes["alice-1"]["properties"]["note"] == "old"


def test_rollback_created_node_with_foreign_edges_is_kept(run, monkeypatch, tmp_path):
    from src.core.saving import ingest_publication, ingest_staging as staging

    graph = FakeGraph()
    alice = _person("alice-1", "Alice")
    bob = _person("bob-1", "Bob")
    _publish_graph(run, monkeypatch, tmp_path, [_node_op(alice), _rel_op(alice, _rel("rel-1"), bob)], graph)
    graph.add_nodes([bob], "brain")
    graph.add_relationship(alice, _rel("foreign-1", "KNOWS"), bob, "brain")

    ingest_publication.rollback_published_run("brain", "staged", Mock(), Mock(), graph)
    staging.finish_publish_rollback("brain", "staged")
    assert "alice-1" in graph.nodes
    assert "rel-1" not in graph.edges
    assert "foreign-1" in graph.edges


def test_rollback_failure_midway_is_retryable(run, monkeypatch, tmp_path):
    from src.core.saving import ingest_publication, ingest_staging as staging

    graph = FakeGraph()
    alice = _person("alice-1", "Alice")
    bob = _person("bob-1", "Bob")
    _publish_graph(run, monkeypatch, tmp_path, [_node_op(alice, bob)], graph)
    graph.fail_remove_nodes_on = 2

    with pytest.raises(RuntimeError, match="injected remove_nodes failure"):
        ingest_publication.rollback_published_run("brain", "staged", Mock(), Mock(), graph)
    stored = cancellation._runs("brain").find_one({"_id": "staged"})
    assert stored["status"] == "publish_partial"
    assert stored["rollback_reason"] == "rollback_apply_failed"
    assert cancellation.public_state("brain", "staged", {})["rollback_reason"] == "rollback_apply_failed"
    remaining = set(graph.nodes)
    assert remaining in ({"alice-1"}, {"bob-1"})
    assert stored.get("rollback_done")

    graph.fail_remove_nodes_on = None
    cancellation._runs("brain").update_one(
        {"_id": "staged"},
        {"$set": {"status": "rollback_failed", "pending_cleanup": False}},
    )
    ingest_publication.rollback_published_run("brain", "staged", Mock(), Mock(), graph)
    staging.finish_publish_rollback("brain", "staged")
    assert graph.nodes == {}
    stored = cancellation._runs("brain").find_one({"_id": "staged"})
    assert stored["status"] == "failed"
    assert stored["pending_cleanup"] is True


def test_rollback_precheck_aborts_without_partial_state(run, monkeypatch, tmp_path):
    from src.core.saving import ingest_publication, ingest_staging as staging

    graph = FakeGraph()
    alice = _person("alice-1", "Alice")
    _publish_graph(run, monkeypatch, tmp_path, [_node_op(alice)], graph)
    cancellation._runs("brain").update_one({"_id": "staged"}, {"$set": {"graph_touches": {}}})

    ingest_publication.rollback_published_run("brain", "staged", Mock(), Mock(), graph)
    stored = cancellation._runs("brain").find_one({"_id": "staged"})
    assert stored["status"] == "rollback_failed"
    assert stored["pending_cleanup"] is False
    assert "created/reused markers" in stored["rollback_reason"]
    assert "alice-1" in graph.nodes
    assert cancellation.public_state("brain", "staged", {})["status"] == "rollback_failed"

    ingest_publication.rollback_published_run("brain", "staged", Mock(), Mock(), graph)
    stored = cancellation._runs("brain").find_one({"_id": "staged"})
    assert stored["status"] == "rollback_failed"
    assert "alice-1" in graph.nodes


def test_publish_records_created_versus_reused_before_merge(run, monkeypatch, tmp_path):
    from src.core.saving import ingest_staging as staging

    graph = FakeGraph()
    graph.add_nodes([_person("alice-1", "Alice", properties={"note": "old"})], "brain")
    alice = _person("alice-1", "Alice", properties={"note": "new"})
    bob = _person("bob-1", "Bob")
    _publish_graph(run, monkeypatch, tmp_path, [_node_op(alice, bob)], graph)
    touches = [touch for values in staging.graph_touches("brain", "staged").values() for touch in values]
    by_uuid = {touch["uuid"]: touch for touch in touches}
    assert by_uuid["alice-1"]["created"] is False
    assert by_uuid["alice-1"]["prior"]["properties"]["note"] == "old"
    assert by_uuid["bob-1"]["created"] is True
    assert by_uuid["bob-1"]["prior"] is None

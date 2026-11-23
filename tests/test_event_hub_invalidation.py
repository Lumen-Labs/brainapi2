import os
import sys
import unittest
from unittest.mock import patch


ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "EMBEDDINGS_LOCAL_MODEL": "local-model",
    "EMBEDDINGS_SMALL_MODEL": "small-model",
    "EMBEDDING_NODES_DIMENSION": "3",
    "EMBEDDING_TRIPLETS_DIMENSION": "3",
    "EMBEDDING_OBSERVATIONS_DIMENSION": "3",
    "EMBEDDING_DATA_DIMENSION": "3",
    "EMBEDDING_RELATIONSHIPS_DIMENSION": "3",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
    "CELERY_WORKER_CONCURRENCY": "1",
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_LLM_SMALL_MODEL": "small",
    "OLLAMA_LLM_LARGE_MODEL": "large",
    "PIPELINE_MODE": "accurate",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.agents import ArchitectAgentEntity, ArchitectAgentRelationship
from src.constants.kg import Node, Predicate


class _FakeGraphAdapter:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.updates = []

    def add_node(self, uuid, name, node_type):
        self.nodes[uuid] = Node(uuid=uuid, name=name, labels=[node_type])
        return self.nodes[uuid]

    def add_edge(self, rel_uuid, tail_uuid, name, tip_uuid, properties=None):
        predicate = Predicate(
            uuid=rel_uuid,
            name=name,
            description=f"desc-{name}",
            properties=dict(properties or {}),
        )
        self.edges.append((tail_uuid, predicate, tip_uuid))
        return predicate

    def predicate(self, rel_uuid):
        for _, predicate, _ in self.edges:
            if predicate.uuid == rel_uuid:
                return predicate
        raise KeyError(rel_uuid)

    def get_neighbors(
        self,
        nodes,
        brain_id="default",
        same_type_only=False,
        limit=None,
        of_types=None,
    ):
        uuids = [node if isinstance(node, str) else node.uuid for node in nodes]
        neighbors = {uuid: [] for uuid in uuids}
        for tail_uuid, predicate, tip_uuid in self.edges:
            if tail_uuid in neighbors:
                neighbors[tail_uuid].append(
                    (
                        predicate.model_copy(update={"direction": "out"}),
                        self.nodes[tip_uuid],
                    )
                )
            if tip_uuid in neighbors:
                neighbors[tip_uuid].append(
                    (
                        predicate.model_copy(update={"direction": "in"}),
                        self.nodes[tail_uuid],
                    )
                )
        return neighbors

    def update_properties(self, uuid, kind, brain_id="default", new_properties=None):
        self.updates.append((uuid, kind, dict(new_properties or {})))
        predicate = self.predicate(uuid)
        predicate.properties = {
            **(predicate.properties or {}),
            **(new_properties or {}),
        }
        if (new_properties or {}).get("deprecated"):
            predicate.deprecated = True


def _entity(uuid, name, entity_type):
    return ArchitectAgentEntity(uuid=uuid, name=name, type=entity_type)


def _relationship(rel_uuid, tail, name, tip, valid_at=None):
    return ArchitectAgentRelationship(
        uuid=rel_uuid,
        flow_key="flow-1",
        tail=tail,
        name=name,
        description=f"desc-{name}",
        tip=tip,
        properties={"valid_at": valid_at} if valid_at else {},
    )


class EventHubInvalidationTests(unittest.TestCase):
    def test_second_event_by_same_actor_keeps_both_attributions(self):
        from src.services.api.controllers import retrieve as retrieve_mod
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("alice", "Alice", "PERSON")
        graph.add_node("event-1", "Purchase", "EVENT")
        graph.add_node("event-2", "Trip", "EVENT")
        graph.add_edge("rel-1", "alice", "MADE", "event-1")
        graph.add_edge("rel-2", "alice", "MADE", "event-2")

        relationship = _relationship(
            "rel-2",
            _entity("alice", "Alice", "PERSON"),
            "MADE",
            _entity("event-2", "Trip", "EVENT"),
            valid_at="02/02/2024",
        )

        with patch.object(ingestion_mod, "graph_adapter", graph):
            ingestion_mod._invalidate_superseded_relationships(
                relationship, brain_id="brain-a"
            )

        self.assertEqual(graph.updates, [])
        self.assertTrue(retrieve_mod._is_currently_valid(graph.predicate("rel-1")))
        self.assertTrue(retrieve_mod._is_currently_valid(graph.predicate("rel-2")))

    def test_functional_attribute_edge_is_still_superseded(self):
        from src.services.api.controllers import retrieve as retrieve_mod
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("alice", "Alice", "PERSON")
        graph.add_node("nyc", "New York City", "CITY")
        graph.add_node("sf", "San Francisco", "CITY")
        graph.add_edge("rel-1", "alice", "LIVES_IN", "nyc")
        graph.add_edge("rel-2", "alice", "LIVES_IN", "sf")

        relationship = _relationship(
            "rel-2",
            _entity("alice", "Alice", "PERSON"),
            "LIVES_IN",
            _entity("sf", "San Francisco", "CITY"),
            valid_at="02/02/2024",
        )

        with patch.object(ingestion_mod, "graph_adapter", graph):
            ingestion_mod._invalidate_superseded_relationships(
                relationship, brain_id="brain-a"
            )

        self.assertEqual(
            graph.updates,
            [
                (
                    "rel-1",
                    "relationship",
                    {
                        "invalid_at": "2024-02-02T00:00:00Z",
                        "deprecated": True,
                        "valid_to": "2024-02-02T00:00:00Z",
                        "superseded_by": "rel-2",
                        "retired_reason": "single-valued predicate replaced",
                    },
                )
            ],
        )
        self.assertFalse(retrieve_mod._is_currently_valid(graph.predicate("rel-1")))
        self.assertTrue(retrieve_mod._is_currently_valid(graph.predicate("rel-2")))

    def test_inbound_edge_on_the_subject_is_not_superseded(self):
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("alice", "Alice", "PERSON")
        graph.add_node("bob", "Bob", "PERSON")
        graph.add_node("carol", "Carol", "PERSON")
        graph.add_edge("rel-1", "bob", "MANAGES", "alice")
        graph.add_edge("rel-2", "alice", "MANAGES", "carol")

        relationship = _relationship(
            "rel-2",
            _entity("alice", "Alice", "PERSON"),
            "MANAGES",
            _entity("carol", "Carol", "PERSON"),
            valid_at="02/02/2024",
        )

        with patch.object(ingestion_mod, "graph_adapter", graph):
            ingestion_mod._invalidate_superseded_relationships(
                relationship, brain_id="brain-a"
            )

        self.assertEqual(graph.updates, [])

    def test_works_with_many_people_is_not_superseded(self):
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("alice", "Alice", "PERSON")
        graph.add_node("bob", "Bob", "PERSON")
        graph.add_node("carol", "Carol", "PERSON")
        graph.add_edge("rel-1", "alice", "WORKS_WITH", "bob")
        graph.add_edge("rel-2", "alice", "WORKS_WITH", "carol")

        relationship = _relationship(
            "rel-2",
            _entity("alice", "Alice", "PERSON"),
            "WORKS_WITH",
            _entity("carol", "Carol", "PERSON"),
            valid_at="02/02/2024",
        )
        with patch.object(ingestion_mod, "graph_adapter", graph):
            ingestion_mod._invalidate_superseded_relationships(
                relationship, brain_id="brain-a"
            )
        self.assertEqual(graph.updates, [])

    def test_requires_many_clauses_is_not_superseded(self):
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("rule", "Rule 12", "RULE")
        graph.add_node("id-check", "Identity check", "CLAUSE")
        graph.add_node("retention", "Retention period", "CLAUSE")
        graph.add_edge("rel-1", "rule", "REQUIRES", "id-check")
        graph.add_edge("rel-2", "rule", "REQUIRES", "retention")

        relationship = _relationship(
            "rel-2",
            _entity("rule", "Rule 12", "RULE"),
            "REQUIRES",
            _entity("retention", "Retention period", "CLAUSE"),
            valid_at="2025-01-01T00:00:00Z",
        )
        with patch.object(ingestion_mod, "graph_adapter", graph):
            ingestion_mod._invalidate_superseded_relationships(
                relationship, brain_id="brain-a"
            )
        self.assertEqual(graph.updates, [])

    def test_applies_to_many_plans_is_not_superseded(self):
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("policy", "Policy 12", "POLICY")
        graph.add_node("plan-a", "Plan A", "PLAN")
        graph.add_node("plan-b", "Plan B", "PLAN")
        graph.add_edge("rel-1", "policy", "APPLIES_TO", "plan-a")
        graph.add_edge("rel-2", "policy", "APPLIES_TO", "plan-b")

        relationship = _relationship(
            "rel-2",
            _entity("policy", "Policy 12", "POLICY"),
            "APPLIES_TO",
            _entity("plan-b", "Plan B", "PLAN"),
            valid_at="2025-01-01T00:00:00Z",
        )
        with patch.object(ingestion_mod, "graph_adapter", graph):
            ingestion_mod._invalidate_superseded_relationships(
                relationship, brain_id="brain-a"
            )
        self.assertEqual(graph.updates, [])

    def test_source_revision_closes_multi_valued_edge_from_superseded_chunk(self):
        from src.services.api.controllers import retrieve as retrieve_mod
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("alice", "Alice", "PERSON")
        graph.add_node("bob", "Bob", "PERSON")
        graph.add_node("dana", "Dana", "PERSON")
        graph.add_edge(
            "rel-1",
            "alice",
            "WORKS_WITH",
            "bob",
            properties={"source_chunk_ids": ["chunk-v1"]},
        )
        graph.add_edge(
            "rel-2",
            "alice",
            "WORKS_WITH",
            "dana",
            properties={"source_chunk_ids": ["chunk-v2"]},
        )
        relationship = _relationship(
            "rel-2",
            _entity("alice", "Alice", "PERSON"),
            "WORKS_WITH",
            _entity("dana", "Dana", "PERSON"),
            valid_at="2025-01-01T00:00:00Z",
        )
        with patch.object(ingestion_mod, "graph_adapter", graph):
            ingestion_mod._invalidate_superseded_relationships(
                relationship,
                brain_id="brain-a",
                superseded_chunk_ids=["chunk-v1"],
            )
        self.assertEqual(len(graph.updates), 1)
        self.assertEqual(graph.updates[0][0], "rel-1")
        self.assertEqual(
            graph.updates[0][2].get("retired_reason"),
            "source revision replaced this edge",
        )
        self.assertFalse(retrieve_mod._is_currently_valid(graph.predicate("rel-1")))
        self.assertTrue(retrieve_mod._is_currently_valid(graph.predicate("rel-2")))

    def test_reobserving_closed_single_valued_edge_gets_a_new_interval_id(self):
        from src.core.saving.identity import stable_relationship_id
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("alice", "Alice", "PERSON")
        graph.add_node("paris", "Paris", "CITY")
        closed_id = stable_relationship_id("alice", "LIVES_IN", "paris")
        graph.add_edge(
            closed_id,
            "alice",
            "LIVES_IN",
            "paris",
            properties={
                "valid_from": "2020-01-01T00:00:00Z",
                "valid_to": "2023-01-01T00:00:00Z",
            },
        )
        relationship = _relationship(
            closed_id,
            _entity("alice", "Alice", "PERSON"),
            "LIVES_IN",
            _entity("paris", "Paris", "CITY"),
            valid_at="2024-01-01T00:00:00Z",
        )
        relationship.properties["valid_from"] = "2024-01-01T00:00:00Z"
        with patch.object(ingestion_mod, "graph_adapter", graph):
            ingestion_mod._assign_relationship_interval(
                relationship, brain_id="brain-a"
            )
        self.assertNotEqual(relationship.uuid, closed_id)
        self.assertEqual(graph.predicate(closed_id).properties["valid_from"], "2020-01-01T00:00:00Z")
        self.assertEqual(graph.predicate(closed_id).properties["valid_to"], "2023-01-01T00:00:00Z")

    def test_undated_reopening_does_not_reuse_closed_interval_id(self):
        from src.core.saving.identity import stable_relationship_id
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("alice", "Alice", "PERSON")
        graph.add_node("paris", "Paris", "CITY")
        closed_id = stable_relationship_id("alice", "LIVES_IN", "paris")
        graph.add_edge(
            closed_id,
            "alice",
            "LIVES_IN",
            "paris",
            properties={"valid_to": "2023-01-01T00:00:00Z"},
        )
        relationship = _relationship(
            closed_id,
            _entity("alice", "Alice", "PERSON"),
            "LIVES_IN",
            _entity("paris", "Paris", "CITY"),
        )
        with patch.object(ingestion_mod, "graph_adapter", graph):
            ingestion_mod._assign_relationship_interval(
                relationship, brain_id="brain-a"
            )
            retry = _relationship(
                closed_id,
                _entity("alice", "Alice", "PERSON"),
                "LIVES_IN",
                _entity("paris", "Paris", "CITY"),
            )
            ingestion_mod._assign_relationship_interval(retry, brain_id="brain-a")
        self.assertNotEqual(relationship.uuid, closed_id)
        self.assertEqual(retry.uuid, relationship.uuid)

        graph.add_edge(
            relationship.uuid,
            "alice",
            "LIVES_IN",
            "paris",
            properties={"valid_to": "2024-01-01T00:00:00Z"},
        )
        later = _relationship(
            closed_id,
            _entity("alice", "Alice", "PERSON"),
            "LIVES_IN",
            _entity("paris", "Paris", "CITY"),
        )
        with patch.object(ingestion_mod, "graph_adapter", graph):
            ingestion_mod._assign_relationship_interval(later, brain_id="brain-a")
        self.assertNotEqual(later.uuid, closed_id)
        self.assertNotEqual(later.uuid, relationship.uuid)

    def test_open_same_endpoint_edge_keeps_original_valid_from(self):
        from src.core.saving.identity import stable_relationship_id
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("alice", "Alice", "PERSON")
        graph.add_node("paris", "Paris", "CITY")
        open_id = stable_relationship_id("alice", "LIVES_IN", "paris")
        graph.add_edge(
            open_id,
            "alice",
            "LIVES_IN",
            "paris",
            properties={"valid_from": "2020-01-01T00:00:00Z"},
        )
        relationship = _relationship(
            open_id,
            _entity("alice", "Alice", "PERSON"),
            "LIVES_IN",
            _entity("paris", "Paris", "CITY"),
            valid_at="2024-06-01T00:00:00Z",
        )
        relationship.properties["valid_from"] = "2024-06-01T00:00:00Z"
        with patch.object(ingestion_mod, "graph_adapter", graph):
            ingestion_mod._assign_relationship_interval(
                relationship, brain_id="brain-a"
            )
        self.assertEqual(relationship.uuid, open_id)
        self.assertEqual(relationship.properties["valid_from"], "2020-01-01T00:00:00Z")

    def test_inbound_reverse_edge_is_not_reused_as_open_match(self):
        from src.core.saving.identity import stable_relationship_id
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("alice", "Alice", "PERSON")
        graph.add_node("bob", "Bob", "PERSON")
        reverse_id = "rev-works-with"
        graph.add_edge(
            reverse_id,
            "bob",
            "WORKS_WITH",
            "alice",
            properties={"valid_from": "2020-01-01T00:00:00Z"},
        )
        incoming = stable_relationship_id("alice", "WORKS_WITH", "bob", "flow-1")
        relationship = _relationship(
            incoming,
            _entity("alice", "Alice", "PERSON"),
            "WORKS_WITH",
            _entity("bob", "Bob", "PERSON"),
            valid_at="2024-06-01T00:00:00Z",
        )
        relationship.properties["valid_from"] = "2024-06-01T00:00:00Z"
        with patch.object(ingestion_mod, "graph_adapter", graph):
            ingestion_mod._assign_relationship_interval(
                relationship, brain_id="brain-a"
            )
        self.assertEqual(relationship.uuid, incoming)
        self.assertNotEqual(relationship.uuid, reverse_id)

    def test_distinct_flow_keys_keep_separate_open_edges(self):
        from src.core.saving.identity import stable_relationship_id
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("alice", "Alice", "PERSON")
        graph.add_node("bob", "Bob", "PERSON")
        existing_id = "works-with-flow-a"
        graph.add_edge(
            existing_id,
            "alice",
            "WORKS_WITH",
            "bob",
            properties={
                "valid_from": "2020-01-01T00:00:00Z",
                "flow_key": "flow-a",
            },
        )
        incoming = stable_relationship_id("alice", "WORKS_WITH", "bob", "flow-b")
        relationship = _relationship(
            incoming,
            _entity("alice", "Alice", "PERSON"),
            "WORKS_WITH",
            _entity("bob", "Bob", "PERSON"),
            valid_at="2024-06-01T00:00:00Z",
        )
        relationship.flow_key = "flow-b"
        relationship.properties["valid_from"] = "2024-06-01T00:00:00Z"
        with patch.object(ingestion_mod, "graph_adapter", graph):
            ingestion_mod._assign_relationship_interval(
                relationship, brain_id="brain-a"
            )
        self.assertEqual(relationship.uuid, incoming)
        self.assertNotEqual(relationship.uuid, existing_id)

    def test_neighbor_lookup_failure_propagates(self):
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("alice", "Alice", "PERSON")
        graph.add_node("paris", "Paris", "CITY")
        graph.get_neighbors = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("graph down")
        )
        relationship = _relationship(
            "rel-closed",
            _entity("alice", "Alice", "PERSON"),
            "LIVES_IN",
            _entity("paris", "Paris", "CITY"),
            valid_at="2024-06-01T00:00:00Z",
        )
        with patch.object(ingestion_mod, "graph_adapter", graph):
            with self.assertRaises(RuntimeError):
                ingestion_mod._assign_relationship_interval(
                    relationship, brain_id="brain-a"
                )
            with self.assertRaises(RuntimeError):
                ingestion_mod._invalidate_superseded_relationships(
                    relationship, brain_id="brain-a"
                )

    def test_assign_and_invalidate_reuse_one_neighbor_fetch(self):
        from src.workers.tasks import ingestion as ingestion_mod

        graph = _FakeGraphAdapter()
        graph.add_node("alice", "Alice", "PERSON")
        graph.add_node("paris", "Paris", "CITY")
        graph.add_node("lyon", "Lyon", "CITY")
        graph.add_edge("rel-old", "alice", "LIVES_IN", "lyon")
        calls = {"n": 0}
        real_get = graph.get_neighbors

        def counting_get(*args, **kwargs):
            calls["n"] += 1
            return real_get(*args, **kwargs)

        graph.get_neighbors = counting_get
        relationship = _relationship(
            "rel-new",
            _entity("alice", "Alice", "PERSON"),
            "LIVES_IN",
            _entity("paris", "Paris", "CITY"),
            valid_at="2024-06-01T00:00:00Z",
        )
        with patch.object(ingestion_mod, "graph_adapter", graph):
            neighbors = ingestion_mod._relationship_neighbor_map(
                relationship, brain_id="brain-a"
            )
            ingestion_mod._assign_relationship_interval(
                relationship, brain_id="brain-a", neighbors=neighbors
            )
            ingestion_mod._invalidate_superseded_relationships(
                relationship, brain_id="brain-a", neighbors=neighbors
            )
        self.assertEqual(calls["n"], 1)
        self.assertEqual(graph.updates[0][0], "rel-old")


if __name__ == "__main__":
    unittest.main()

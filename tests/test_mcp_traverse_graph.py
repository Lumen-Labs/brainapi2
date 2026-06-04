import os
import sys
import unittest

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
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters.graph import GraphAdapter
from src.constants.kg import Node, Predicate
from src.core.search.traverse import flatten_neighborhood, MAX_TRAVERSE_DEPTH, MAX_TRAVERSE_HOPS


def _neighbor(
    pred_name: str,
    direction: str,
    node_uuid: str,
    node_name: str,
    labels: list[str],
    nested: list[dict] | None = None,
) -> dict:
    return {
        "predicate": Predicate(name=pred_name, description="", direction=direction),
        "node": Node(uuid=node_uuid, name=node_name, labels=labels),
        "neighbors": nested or [],
    }


class FlattenNeighborhoodTests(unittest.TestCase):
    def test_flattens_nested_neighbors(self):
        neighborhood = [
            _neighbor(
                "KNOWS",
                "out",
                "b",
                "Bob",
                ["PERSON"],
                nested=[_neighbor("WORKS_AT", "out", "c", "Acme", ["ORGANIZATION"])],
            )
        ]
        hops, truncated = flatten_neighborhood(
            neighborhood,
            rel_types=None,
            node_labels=None,
            direction="both",
            limit=50,
        )
        self.assertFalse(truncated)
        self.assertEqual(len(hops), 2)
        self.assertEqual(hops[0]["depth"], 1)
        self.assertEqual(hops[0]["node"]["uuid"], "b")
        self.assertEqual(hops[1]["depth"], 2)
        self.assertEqual(hops[1]["via_uuid"], "b")
        self.assertEqual(hops[1]["node"]["uuid"], "c")

    def test_filters_by_rel_type(self):
        neighborhood = [
            _neighbor("KNOWS", "out", "b", "Bob", ["PERSON"]),
            _neighbor("WORKS_AT", "out", "c", "Acme", ["ORGANIZATION"]),
        ]
        hops, _ = flatten_neighborhood(
            neighborhood,
            rel_types=["WORKS_AT"],
            node_labels=None,
            direction="both",
            limit=50,
        )
        self.assertEqual(len(hops), 1)
        self.assertEqual(hops[0]["predicate"]["name"], "WORKS_AT")

    def test_filters_by_node_labels(self):
        neighborhood = [
            _neighbor("KNOWS", "out", "b", "Bob", ["PERSON"]),
            _neighbor("LOCATED_IN", "out", "c", "Paris", ["LOCATION"]),
        ]
        hops, _ = flatten_neighborhood(
            neighborhood,
            rel_types=None,
            node_labels=["LOCATION"],
            direction="both",
            limit=50,
        )
        self.assertEqual(len(hops), 1)
        self.assertEqual(hops[0]["node"]["labels"], ["LOCATION"])

    def test_filters_by_direction(self):
        neighborhood = [
            _neighbor("KNOWS", "out", "b", "Bob", ["PERSON"]),
            _neighbor("KNOWS", "in", "c", "Carol", ["PERSON"]),
        ]
        hops, _ = flatten_neighborhood(
            neighborhood,
            rel_types=None,
            node_labels=None,
            direction="out",
            limit=50,
        )
        self.assertEqual(len(hops), 1)
        self.assertEqual(hops[0]["direction"], "out")

    def test_respects_limit_and_sets_truncated(self):
        neighborhood = [
            _neighbor("REL", "out", f"n{i}", f"Node{i}", ["THING"])
            for i in range(5)
        ]
        hops, truncated = flatten_neighborhood(
            neighborhood,
            rel_types=None,
            node_labels=None,
            direction="both",
            limit=2,
        )
        self.assertTrue(truncated)
        self.assertEqual(len(hops), 2)


class GraphAdapterTraverseGraphTests(unittest.TestCase):
    def test_traverse_graph_by_uuid(self):
        start = Node(uuid="a", name="Alice", labels=["PERSON"])
        neighborhood = [_neighbor("KNOWS", "out", "b", "Bob", ["PERSON"])]

        class FakeGraphClient:
            def get_by_uuid(self, uuid, brain_id):
                return start if uuid == "a" else None

            def get_neighborhood(self, node, depth, brain_id):
                return neighborhood

        adapter = GraphAdapter()
        adapter.add_client(FakeGraphClient())
        result = adapter.traverse_graph(start_uuid="a", brain_id="default", depth=2)
        self.assertEqual(result["start"]["uuid"], "a")
        self.assertEqual(len(result["hops"]), 1)
        self.assertFalse(result["truncated"])

    def test_traverse_graph_by_name_and_labels(self):
        start = Node(uuid="a", name="Alice", labels=["PERSON"])
        neighborhood = []

        class FakeGraphClient:
            def get_by_identification_params(self, params, brain_id, entity_types=None):
                return start

            def get_neighborhood(self, node, depth, brain_id):
                return neighborhood

        adapter = GraphAdapter()
        adapter.add_client(FakeGraphClient())
        result = adapter.traverse_graph(
            start_name="Alice",
            start_labels=["PERSON"],
            brain_id="default",
        )
        self.assertEqual(result["start"]["name"], "Alice")

    def test_traverse_graph_requires_start_identifiers(self):
        adapter = GraphAdapter()
        adapter.add_client(type("Fake", (), {})())
        result = adapter.traverse_graph(brain_id="default")
        self.assertIn("error", result)

    def test_traverse_graph_clamps_depth_and_limit(self):
        start = Node(uuid="a", name="Alice", labels=["PERSON"])

        class FakeGraphClient:
            captured_depth = None

            def get_by_uuid(self, uuid, brain_id):
                return start

            def get_neighborhood(self, node, depth, brain_id):
                FakeGraphClient.captured_depth = depth
                return []

        adapter = GraphAdapter()
        adapter.add_client(FakeGraphClient())
        adapter.traverse_graph(
            start_uuid="a",
            depth=99,
            limit=999,
        )
        self.assertEqual(FakeGraphClient.captured_depth, MAX_TRAVERSE_DEPTH)


class TraverseConstantsTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(MAX_TRAVERSE_DEPTH, 5)
        self.assertEqual(MAX_TRAVERSE_HOPS, 100)


if __name__ == "__main__":
    unittest.main()

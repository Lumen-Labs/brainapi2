import os
import unittest
from unittest.mock import MagicMock, patch


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


class NetworkXUpsertContractTests(unittest.TestCase):
    def test_same_uuid_upsert_keeps_one_node_and_preserves_uuid(self):
        from src.lib.postgresql.graph_store import _BrainGraph

        brain = _BrainGraph("brain-a")
        first = brain.upsert_node(
            ["PERSON"],
            {"uuid": "alice-1", "name": "Alice"},
            {"uuid": "alice-1", "name": "Alice", "description": "v1"},
        )
        second = brain.upsert_node(
            ["PERSON"],
            {"uuid": "alice-1", "name": "Alice"},
            {"uuid": "alice-1", "name": "Alice", "description": "v2"},
        )
        self.assertEqual(first, "alice-1")
        self.assertEqual(second, "alice-1")
        self.assertEqual(len(list(brain.graph.nodes)), 1)
        self.assertEqual(brain.node_data("alice-1")["uuid"], "alice-1")
        self.assertEqual(brain.node_data("alice-1")["description"], "v1 | v2")

    def test_same_name_different_uuid_does_not_overwrite_existing_uuid(self):
        from src.lib.postgresql.graph_store import _BrainGraph

        brain = _BrainGraph("brain-a")
        brain.upsert_node(
            ["PERSON"],
            {"uuid": "alice-1", "name": "Alice"},
            {"uuid": "alice-1", "name": "Alice"},
        )
        brain.upsert_node(
            ["PERSON"],
            {"uuid": "alice-2", "name": "Alice"},
            {"uuid": "alice-2", "name": "Alice"},
        )
        self.assertEqual(len(list(brain.graph.nodes)), 2)
        self.assertEqual(brain.node_data("alice-1")["uuid"], "alice-1")
        self.assertEqual(brain.node_data("alice-2")["uuid"], "alice-2")

    def test_relationship_upsert_by_uuid_is_idempotent(self):
        from src.lib.postgresql.graph_store import PostgreSQLGraphStore

        store = PostgreSQLGraphStore.__new__(PostgreSQLGraphStore)
        from src.lib.postgresql.graph_store import _BrainGraph

        brain = _BrainGraph("brain-a")
        brain.upsert_node(["PERSON"], {"uuid": "a"}, {"uuid": "a", "name": "Alice"})
        brain.upsert_node(["PRODUCT"], {"uuid": "b"}, {"uuid": "b", "name": "Widget"})
        store._brains = {"brain-a": brain}
        store._brains_lock = __import__("threading").RLock()
        store._persist_relationship = MagicMock()
        store._persist_node = MagicMock()

        first = store.merge_relationship(
            "brain-a",
            ["PERSON"],
            "Alice",
            ["PRODUCT"],
            "Widget",
            "OWNS",
            {"uuid": "rel-1", "description": "v1"},
            subject_uuid="a",
            object_uuid="b",
        )
        second = store.merge_relationship(
            "brain-a",
            ["PERSON"],
            "Alice",
            ["PRODUCT"],
            "Widget",
            "OWNS",
            {"uuid": "rel-1", "description": "v2"},
            subject_uuid="a",
            object_uuid="b",
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        edges = list(brain.graph.edges(keys=True, data=True))
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][2], "rel-1")
        self.assertEqual(edges[0][3]["description"], "v2")

    def test_relationship_upsert_creates_missing_endpoint_nodes(self):
        from src.lib.postgresql.graph_store import PostgreSQLGraphStore
        from src.lib.postgresql.graph_store import _BrainGraph

        store = PostgreSQLGraphStore.__new__(PostgreSQLGraphStore)
        brain = _BrainGraph("brain-a")
        store._brains = {"brain-a": brain}
        store._brains_lock = __import__("threading").RLock()
        store._persist_node = MagicMock()
        store._persist_relationship = MagicMock()

        result = store.merge_relationship(
            "brain-a",
            ["PERSON"],
            "Alice",
            ["PRODUCT"],
            "Widget",
            "OWNS",
            {"uuid": "rel-1", "description": "created"},
            subject_uuid="a",
            object_uuid="b",
        )
        self.assertIsNotNone(result)
        self.assertEqual(set(brain.graph.nodes), {"a", "b"})
        self.assertEqual(len(list(brain.graph.edges(keys=True))), 1)
        store._persist_node.assert_called()
        store._persist_relationship.assert_called_once()

    def test_normalize_read_sql_rewrites_jsonb_ilike(self):
        from src.lib.postgresql.graph_store import normalize_read_sql

        sql = (
            "SELECT uuid, data FROM kg_nodes WHERE data->>'name' ILIKE '%Reminding%' "
            "OR data->'description' ILIKE '%remind%' LIMIT 50"
        )
        fixed = normalize_read_sql(sql)
        self.assertIn("data->>'description' ILIKE", fixed)
        self.assertNotIn("data->'description' ILIKE", fixed)

    def test_stale_cache_reloads_when_db_counts_diverge(self):
        from src.lib.postgresql.graph_store import PostgreSQLGraphStore
        from src.lib.postgresql.graph_store import _BrainGraph

        store = PostgreSQLGraphStore.__new__(PostgreSQLGraphStore)
        stale = _BrainGraph("brain-a")
        store._brains = {"brain-a": stale}
        store._brains_lock = __import__("threading").RLock()
        store._db_graph_state = MagicMock(return_value=(2, 1, 0))

        fresh = _BrainGraph("brain-a")
        fresh.graph.add_node("a", name="Alice", labels=["PERSON"])
        fresh.graph.add_node("b", name="Bob", labels=["PERSON"])
        fresh.graph.add_edge("a", "b", key="r1", rel_type="KNOWS")
        store._reload_brain_locked = MagicMock(return_value=fresh)

        loaded = store._load_brain("brain-a")
        self.assertIs(loaded, fresh)
        store._reload_brain_locked.assert_called_once_with("brain-a")


class Neo4jUpsertContractTests(unittest.TestCase):
    def test_add_nodes_merges_by_uuid_not_name(self):
        from src.constants.kg import Node
        from src.lib.neo4j.client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.ensure_database = MagicMock()
        queries = []

        def _capture(query, brain_id):
            queries.append(query)
            return "ok"

        client._execute_query_with_retry = _capture
        client._clean_labels = lambda labels: [str(l).upper() for l in labels]
        client._format_property_key = lambda key: key
        client._format_value = lambda value: (
            "null"
            if value is None
            else f"'{value}'"
            if isinstance(value, str)
            else str(value)
        )
        client._clean_property_key = lambda key: key

        node = Node(
            uuid="alice-1",
            name="Alice",
            labels=["PERSON"],
            description="v1",
            properties={},
        )
        client.add_nodes([node], "brain-a")
        self.assertEqual(len(queries), 1)
        self.assertIn("MERGE (n:PERSON {uuid: 'alice-1'})", queries[0])
        self.assertNotIn("{name:", queries[0])

    def test_add_relationship_merges_by_endpoint_and_relationship_uuid(self):
        from src.constants.kg import Node, Predicate
        from src.lib.neo4j.client import Neo4jClient

        client = Neo4jClient.__new__(Neo4jClient)
        client.ensure_database = MagicMock()
        client.driver = MagicMock()
        client._clean_labels = lambda labels: [str(l).upper() for l in labels]
        client._format_value = lambda value: (
            "null"
            if value is None
            else f"'{value}'"
            if isinstance(value, str)
            else str(value)
        )

        subject = Node(uuid="a", name="Alice", labels=["PERSON"], properties={})
        obj = Node(uuid="b", name="Widget", labels=["PRODUCT"], properties={})
        predicate = Predicate(
            uuid="rel-1",
            name="OWNS",
            description="owns",
            properties={"v_id": "v1"},
            flow_key="flow-1",
        )
        client.add_relationship(subject, predicate, obj, "brain-a")
        query = client.driver.execute_query.call_args.args[0]
        self.assertIn("WHERE a['uuid'] = 'a'", query)
        self.assertIn("WHERE b['uuid'] = 'b'", query)
        self.assertIn("MERGE (a)-[r:OWNS {uuid: 'rel-1'}]->(b)", query)


if __name__ == "__main__":
    unittest.main()

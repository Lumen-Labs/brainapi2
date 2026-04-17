import json
import os
import unittest
from pathlib import Path

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

from src.adapters.graph import GraphAdapter
from src.adapters.graph_operation_result_serializer import Neo4jResultSerializer
from src.constants.kg import Node, Predicate
from src.constants.tasks.ingestion import IngestionTaskJsonArgs, IngestionTaskTextArgs
from src.core.ingestion.text_strategy import (
    IngestionTextStrategyFactory,
    extract_ingestion_text,
)


class FakeVector:
    def __init__(self, embeddings, metadata):
        self.embeddings = embeddings
        self.metadata = metadata


class IngestionTextStrategyTests(unittest.TestCase):
    def test_extract_ingestion_text_uses_raw_text_payload(self):
        payload = IngestionTaskTextArgs(text_data="plain text")
        self.assertEqual(extract_ingestion_text(payload), "plain text")

    def test_extract_ingestion_text_uses_json_payload(self):
        payload = IngestionTaskJsonArgs(json_data={"k": "v"})
        self.assertEqual(extract_ingestion_text(payload), json.dumps({"k": "v"}))

    def test_factory_raises_for_unknown_payload_type(self):
        factory = IngestionTextStrategyFactory()
        with self.assertRaises(ValueError):
            factory.create(object())


class IngestionTaskSourceRefactorTests(unittest.TestCase):
    def test_ingest_data_uses_shared_payload_text_for_enrichment(self):
        source = (
            Path(__file__).resolve().parent.parent / "src/workers/tasks/ingestion.py"
        ).read_text(encoding="utf-8")
        self.assertIn("payload_text = extract_ingestion_text(payload.data)", source)
        self.assertIn(
            "enrich_kg_from_input(payload_text, brain_id=payload.brain_id)", source
        )
        self.assertNotIn(
            "enrich_kg_from_input(payload.data.text_data, brain_id=payload.brain_id)",
            source,
        )


class GraphAdapterNeighborAssemblyTests(unittest.TestCase):
    def test_vectors_with_descriptions_uses_node_descriptions(self):
        adapter = GraphAdapter()
        predicate = Predicate(name="RELATES_TO", description="desc")
        neighbors = [
            (
                predicate,
                Node(uuid="node-a", labels=["Person"], name="Alice", description="A"),
            ),
            (
                predicate,
                Node(uuid="node-b", labels=["Person"], name="Bob", description=None),
            ),
            (
                predicate,
                Node(
                    uuid="node-c",
                    labels=["Person"],
                    name="Charlie",
                    description="C",
                ),
            ),
        ]
        vectors = {
            "node-a": FakeVector([1.0, 0.0], {"uuid": "node-a"}),
            "node-b": FakeVector([0.0, 1.0], {"uuid": "node-b"}),
        }
        result = adapter._vectors_with_descriptions(neighbors, vectors)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["description"], "A")
        self.assertIsNone(result[1]["description"])

    def test_node_description_returns_none_for_missing_node(self):
        adapter = GraphAdapter()
        self.assertIsNone(adapter._node_description(None))


class Neo4jResultSerializerTests(unittest.TestCase):
    def test_serializer_falls_back_to_string_for_non_mapping_records(self):
        class NonMappingRecord:
            def __str__(self):
                return "non-mapping-record"

        class FakeResult:
            def __init__(self):
                self.records = [NonMappingRecord()]

            def keys(self):
                return ["value"]

        serializer = Neo4jResultSerializer()
        with self.assertLogs(
            "src.adapters.graph_operation_result_serializer", level="WARNING"
        ) as captured_logs:
            payload = json.loads(serializer.serialize(FakeResult()))
        self.assertEqual(payload["records"], ["non-mapping-record"])
        self.assertEqual(payload["keys"], ["value"])
        self.assertTrue(
            any("Failed to map neo4j record" in message for message in captured_logs.output)
        )

    def test_serializer_supports_iterable_keys_attribute(self):
        class FakeResult:
            def __init__(self):
                self.records = []
                self.keys = ("a", "b")

        serializer = Neo4jResultSerializer()
        payload = json.loads(serializer.serialize(FakeResult()))
        self.assertEqual(payload["keys"], ["a", "b"])
        self.assertFalse(payload["truncated"])


if __name__ == "__main__":
    unittest.main()

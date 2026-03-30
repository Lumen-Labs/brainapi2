import json
import os
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
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from src.adapters.embeddings import (
    EmbeddingError,
    EmbeddingsAdapter,
    RaiseEmbeddingFailureStrategy,
)
from src.adapters.graph import GraphAdapter
from src.constants.embeddings import Vector
from src.core.agents.core.runtime_agent_factory import RuntimeAgentFactory
from src.services.api.controllers.entities import get_entity_status


def raise_embedding_error(*args, **kwargs):
    raise EmbeddingError("embedding failed")


class RuntimeAgentFactoryTests(unittest.TestCase):
    def test_factory_uses_custom_backend_when_forced(self):
        class FakeCustomAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def fake_create_agent(**kwargs):
            return {"kind": "langchain", "kwargs": kwargs}

        factory = RuntimeAgentFactory(
            create_agent_fn=fake_create_agent, custom_agent_cls=FakeCustomAgent
        )
        built = factory.build(
            model=object(),
            tools=[],
            system_prompt="system",
            output_schema=dict,
            architecture="langchain",
            use_custom_backend=True,
            debug=True,
        )
        self.assertIsInstance(built, FakeCustomAgent)
        self.assertEqual(built.kwargs["output_schema"], dict)
        self.assertEqual(built.kwargs["system_prompt"], "system")

    def test_factory_uses_langchain_backend_when_supported(self):
        class FakeCustomAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def fake_create_agent(**kwargs):
            return {"kind": "langchain", "kwargs": kwargs}

        factory = RuntimeAgentFactory(
            create_agent_fn=fake_create_agent, custom_agent_cls=FakeCustomAgent
        )
        built = factory.build(
            model=object(),
            tools=["t1"],
            system_prompt="system",
            output_schema=dict,
            architecture="langchain",
            use_custom_backend=False,
            debug=False,
        )
        self.assertEqual(built["kind"], "langchain")
        self.assertEqual(built["kwargs"]["response_format"], dict)
        self.assertEqual(built["kwargs"]["tools"], ["t1"])


class EmbeddingsStrategyTests(unittest.TestCase):
    def test_embeddings_default_strategy_returns_empty_vectors(self):
        adapter = EmbeddingsAdapter()
        adapter._embed_text_with_retry = raise_embedding_error
        adapter._embed_texts_with_retry = raise_embedding_error
        single = adapter.embed_text("hello")
        many = adapter.embed_texts(["a", "b"])
        self.assertEqual(single.embeddings, [])
        self.assertEqual(len(many), 2)
        self.assertTrue(all(v.embeddings == [] for v in many))

    def test_embeddings_raise_strategy_raises_error(self):
        adapter = EmbeddingsAdapter()
        adapter._embed_text_with_retry = raise_embedding_error
        adapter._embed_texts_with_retry = raise_embedding_error
        adapter.set_failure_strategy(RaiseEmbeddingFailureStrategy())
        with self.assertRaises(EmbeddingError):
            adapter.embed_text("hello")
        with self.assertRaises(EmbeddingError):
            adapter.embed_texts(["a"])


class GraphAdapterReductionTests(unittest.TestCase):
    def test_reduce_neighbor_vectors_without_description(self):
        adapter = GraphAdapter()
        vectors = [
            {"metadata": {"uuid": "a"}, "embeddings": [1.0, 0.0], "description": None},
            {"metadata": {"uuid": "b"}, "embeddings": [0.0, 1.0], "description": None},
        ]
        reduced = adapter._reduce_neighbor_vectors(
            vectors_with_desc=vectors,
            averaged_vector=[1.0, 0.0],
            similarity_threshold=0.5,
            description=None,
        )
        self.assertEqual(reduced, {"a"})

    def test_reduce_neighbor_vectors_with_description_reranks(self):
        adapter = GraphAdapter()
        vectors = [
            {"metadata": {"uuid": "a"}, "embeddings": [1.0, 0.0], "description": "d1"},
            {"metadata": {"uuid": "b"}, "embeddings": [1.0, 0.0], "description": "d2"},
        ]
        with patch("src.adapters.graph.reduce_list", return_value=[vectors[1]]) as reduced:
            result = adapter._reduce_neighbor_vectors(
                vectors_with_desc=vectors,
                averaged_vector=[1.0, 0.0],
                similarity_threshold=0.0,
                description="focus",
            )
        reduced.assert_called_once()
        self.assertEqual(result, {"b"})

    def test_reduce_neighbor_vectors_skips_invalid_embeddings(self):
        adapter = GraphAdapter()
        vectors = [
            {"metadata": {"uuid": "a"}, "embeddings": [1.0, 0.0]},
            {"metadata": {"uuid": "b"}, "embeddings": [1.0]},
            {"metadata": {"uuid": "c"}, "embeddings": None},
            {"metadata": {}, "embeddings": [1.0, 0.0]},
        ]
        reduced = adapter._reduce_neighbor_vectors(
            vectors_with_desc=vectors,
            averaged_vector=[1.0, 0.0],
            similarity_threshold=0.5,
            description=None,
        )
        self.assertEqual(reduced, {"a"})

    def test_average_embeddings_ignores_invalid_and_mismatched_dimensions(self):
        adapter = GraphAdapter()
        vectors = [
            Vector(id="1", embeddings=[1.0, 2.0], metadata={}),
            Vector(id="2", embeddings=[3.0, 4.0], metadata={}),
            Vector(id="3", embeddings=[10.0], metadata={}),
            Vector(id="4", embeddings=None, metadata={}),
            Vector(id="5", embeddings=[], metadata={}),
        ]
        averaged = adapter._average_embeddings(vectors)
        self.assertEqual(averaged, [2.0, 3.0])


class GraphOperationSerializationTests(unittest.TestCase):
    def test_execute_operation_serializes_none_result(self):
        class FakeGraphClient:
            def execute_operation(self, operation: str, brain_id: str):
                return None

        adapter = GraphAdapter()
        adapter.add_client(FakeGraphClient())
        result = adapter.execute_operation("RETURN 1", brain_id="default")
        self.assertEqual(result, "")

    def test_execute_operation_serializes_mapping_result(self):
        class FakeGraphClient:
            def execute_operation(self, operation: str, brain_id: str):
                return {"operation": operation, "brain_id": brain_id, "ok": True}

        adapter = GraphAdapter()
        adapter.add_client(FakeGraphClient())
        result = adapter.execute_operation("RETURN 1", brain_id="test-brain")
        self.assertEqual(
            json.loads(result),
            {"operation": "RETURN 1", "brain_id": "test-brain", "ok": True},
        )

    def test_execute_operation_serializes_neo4j_like_result_with_limit(self):
        class FakeRecord:
            def __init__(self, value: int):
                self.value = value

            def data(self):
                return {"value": self.value}

        class FakeNeo4jResult:
            def __init__(self):
                self.records = [FakeRecord(value) for value in range(25)]

            def keys(self):
                return ["value"]

        class FakeGraphClient:
            def execute_operation(self, operation: str, brain_id: str):
                return FakeNeo4jResult()

        adapter = GraphAdapter()
        adapter.add_client(FakeGraphClient())
        result = adapter.execute_operation("MATCH (n) RETURN n", brain_id="default")
        payload = json.loads(result)
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["keys"], ["value"])
        self.assertEqual(len(payload["records"]), 20)
        self.assertEqual(payload["records"][0], {"value": 0})


class EntityStatusControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_entity_status_skips_missing_graph_nodes(self):
        fake_vector = type(
            "FakeVector",
            (),
            {"metadata": {"uuid": "missing-node"}, "embeddings": [0.1, 0.2]},
        )()
        with (
            patch(
                "src.services.api.controllers.entities.embeddings_adapter.embed_text",
                return_value=Vector(id="q", embeddings=[0.1, 0.2], metadata={}),
            ),
            patch(
                "src.services.api.controllers.entities.vector_store_adapter.search_vectors",
                return_value=[fake_vector],
            ),
            patch(
                "src.services.api.controllers.entities.graph_adapter.get_by_uuids",
                return_value=[],
            ),
        ):
            response = await get_entity_status("target", types=None, brain_id="default")
        self.assertFalse(response.exists)
        self.assertFalse(response.has_relationships)
        self.assertEqual(response.relationships, [])
        self.assertEqual(response.observations, [])


if __name__ == "__main__":
    unittest.main()

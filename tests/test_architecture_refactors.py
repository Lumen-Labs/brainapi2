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
from src.core.agents.core.runtime_agent_factory import RuntimeAgentFactory


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


if __name__ == "__main__":
    unittest.main()

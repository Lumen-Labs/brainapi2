import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock


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


class VectorSimilarityContractTests(unittest.TestCase):
    def test_adapter_sorts_nearest_first_by_cosine_distance(self):
        from src.adapters.embeddings import VectorStoreAdapter
        from src.constants.embeddings import Vector

        store = MagicMock()
        store.search_vectors.return_value = [
            Vector(id="far", metadata={}, distance=0.8),
            Vector(id="near", metadata={}, distance=0.05),
            Vector(id="mid", metadata={}, distance=0.3),
        ]
        adapter = VectorStoreAdapter()
        adapter.add_client(store)

        results = adapter.search_vectors([1.0, 0.0, 0.0], brain_id="b", store="nodes", k=3)
        self.assertEqual([v.id for v in results], ["near", "mid", "far"])

    def test_adapter_equal_distance_tiebreaks_by_uuid(self):
        from src.adapters.embeddings import VectorStoreAdapter
        from src.constants.embeddings import Vector

        store = MagicMock()
        store.search_vectors.return_value = [
            Vector(id="2", metadata={"uuid": "z"}, distance=0.1),
            Vector(id="1", metadata={"uuid": "a"}, distance=0.1),
        ]
        adapter = VectorStoreAdapter()
        adapter.add_client(store)
        results = adapter.search_vectors([1.0], brain_id="b", store="nodes", k=2)
        self.assertEqual([v.metadata["uuid"] for v in results], ["a", "z"])

    def test_pgvector_neighbor_search_binds_read_embeddings_as_python_floats(self):
        import numpy as np
        from psycopg2.extensions import adapt

        from src.lib.postgresql.vectors import PostgreSQLVectorStoreClient

        client = PostgreSQLVectorStoreClient.__new__(PostgreSQLVectorStoreClient)
        client._ensure_store = MagicMock()
        client._connection = MagicMock()
        connection = client._connection.return_value.__enter__.return_value
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.side_effect = [
            [{"id": 1, "embeddings": np.array([1, 0, 0], dtype=np.float32)}],
            [],
        ]

        self.assertEqual(client.search_similar_by_ids(["1"], "brain", "nodes", 0.5), {"1": []})
        _, params = cursor.execute.call_args.args
        self.assertEqual(adapt(params[0]).getquoted(), b"ARRAY[1.0,0.0,0.0]")

    def test_milvus_converts_similarity_score_to_cosine_distance(self):
        from src.constants.embeddings import Vector
        from src.lib.milvus.client import MilvusClient

        client = MilvusClient.__new__(MilvusClient)
        milvus = MagicMock()
        milvus.search.return_value = [
            [
                {
                    "id": 1,
                    "distance": 0.95,
                    "entity": {"uuid": "near"},
                },
                {
                    "id": 2,
                    "distance": 0.2,
                    "entity": {"uuid": "far"},
                },
            ]
        ]
        client._get_client = MagicMock(return_value=milvus)
        client._ensure_store = MagicMock()

        results = MilvusClient.search_vectors(
            client, [1.0, 0.0, 0.0], "brain-a", "nodes", k=2
        )
        self.assertEqual(results[0].id, "1")
        self.assertAlmostEqual(results[0].distance, 0.05)
        self.assertEqual(results[1].id, "2")
        self.assertAlmostEqual(results[1].distance, 0.8)
        self.assertTrue(isinstance(results[0], Vector))

    def test_relationship_dedup_skips_self_and_uses_distance_threshold(self):
        from src.workers.tasks.ingestion import (
            nearest_existing_vector,
            should_dedup_relationship,
        )

        self_hit = SimpleNamespace(id="v-self", distance=0.0, metadata={"uuid": "self"})
        near = SimpleNamespace(id="v-near", distance=0.05, metadata={"uuid": "near"})
        far = SimpleNamespace(id="v-far", distance=0.5, metadata={"uuid": "far"})

        nearest = nearest_existing_vector([self_hit, near, far], "v-self")
        self.assertEqual(nearest.id, "v-near")
        self.assertTrue(should_dedup_relationship(nearest))
        self.assertFalse(should_dedup_relationship(far))
        self.assertIsNone(nearest_existing_vector([self_hit], "v-self"))


if __name__ == "__main__":
    unittest.main()

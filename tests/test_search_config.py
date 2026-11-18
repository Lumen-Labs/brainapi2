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
    "PIPELINE_MODE": "accurate",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from src.config import (
    validate_context_passage_mode,
    validate_search_config,
    validate_search_fusion,
)


class SearchConfigValidationTests(unittest.TestCase):
    def test_defaults_are_search_off_and_fused(self):
        self.assertEqual(validate_search_fusion(None), "rrf")
        self.assertEqual(validate_context_passage_mode(None), "hybrid")
        validate_search_config(
            enabled=False,
            use_dense=False,
            use_bm25=False,
            data_db="mongo",
            bm25_k1=1.2,
            bm25_b=0.75,
        )

    def test_bm25_requires_postgres_data_db(self):
        with self.assertRaises(ValueError) as ctx:
            validate_search_config(
                enabled=True,
                use_dense=True,
                use_bm25=True,
                data_db="mongo",
                bm25_k1=1.2,
                bm25_b=0.75,
            )
        self.assertIn("DATA_DB=postgresql", str(ctx.exception))

    def test_dense_only_search_supports_non_postgres_data_db(self):
        validate_search_config(
            enabled=True,
            use_dense=True,
            use_bm25=False,
            data_db="mongo",
            bm25_k1=1.2,
            bm25_b=0.75,
        )

    def test_enabled_requires_at_least_one_channel(self):
        with self.assertRaises(ValueError) as ctx:
            validate_search_config(
                enabled=True,
                use_dense=False,
                use_bm25=False,
                data_db="postgresql",
                bm25_k1=1.2,
                bm25_b=0.75,
            )
        self.assertIn("SEARCH_USE_DENSE", str(ctx.exception))

    def test_invalid_fusion_and_passage_mode(self):
        with self.assertRaises(ValueError):
            validate_search_fusion("weighted")
        with self.assertRaises(ValueError):
            validate_context_passage_mode("splade")

    def test_config_reads_search_defaults(self):
        env = {
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
            "DATA_DB": "postgresql",
        }
        with patch.dict(os.environ, env, clear=False):
            for key in (
                "SEARCH_ENABLED",
                "SEARCH_USE_DENSE",
                "SEARCH_USE_BM25",
                "SEARCH_FUSION",
                "SEARCH_LITERAL_FILL",
                "CONTEXT_PASSAGE_MODE",
                "GRAPH_RAW_WRITE_ENABLED",
            ):
                os.environ.pop(key, None)
            from src.config import Config

            cfg = Config()
            self.assertFalse(cfg.search_enabled)
            self.assertFalse(cfg.graph_raw_write_enabled)
            self.assertTrue(cfg.search_use_dense)
            self.assertTrue(cfg.search_use_bm25)
            self.assertEqual(cfg.search_fusion, "rrf")
            self.assertEqual(cfg.search_bm25_k1, 1.2)
            self.assertEqual(cfg.search_bm25_b, 0.75)
            self.assertEqual(cfg.context_passage_mode, "hybrid")
            self.assertFalse(cfg.search_literal_fill)


if __name__ == "__main__":
    unittest.main()

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

from src.utils.tokens import pat_matches


class PatMatchesTests(unittest.TestCase):
    def test_equal_tokens_match(self):
        self.assertTrue(pat_matches("abc123", "abc123"))

    def test_unequal_tokens_do_not_match(self):
        self.assertFalse(pat_matches("abc123", "abc124"))

    def test_different_lengths_do_not_match(self):
        self.assertFalse(pat_matches("short", "much-longer-token"))

    def test_empty_or_missing_do_not_match(self):
        self.assertFalse(pat_matches("", "abc"))
        self.assertFalse(pat_matches("abc", ""))
        self.assertFalse(pat_matches(None, "abc"))
        self.assertFalse(pat_matches("abc", None))
        self.assertFalse(pat_matches("", ""))

    def test_uses_compare_digest(self):
        with patch("src.utils.tokens.hmac.compare_digest", return_value=True) as digest:
            self.assertTrue(pat_matches("token-a", "token-a"))
        digest.assert_called()
        left, right = digest.call_args.args
        self.assertEqual(left, b"token-a")
        self.assertEqual(right, b"token-a")


if __name__ == "__main__":
    unittest.main()

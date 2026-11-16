from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime
from types import SimpleNamespace

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
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "postgres",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.saving.history_intent import prefers_history
from src.services.api.controllers import retrieve as retrieve_mod
from src.utils.dates import parse_date_string


class HistoryModeTests(unittest.TestCase):
    def test_words_alone_do_not_select_history(self):
        self.assertFalse(
            retrieve_mod._wants_historical_facts(
                "Can you list the order in which I brought up aspects?"
            )
        )
        self.assertFalse(
            retrieve_mod._wants_historical_facts("com'era prima il codice")
        )

    def test_explicit_flag_overrides_similarity(self):
        history = [[1.0, 0.0]]
        current = [[0.0, 1.0]]
        self.assertTrue(
            retrieve_mod._wants_historical_facts(
                "what is the current value",
                explicit=True,
                query_vector=[0.0, 1.0],
                history_vectors=history,
                current_vectors=current,
            )
        )
        self.assertFalse(
            retrieve_mod._wants_historical_facts(
                "what did this say before",
                explicit=False,
                query_vector=[1.0, 0.0],
                history_vectors=history,
                current_vectors=current,
            )
        )

    def test_closer_history_example_selects_prior_revisions(self):
        self.assertTrue(
            prefers_history([1.0, 0.0], [[1.0, 0.0]], [[0.0, 1.0]])
        )
        self.assertFalse(
            prefers_history([0.0, 1.0], [[1.0, 0.0]], [[0.0, 1.0]])
        )
        self.assertFalse(prefers_history(None, [[1.0, 0.0]], [[0.0, 1.0]]))

    def test_fact_predicates_include_superseded_in_history_mode(self):
        current = SimpleNamespace(properties={}, deprecated=False)
        old = SimpleNamespace(properties={"invalid_at": "2024-03-31"}, deprecated=False)
        self.assertFalse(
            retrieve_mod._fact_predicates_allowed(old, current, include_history=False)
        )
        self.assertTrue(
            retrieve_mod._fact_predicates_allowed(old, current, include_history=True)
        )


class SessionIdParseTests(unittest.TestCase):
    def test_session_and_batch_turn_ids(self):
        text = "Session id: session_12. Unit id: b3_t2. Also session_4."
        ids = retrieve_mod._session_ids_from_text(text)
        self.assertIn("session_12", ids)
        self.assertIn("session_4", ids)
        self.assertIn("session_b3_t2", ids)


class BeamDateStampTests(unittest.TestCase):
    def test_hyphenated_beam_anchor_parses(self):
        self.assertEqual(
            parse_date_string("March-15-2024"),
            datetime(2024, 3, 15),
        )

    def test_normalized_beam_anchor_parses(self):
        self.assertEqual(
            parse_date_string("March 15, 2024"),
            datetime(2024, 3, 15),
        )


if __name__ == "__main__":
    unittest.main()

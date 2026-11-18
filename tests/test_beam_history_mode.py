from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
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

from src.core.saving.history_intent import (
    PrototypeWarmupError,
    lexical_history_intent,
    prefers_history,
    prototype_vectors,
    reset_prototype_cache,
    resolve_include_history,
)

try:
    from src.services.api.controllers import retrieve as retrieve_mod
except ModuleNotFoundError:
    retrieve_mod = None

try:
    from src.utils.dates import parse_date_string
except ModuleNotFoundError:
    parse_date_string = None

_FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "history_intent_minilm.json").read_text(
        encoding="utf-8"
    )
)


def _wants(text: str, **kwargs) -> bool:
    if retrieve_mod is not None:
        return retrieve_mod._wants_historical_facts(text, **kwargs)
    return resolve_include_history(
        kwargs.get("explicit"),
        query_text=text,
        query=kwargs.get("query_vector"),
        history_vectors=kwargs.get("history_vectors"),
        current_vectors=kwargs.get("current_vectors"),
    )


class HistoryModeTests(unittest.TestCase):
    def test_ordering_query_wants_history(self):
        self.assertTrue(
            _wants("Can you list the order in which I brought up aspects?")
        )

    def test_first_sprint_wants_history(self):
        self.assertTrue(_wants("When does my first sprint end?"))

    def test_have_i_contradiction_probe_wants_history(self):
        self.assertTrue(
            _wants("Have I integrated Flask-Login for session management?")
        )

    def test_italian_previous_version_wants_history(self):
        self.assertTrue(_wants("com'era prima il codice"))

    def test_knowledge_update_stays_current_only(self):
        self.assertFalse(
            _wants("What is the average response time of the dashboard API?")
        )
        self.assertFalse(
            _wants("How many commits have been merged into the main branch?")
        )
        self.assertFalse(
            _wants("What is the deadline for completing my first sprint?")
        )
        self.assertFalse(
            _wants("What is the current session management setting?")
        )
        self.assertTrue(
            _wants(
                "How many days do I have between finishing the memory store "
                "and the deadline for completing GPT-4 fine-tuning?"
            )
        )

    def test_current_value_veto_beats_history_embedding(self):
        self.assertFalse(
            resolve_include_history(
                None,
                query_text="what is the current value",
                query=[1.0, 0.0],
                history_vectors=[[1.0, 0.0]],
                current_vectors=[[0.0, 1.0]],
            )
        )

    def test_explicit_flag_overrides_similarity_and_lexical(self):
        history = [[1.0, 0.0]]
        current = [[0.0, 1.0]]
        self.assertTrue(
            _wants(
                "what is the current value",
                explicit=True,
                query_vector=[0.0, 1.0],
                history_vectors=history,
                current_vectors=current,
            )
        )
        self.assertFalse(
            _wants(
                "Have I integrated Flask-Login for session management?",
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
        if retrieve_mod is None:
            self.skipTest("retrieve controller dependencies are not installed")
        current = SimpleNamespace(properties={}, deprecated=False)
        old = SimpleNamespace(properties={"invalid_at": "2024-03-31"}, deprecated=False)
        self.assertFalse(
            retrieve_mod._fact_predicates_allowed(old, current, include_history=False)
        )
        self.assertTrue(
            retrieve_mod._fact_predicates_allowed(old, current, include_history=True)
        )


class RecordedEmbeddingProbeTests(unittest.TestCase):
    def test_real_wording_probes_match_held_out_labels(self):
        history = _FIXTURE["history_vectors"]
        current = _FIXTURE["current_vectors"]
        misses = []
        for probe in _FIXTURE["probes"]:
            got = resolve_include_history(
                None,
                query_text=probe["text"],
                query=probe["vector"],
                history_vectors=history,
                current_vectors=current,
            )
            if got is not probe["want_history"]:
                misses.append((probe["category"], probe["text"], got, probe["want_history"]))
        self.assertEqual(misses, [])

    def test_between_days_probe_needs_lexical_when_embedding_is_current(self):
        probe = next(
            item
            for item in _FIXTURE["probes"]
            if item["text"].startswith("How many days are there between")
        )
        self.assertTrue(lexical_history_intent(probe["text"]))
        self.assertFalse(
            prefers_history(
                probe["vector"],
                _FIXTURE["history_vectors"],
                _FIXTURE["current_vectors"],
            )
        )
        self.assertTrue(
            resolve_include_history(
                None,
                query_text=probe["text"],
                query=probe["vector"],
                history_vectors=_FIXTURE["history_vectors"],
                current_vectors=_FIXTURE["current_vectors"],
            )
        )


class PrototypeCacheTests(unittest.TestCase):
    def tearDown(self):
        reset_prototype_cache()

    def test_failed_warmup_does_not_retry_exemplars_during_backoff(self):
        reset_prototype_cache()
        calls: list[str] = []

        def boom(text: str) -> list[float]:
            calls.append(text)
            raise RuntimeError("embed down")

        with self.assertRaises(RuntimeError):
            prototype_vectors(boom)
        first = len(calls)
        self.assertGreater(first, 0)
        with self.assertRaises(PrototypeWarmupError):
            prototype_vectors(boom)
        self.assertEqual(len(calls), first)

    def test_successful_warmup_is_cached(self):
        reset_prototype_cache()
        calls: list[str] = []

        def embed(text: str) -> list[float]:
            calls.append(text)
            return [1.0, 0.0] if "previous" in text or "prima" in text or "antes" in text else [0.0, 1.0]

        first = prototype_vectors(embed)
        n = len(calls)
        second = prototype_vectors(embed)
        self.assertEqual(len(calls), n)
        self.assertIs(first, second)


class SessionIdParseTests(unittest.TestCase):
    def test_session_and_batch_turn_ids(self):
        if retrieve_mod is None:
            self.skipTest("retrieve controller dependencies are not installed")
        text = "Session id: session_12. Unit id: b3_t2. Also session_4."
        ids = retrieve_mod._session_ids_from_text(text)
        self.assertIn("session_12", ids)
        self.assertIn("session_4", ids)
        self.assertIn("session_b3_t2", ids)


class BeamDateStampTests(unittest.TestCase):
    def test_hyphenated_beam_anchor_parses(self):
        if parse_date_string is None:
            self.skipTest("date utilities are not installed")
        self.assertEqual(
            parse_date_string("March-15-2024"),
            datetime(2024, 3, 15),
        )

    def test_normalized_beam_anchor_parses(self):
        if parse_date_string is None:
            self.skipTest("date utilities are not installed")
        self.assertEqual(
            parse_date_string("March 15, 2024"),
            datetime(2024, 3, 15),
        )


if __name__ == "__main__":
    unittest.main()

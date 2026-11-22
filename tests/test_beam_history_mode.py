from __future__ import annotations

import json
import os
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
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
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "postgres",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.saving import history_intent as history_mod
from src.core.saving.history_intent import (
    CURRENT_EXAMPLES,
    HISTORY_EXAMPLES,
    HISTORY_MARGIN,
    PrototypeWarmupError,
    lexical_history_decision,
    lexical_history_intent,
    prefers_history,
    peek_prototype_vectors,
    prototype_vectors,
    reset_prototype_cache,
    resolve_include_history,
)


def _mainline_resolve_include_history(
    explicit,
    query_text=None,
    query=None,
    history_vectors=None,
    current_vectors=None,
    *,
    margin=0.02,
):
    if explicit is not None:
        return explicit
    if lexical_history_intent(query_text):
        return True
    cleaned = (query_text or "").strip()
    if (
        cleaned
        and history_mod._CURRENT_VALUE_RE.search(cleaned)
        and not history_mod._STRONG_HISTORY_RE.search(cleaned)
    ):
        return False
    return prefers_history(
        query,
        history_vectors or [],
        current_vectors or [],
        margin=margin,
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

    def test_have_i_before_wants_history(self):
        self.assertTrue(_wants("Have I implemented this library before?"))
        self.assertTrue(_wants("Have I ever used this library?"))
        self.assertTrue(_wants("Did I use this library last year?"))

    def test_did_i_already_wants_history(self):
        self.assertTrue(_wants("Did I already complete this?"))

    def test_ttl_current_lookup_stays_current(self):
        self.assertFalse(
            _wants("What is the TTL for the cache that was changed?")
        )
        self.assertTrue(_wants("What was the TTL before the change?"))

    def test_difference_between_before_change_wants_history(self):
        self.assertTrue(
            _wants(
                "What is the difference between the timeout before the change "
                "and the one we use today?"
            )
        )
        self.assertFalse(
            _wants("what is the difference between redis and memcached")
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
        self.assertFalse(_wants("When does my first sprint end?"))
        self.assertFalse(
            _wants("Have I integrated Flask-Login for session management?")
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
                "Have I implemented this library before?",
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
                [0.0, 1.0],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            )
        )
        self.assertTrue(
            resolve_include_history(
                None,
                query_text=probe["text"],
                query=[0.0, 1.0],
                history_vectors=[[1.0, 0.0]],
                current_vectors=[[0.0, 1.0]],
            )
        )


CURRENT_VALUE_NEGATIVES = (
    "what did I set the port to",
    "what is the difference between redis and memcached",
    "What is the TTL for the cache that was changed?",
    "what's the current timeout",
    "how many workers are running",
    "what port is the server on",
    "what is my email address",
    "how many days until the deadline",
    "did I set the debug flag",
    "have I enabled caching",
    "what's the API base URL",
    "what is the configured hostname",
    "how many items are in the cart",
    "what is the status of the job",
    "tell me the latest count",
    "what value is in the config now",
    "what is currently used to cache responses",
    "what library is currently used to do response caching",
    "what is the current timeout that was changed yesterday",
    "what is the current timeout which was updated yesterday",
    "What is the TTL for the cache which was changed?",
    "did I use postgres or mysql",
    "have I turned on TLS",
    "what timezone is set",
    "how many retries are configured",
    "what's the database name",
    "qual è la porta configurata",
    "che differenza c'è tra X e Y",
    "was ist der aktuelle Port",
    "wie lautet der Timeout",
    "cuál es el puerto actual",
    "cuál es la diferencia entre X e Y",
    "quel est le port actuel",
    "quelle est la différence entre X et Y",
    "qual é a porta atual",
    "ポートは何に設定されていますか",
    "今のタイムアウトは何ですか",
    "当前端口是多少",
)

HISTORY_POSITIVES = (
    "what did this say before the change",
    "show the previous version of this",
    "what was deleted from this",
    "have I implemented this library before",
    "Have I ever used this library?",
    "Did I use this library last year?",
    "What is the difference between the timeout before the change and the one we use today?",
    "What was the TTL before the change?",
    "what was previously configured for the timeout",
    "what was originally the cache TTL",
    "what was the original value of the cedar signal",
    "what was first said about the timeout",
    "what did the cedar code used to be",
    "what was the timeout before it changed",
    "which settings were removed from the policy",
    "show me the settings that were removed",
    "show me the TTL settings which were updated",
    "which TTL settings were updated",
    "which fields were changed in the last revision",
    "which rules were updated last quarter",
    "did I already complete this",
    "did I say I finished this or that I never started it",
    "how many days are there between these two deadlines",
    "can you list the order in which I brought up these topics",
    "walk me through the chronological sequence of events",
    "these two statements contradict each other",
    "com'era prima questo testo",
    "qual era la versione precedente",
    "ho già implementato questo servizio",
    "habe ich das schon integriert",
    "ai-je déjà implémenté cela",
    "he integrado esto antes",
)


class EverydayHistoryIntentTests(unittest.TestCase):
    def test_current_value_questions_do_not_trigger_history(self):
        self.assertGreaterEqual(len(CURRENT_VALUE_NEGATIVES), 25)
        misses = [text for text in CURRENT_VALUE_NEGATIVES if _wants(text)]
        self.assertEqual(misses, [])

    def test_history_questions_still_trigger(self):
        misses = [text for text in HISTORY_POSITIVES if not _wants(text)]
        self.assertEqual(misses, [])

    def test_past_version_cues_trigger_history(self):
        for text in (
            "what was previously the current timeout",
            "what was originally the current TTL",
            "what was first said about the current setting",
            "what did the timeout used to be",
            "what was the timeout before it changed",
            "which settings were removed",
            "which fields were changed",
            "which rules were updated",
            "show me the settings that were removed",
            "show me the TTL settings which were updated",
            "which TTL settings were updated",
        ):
            with self.subTest(text=text):
                self.assertTrue(_wants(text), text)

    def test_current_value_questions_still_skip_history(self):
        for text in (
            "what is the current timeout",
            "what is the current TTL",
            "what is the current value",
            "What is the TTL for the cache that was changed?",
            "what is currently used to cache responses",
            "what library is currently used to do response caching",
            "what is the current timeout that was changed yesterday",
            "what is the current timeout which was updated yesterday",
            "What is the TTL for the cache which was changed?",
        ):
            with self.subTest(text=text):
                self.assertFalse(_wants(text), text)

    def test_include_history_flag_still_wins(self):
        self.assertTrue(_wants("what did I set the port to", explicit=True))
        self.assertFalse(
            _wants("what did this say before the change", explicit=False)
        )


class MainlineHistoryParityTests(unittest.TestCase):
    def test_margin_and_strict_greater_than_threshold_match_main(self):
        self.assertEqual(HISTORY_MARGIN, 0.02)
        history = [[1.0, 0.0]]
        current = [[0.0, 1.0]]
        self.assertTrue(prefers_history([1.0, 0.0], history, current, margin=0.02))
        self.assertFalse(prefers_history([0.0, 1.0], history, current, margin=0.02))
        self.assertFalse(
            prefers_history([1.0, 0.0], [[1.0, 0.0]], [[1.0, 0.0]], margin=0.02)
        )

    def test_explicit_none_true_false_match_mainline_resolver(self):
        history = [[1.0, 0.0]]
        current = [[0.0, 1.0]]
        queries = (
            list(HISTORY_EXAMPLES)
            + list(CURRENT_EXAMPLES)
            + [
                "who is Alice",
                "what is the current value",
                "what did this say before the change",
                "Have I implemented this library before?",
                "com'era prima questo testo",
                "was steht hier jetzt",
                "以前の内容は何でしたか",
                "现在的内容是什么",
                "ما المكتوب الآن",
                "что здесь было написано раньше",
            ]
        )
        vectors = (None, [1.0, 0.0], [0.0, 1.0])
        flags = (True, False, None)
        mismatches = []
        for text in queries:
            for explicit in flags:
                for query in vectors:
                    got = resolve_include_history(
                        explicit,
                        query_text=text,
                        query=query,
                        history_vectors=history,
                        current_vectors=current,
                    )
                    expected = _mainline_resolve_include_history(
                        explicit,
                        query_text=text,
                        query=query,
                        history_vectors=history,
                        current_vectors=current,
                    )
                    if got is not expected:
                        mismatches.append((text, explicit, query, got, expected))
        self.assertEqual(mismatches, [])

    def test_fixture_probes_match_mainline_and_held_out_labels(self):
        history = _FIXTURE["history_vectors"]
        current = _FIXTURE["current_vectors"]
        mismatches = []
        for probe in _FIXTURE["probes"]:
            kwargs = dict(
                explicit=None,
                query_text=probe["text"],
                query=probe["vector"],
                history_vectors=history,
                current_vectors=current,
            )
            got = resolve_include_history(**kwargs)
            mainline = _mainline_resolve_include_history(**kwargs)
            if got is not mainline or got is not probe["want_history"]:
                mismatches.append(
                    {
                        "text": probe["text"],
                        "got": got,
                        "mainline": mainline,
                        "want": probe["want_history"],
                    }
                )
        self.assertEqual(mismatches, [])


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

    def test_warmup_backoff_starts_from_failure_time(self):
        reset_prototype_cache()
        clock = {"now": 1000.0}
        calls: list[str] = []

        def monotonic() -> float:
            return clock["now"]

        def boom(text: str) -> list[float]:
            calls.append(text)
            clock["now"] += 45.0
            raise RuntimeError("embed down")

        with patch("src.core.saving.history_intent.time.monotonic", monotonic):
            with self.assertRaises(RuntimeError):
                prototype_vectors(boom)
            self.assertEqual(len(calls), 1)
            with self.assertRaises(PrototypeWarmupError):
                prototype_vectors(boom)
            self.assertEqual(len(calls), 1)
            clock["now"] += 29.0
            with self.assertRaises(PrototypeWarmupError):
                prototype_vectors(boom)
            self.assertEqual(len(calls), 1)
            clock["now"] += 2.0
            with self.assertRaises(RuntimeError):
                prototype_vectors(boom)
            self.assertEqual(len(calls), 2)

    def test_successful_warmup_is_cached(self):
        reset_prototype_cache()
        calls: list[str] = []

        def embed(text: str) -> list[float]:
            calls.append(text)
            return [1.0, 0.0] if "previous" in text or "prima" in text or "antes" in text else [0.0, 1.0]

        self.assertIsNone(peek_prototype_vectors())
        first = prototype_vectors(embed)
        n = len(calls)
        self.assertIs(peek_prototype_vectors(), first)
        second = prototype_vectors(embed)
        self.assertEqual(len(calls), n)
        self.assertIs(first, second)

    def test_lexical_decision_skips_embeddings(self):
        self.assertTrue(lexical_history_decision("what did this say before the change"))
        self.assertFalse(lexical_history_decision("what is the current value"))
        self.assertIsNone(lexical_history_decision("who is Alice"))

    def test_warmup_timeout_returns_without_waiting_for_worker(self):
        reset_prototype_cache()
        started = time.monotonic()

        def slow_many(texts: list[str]) -> list[list[float]]:
            time.sleep(1.2)
            return [[1.0, 0.0] for _ in texts]

        with patch.object(history_mod, "_PROTOTYPE_WARMUP_TIMEOUT", 0.15):
            with self.assertRaises(PrototypeWarmupError):
                prototype_vectors(lambda _text: [1.0, 0.0], slow_many)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.6)
        with self.assertRaises(PrototypeWarmupError):
            prototype_vectors(lambda _text: [1.0, 0.0], slow_many)
        self.assertIsNone(peek_prototype_vectors())

    def test_embed_many_warms_prototypes_in_one_call(self):
        reset_prototype_cache()
        single_calls: list[str] = []
        batch_calls: list[int] = []

        def embed(text: str) -> list[float]:
            single_calls.append(text)
            return [1.0, 0.0]

        def embed_many(texts: list[str]) -> list[list[float]]:
            batch_calls.append(len(texts))
            return [[1.0, 0.0] for _ in texts]

        history, current = prototype_vectors(embed, embed_many)
        self.assertEqual(single_calls, [])
        self.assertEqual(batch_calls, [len(HISTORY_EXAMPLES) + len(CURRENT_EXAMPLES)])
        self.assertEqual(len(history), len(HISTORY_EXAMPLES))
        self.assertEqual(len(current), len(CURRENT_EXAMPLES))


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

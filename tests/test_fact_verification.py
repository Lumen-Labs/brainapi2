import os
import sys
import unittest

from fastapi import HTTPException

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
    "POSTGRES_PASSWORD": "test-password",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.kg import Node, Predicate
from src.core.search.traverse import flatten_neighborhood
from src.core.verification.facts import (
    REJECTED,
    UNVERIFIED,
    VERIFIED,
    extract_verification,
    fact_pair_allowed,
    fact_tuple_is_rejected,
    include_rejected_enabled,
    parse_history,
    verification_scope,
    verification_write_properties,
    verified_rank_boost,
)
from src.core.verification.service import apply_verification_action
from src.services.api.constants.requests import GetContextRequestBody
from src.services.api.controllers import retrieve as retrieve_mod


def _node(uuid, name, *, labels=None, properties=None, happened_at=None):
    return Node(
        uuid=uuid,
        name=name,
        labels=labels or ["PERSON"],
        description=f"desc-{name}",
        properties=properties or {},
        happened_at=happened_at,
    )


def _pred(uuid, name, *, properties=None):
    return Predicate(
        uuid=uuid,
        name=name,
        description=f"desc-{name}",
        properties=properties or {},
    )


class VerificationHelpersTests(unittest.TestCase):
    def test_missing_state_is_unverified(self):
        pred = _pred("r1", "KNOWS")
        self.assertEqual(extract_verification(pred).state, UNVERIFIED)
        self.assertFalse(fact_tuple_is_rejected(pred))

    def test_rejected_is_hidden_unless_opted_in(self):
        pred = _pred("r1", "KNOWS", properties={"verification_state": REJECTED})
        self.assertTrue(fact_tuple_is_rejected(pred))
        self.assertFalse(fact_pair_allowed(pred, None))
        with verification_scope(include_rejected=True):
            self.assertTrue(fact_pair_allowed(pred, None))

    def test_history_round_trips_json_strings(self):
        raw = [
            '{"id":"1","from_state":"unverified","to_state":"verified",'
            '"verified_by":"agent-a","verified_at":"2026-11-16T00:00:00+00:00",'
            '"verification_source":"doc-1","note":"ok"}'
        ]
        history = parse_history(raw)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].to_state, VERIFIED)
        self.assertEqual(history[0].verified_by, "agent-a")

    def test_rank_boost_default_is_positive(self):
        self.assertGreater(verified_rank_boost(), 0)


class RetrieveVerificationTests(unittest.TestCase):
    def test_predicates_allowed_hides_rejected(self):
        live = _pred("r1", "MADE")
        rejected = _pred("r2", "TARGETED", properties={"verification_state": REJECTED})
        self.assertTrue(
            retrieve_mod._fact_predicates_allowed(live, live, include_history=False)
        )
        self.assertFalse(
            retrieve_mod._fact_predicates_allowed(
                live, rejected, include_history=False
            )
        )
        with verification_scope(include_rejected=True):
            self.assertTrue(
                retrieve_mod._fact_predicates_allowed(
                    live, rejected, include_history=False
                )
            )

    def test_verified_facts_rank_higher(self):
        unverified = {
            "triple": (
                _node("a", "Alice"),
                _pred("r1", "MADE"),
                _node("e", "Event", labels=["EVENT"]),
                _pred("r2", "TARGETED"),
                _node("b", "Bob"),
            ),
            "score": 0.4,
            "ppr_mass": 0.0,
        }
        verified = {
            "triple": (
                _node("a", "Alice"),
                _pred(
                    "r3",
                    "MADE",
                    properties={"verification_state": VERIFIED, "verified_by": "expert"},
                ),
                _node("e2", "Event2", labels=["EVENT"]),
                _pred("r4", "TARGETED"),
                _node("c", "Carol"),
            ),
            "score": 0.4,
            "ppr_mass": 0.0,
        }
        self.assertGreater(
            retrieve_mod._relevance_score(verified),
            retrieve_mod._relevance_score(unverified),
        )
        ranked = retrieve_mod._rank_facts_with_completeness([unverified, verified])
        self.assertEqual(ranked[0]["triple"][1].uuid, "r3")

    def test_build_fact_channel_exposes_verification(self):
        verified_pred = _pred(
            "r1",
            "MADE",
            properties={
                "verification_state": VERIFIED,
                "verified_by": "analyst-1",
                "verified_at": "2026-11-16T00:00:00+00:00",
                "verification_source": "src-9",
            },
        )
        candidate = {
            "identified_entity": "Alice",
            "triple": (
                _node("a", "Alice"),
                verified_pred,
                _node("e", "Event", labels=["EVENT"]),
                _pred("r2", "TARGETED"),
                _node("b", "Bob"),
            ),
            "text": "Alice MADE Event TARGETED Bob",
            "chunk_ids": [],
            "session_ids": [],
        }
        _lines, triples, _sessions = retrieve_mod._build_fact_channel([candidate], {})
        self.assertEqual(len(triples), 1)
        self.assertIsNotNone(triples[0].verification)
        self.assertEqual(triples[0].verification.state, VERIFIED)
        self.assertEqual(triples[0].verification.verified_by, "analyst-1")
        self.assertEqual(triples[0].verification.verification_source, "src-9")

    def test_flatten_neighborhood_skips_rejected_hops(self):
        neighborhood = [
            {
                "predicate": _pred(
                    "r-bad", "KNOWS", properties={"verification_state": REJECTED}
                ),
                "node": _node("b", "Bob"),
                "neighbors": [],
            },
            {
                "predicate": _pred("r-ok", "KNOWS"),
                "node": _node("c", "Carol"),
                "neighbors": [],
            },
        ]
        hops, truncated = flatten_neighborhood(
            neighborhood,
            rel_types=None,
            node_labels=None,
            direction="both",
            limit=10,
        )
        self.assertFalse(truncated)
        self.assertEqual(len(hops), 1)
        self.assertEqual(hops[0]["node"]["uuid"], "c")


class ContextRequestTests(unittest.TestCase):
    def test_include_rejected_defaults_false(self):
        body = GetContextRequestBody(text="hello")
        self.assertFalse(body.include_rejected)


class FakeGraph:
    def __init__(self, row):
        self.row = row
        self.writes = []

    def get_fact_record(self, fact_id, brain_id):
        if self.row and self.row.get("uuid") == fact_id:
            return self.row
        return None

    def set_fact_verification(self, fact_id, brain_id, *, kind, properties):
        self.writes.append((fact_id, brain_id, kind, dict(properties)))
        props = dict(self.row.get("properties") or {})
        props.update(properties)
        self.row["properties"] = props
        return self.row


class VerificationServiceTests(unittest.TestCase):
    def _row(self, fact_id="rel-1"):
        return {
            "kind": "relationship",
            "uuid": fact_id,
            "name": "WORKS_AT",
            "description": "works at",
            "properties": {"uuid": fact_id, "verification_state": UNVERIFIED},
            "subject_uuid": "alice",
            "subject_name": "Alice",
            "subject_labels": ["PERSON"],
            "object_uuid": "acme",
            "object_name": "Acme",
            "object_labels": ["ORG"],
            "labels": [],
        }

    def test_verify_requires_actor(self):
        with self.assertRaises(HTTPException) as raised:
            apply_verification_action(
                FakeGraph(self._row()),
                fact_id="rel-1",
                brain_id="brain",
                action="verify",
            )
        self.assertEqual(raised.exception.status_code, 400)

    def test_verify_appends_history_and_reset_keeps_it(self):
        graph = FakeGraph(self._row())
        verified = apply_verification_action(
            graph,
            fact_id="rel-1",
            brain_id="brain",
            action="verify",
            verified_by="analyst-1",
            verification_source="doc-9",
            note="checked",
        )
        self.assertEqual(verified.verification.state, VERIFIED)
        self.assertEqual(verified.verification.verified_by, "analyst-1")
        self.assertEqual(verified.verification.verification_source, "doc-9")
        self.assertEqual(len(verified.history), 1)
        self.assertEqual(verified.history[0].from_state, UNVERIFIED)
        self.assertEqual(verified.history[0].to_state, VERIFIED)

        reset = apply_verification_action(
            graph,
            fact_id="rel-1",
            brain_id="brain",
            action="reset",
            verified_by="analyst-1",
            note="reopen",
        )
        self.assertEqual(reset.verification.state, UNVERIFIED)
        self.assertIsNone(reset.verification.verified_by)
        self.assertEqual(len(reset.history), 2)
        self.assertEqual(reset.history[1].to_state, UNVERIFIED)
        write_keys = set(graph.writes[-1][3])
        self.assertTrue(
            {"verification_state", "verified_by", "verified_at", "verification_history"}
            <= write_keys
        )
        self.assertNotIn("valid_from", write_keys)
        self.assertNotIn("valid_to", write_keys)
        self.assertNotIn("superseded_by", write_keys)

    def test_scope_does_not_leak(self):
        self.assertFalse(include_rejected_enabled())
        with verification_scope(include_rejected=True):
            self.assertTrue(include_rejected_enabled())
        self.assertFalse(include_rejected_enabled())

    def test_write_properties_stay_on_verification_fields(self):
        props = verification_write_properties(
            state=VERIFIED,
            verified_by="a",
            verified_at="2026-11-16T00:00:00+00:00",
            verification_source="src",
            note="n",
            history=[],
        )
        self.assertEqual(
            set(props),
            {
                "verification_state",
                "verified_by",
                "verified_at",
                "verification_source",
                "verification_note",
                "verification_history",
            },
        )

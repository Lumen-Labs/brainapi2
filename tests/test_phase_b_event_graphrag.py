import os
import sys
import time
import unittest

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

from src.constants.kg import Node, Predicate
from src.core.search.fact_filter import personalized_pagerank
from src.services.api.controllers import retrieve as retrieve_mod


def _node(uuid: str, name: str, *, labels=None, happened_at=None):
    return Node(
        uuid=uuid,
        name=name,
        labels=labels or ["PERSON"],
        description=f"desc-{name}",
        properties={},
        happened_at=happened_at,
    )


def _pred(uuid: str, name: str, *, flow_key: str = "", invalid=False, deprecated=False):
    props = {"invalid_at": "01/01/2020"} if invalid else {}
    return Predicate(
        uuid=uuid,
        name=name,
        description=f"desc-{name}",
        flow_key=flow_key or None,
        properties=props,
        deprecated=deprecated,
    )


def _candidate(
    *,
    score: float,
    r_uuid: str,
    r2_uuid: str,
    flow_key: str = "flow",
    n_uuid: str = "n",
    m_uuid: str = "m",
    b_uuid: str = "b",
    r_name: str = "MADE",
    r2_name: str = "TARGETED",
    n_labels=None,
    m_labels=None,
    b_labels=None,
    session_ids=None,
    happened_at=None,
    entity: str = "q",
):
    n = _node(n_uuid, n_uuid, labels=n_labels or ["PERSON"])
    r = _pred(r_uuid, r_name, flow_key=flow_key)
    m = _node(
        m_uuid,
        m_uuid,
        labels=m_labels or ["EVENT"],
        happened_at=happened_at,
    )
    r2 = _pred(r2_uuid, r2_name, flow_key=flow_key)
    b = _node(b_uuid, b_uuid, labels=b_labels or ["PERSON"])
    return {
        "identified_entity": entity,
        "triple": (n, r, m, r2, b),
        "score": score,
        "key": (r_uuid, r2_uuid),
        "text": retrieve_mod._format_event_fact(n, r, m, r2, b),
        "chunk_ids": [],
        "session_ids": list(session_ids or []),
    }


class DiversificationTests(unittest.TestCase):
    def test_coverage_beats_pure_truncation_under_max_facts(self):
        ranked = []
        for i in range(8):
            ranked.append(
                _candidate(
                    score=0.01 * i,
                    r_uuid=f"r-hub0-{i}",
                    r2_uuid=f"r2-hub0-{i}",
                    flow_key="hub-0",
                    m_uuid="event-0",
                    session_ids=["session_1"],
                )
            )
        for i in range(4):
            ranked.append(
                _candidate(
                    score=0.5 + 0.01 * i,
                    r_uuid=f"r-hub{i+1}",
                    r2_uuid=f"r2-hub{i+1}",
                    flow_key=f"hub-{i+1}",
                    m_uuid=f"event-{i+1}",
                    session_ids=[f"session_{i+2}"],
                )
            )
        control = ranked[:5]
        diversified = retrieve_mod._diversify_facts(ranked, max_facts=5)
        control_hubs = {retrieve_mod._event_hub_id(c) for c in control}
        control_sessions = {s for c in control for s in c["session_ids"]}
        div_hubs = {retrieve_mod._event_hub_id(c) for c in diversified}
        div_sessions = {s for c in diversified for s in c["session_ids"]}
        self.assertGreaterEqual(len(div_hubs), len(control_hubs))
        self.assertGreater(len(div_sessions), len(control_sessions))
        self.assertEqual(len(diversified), 5)

    def test_diversify_is_deterministic(self):
        ranked = [
            _candidate(
                score=0.2,
                r_uuid=f"r{i}",
                r2_uuid=f"x{i}",
                flow_key=f"h{i % 3}",
                m_uuid=f"e{i % 3}",
                session_ids=[f"session_{(i % 4) + 1}"],
            )
            for i in range(12)
        ]
        a = retrieve_mod._diversify_facts(ranked, max_facts=6)
        b = retrieve_mod._diversify_facts(list(reversed(ranked)), max_facts=6)
        self.assertEqual([c["key"] for c in a], [c["key"] for c in b])

    def test_diversify_max_facts_zero(self):
        ranked = [_candidate(score=0.1, r_uuid="r", r2_uuid="x")]
        self.assertEqual(retrieve_mod._diversify_facts(ranked, max_facts=0), [])

    def test_diversify_latency_under_budget(self):
        ranked = [
            _candidate(
                score=0.001 * i,
                r_uuid=f"r{i}",
                r2_uuid=f"x{i}",
                flow_key=f"h{i % 40}",
                m_uuid=f"e{i % 40}",
                session_ids=[f"session_{(i % 20) + 1}"],
            )
            for i in range(200)
        ]
        retrieve_mod._diversify_facts(ranked, max_facts=50)
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            out = retrieve_mod._diversify_facts(ranked, max_facts=50)
            samples.append((time.perf_counter() - start) * 1000)
        self.assertEqual(len(out), 50)
        self.assertLess(min(samples), 100.0)


class HubCompletenessTests(unittest.TestCase):
    def test_spine_scores_above_context_only(self):
        spine = _candidate(
            score=0.5,
            r_uuid="r1",
            r2_uuid="r2",
            r_name="MADE",
            r2_name="TARGETED",
            m_uuid="e1",
        )
        context = _candidate(
            score=0.5,
            r_uuid="r3",
            r2_uuid="r4",
            r_name="RELATED",
            r2_name="OCCURRED_WITHIN",
            m_uuid="e2",
            b_uuid="ctx",
            b_labels=["CONTEXT"],
        )
        self.assertGreater(
            retrieve_mod._hub_completeness_score(spine),
            retrieve_mod._hub_completeness_score(context),
        )

    def test_completeness_prefers_full_triangle_when_scores_tied(self):
        weak = _candidate(
            score=0.4,
            r_uuid="rw",
            r2_uuid="rw2",
            r_name="MENTIONED",
            r2_name="NEAR",
            m_uuid="ew",
            m_labels=["THING"],
        )
        strong = _candidate(
            score=0.4,
            r_uuid="rs",
            r2_uuid="rs2",
            r_name="MADE",
            r2_name="TARGETED",
            m_uuid="es",
        )
        ranked = retrieve_mod._rank_facts_with_completeness([weak, strong])
        self.assertEqual(ranked[0]["key"], ("rs", "rs2"))

    def test_complementary_legs_kept_under_budget(self):
        spine = _candidate(
            score=0.1,
            r_uuid="r-spine",
            r2_uuid="r2-spine",
            flow_key="same-hub",
            m_uuid="event-same",
            r_name="MADE",
            r2_name="TARGETED",
            session_ids=["session_1"],
        )
        context = _candidate(
            score=0.11,
            r_uuid="r-ctx",
            r2_uuid="r2-ctx",
            flow_key="same-hub",
            m_uuid="event-same",
            r_name="MADE",
            r2_name="OCCURRED_WITHIN",
            b_uuid="place",
            b_labels=["LOCATION"],
            session_ids=["session_1"],
        )
        other = _candidate(
            score=0.12,
            r_uuid="r-other",
            r2_uuid="r2-other",
            flow_key="other-hub",
            m_uuid="event-other",
            session_ids=["session_2"],
        )
        noise = [
            _candidate(
                score=0.8,
                r_uuid=f"rn{i}",
                r2_uuid=f"xn{i}",
                flow_key="noise-hub",
                m_uuid="event-noise",
                session_ids=["session_1"],
            )
            for i in range(5)
        ]
        curated = retrieve_mod._diversify_facts(
            [spine, context, other] + noise, max_facts=3
        )
        keys = {c["key"] for c in curated}
        self.assertIn(("r-spine", "r2-spine"), keys)
        self.assertIn(("r-ctx", "r2-ctx"), keys)


class TypedPprTests(unittest.TestCase):
    def test_weighted_ppr_accepts_edge_weights(self):
        adjacency = {
            "seed": [("spine", 1.0), ("ctx", 0.2)],
            "spine": [("seed", 1.0)],
            "ctx": [("seed", 0.2)],
            "far": [("x", 1.0)],
            "x": [("far", 1.0)],
        }
        scores = personalized_pagerank(adjacency, {"seed": 1.0}, iterations=40)
        self.assertGreater(scores["spine"], scores["ctx"])
        self.assertGreater(scores["seed"], scores["far"])

    def test_adjacency_skips_invalid_legs(self):
        valid_n = _node("n1", "Alice")
        valid_r = _pred("r1", "MADE", flow_key="f1")
        event = _node("e1", "Party", labels=["EVENT"])
        valid_r2 = _pred("r2", "TARGETED", flow_key="f1")
        obj = _node("b1", "Bob")
        bad_r = _pred("rb", "MADE", flow_key="f2", invalid=True)
        bad_r2 = _pred("rb2", "TARGETED", flow_key="f2")
        neighbors = [
            (valid_n, valid_r, event, valid_r2, obj),
            (valid_n, bad_r, event, bad_r2, obj),
        ]

        class _Graph:
            def get_event_centric_neighbors(self, seed_uuids, brain_id="default"):
                return neighbors

        original = retrieve_mod.graph_adapter
        retrieve_mod.graph_adapter = _Graph()
        try:
            adj = retrieve_mod._build_adjacency_from_seeds(["n1"], "brain")
        finally:
            retrieve_mod.graph_adapter = original
        self.assertIn("n1", adj)
        self.assertIn("e1", adj)
        edge_targets = {dst for dst, _w in adj["n1"]}
        self.assertIn("e1", edge_targets)

    def test_typed_adjacency_weights_spine_over_context(self):
        actor = _node("n1", "Alice")
        made = _pred("r1", "MADE", flow_key="f1")
        event = _node("e1", "Party", labels=["EVENT"])
        targeted = _pred("r2", "TARGETED", flow_key="f1")
        obj = _node("b1", "Bob")
        occurred = _pred("r3", "OCCURRED_WITHIN", flow_key="f1")
        place = _node("p1", "Park", labels=["LOCATION"])
        neighbors = [
            (actor, made, event, targeted, obj),
            (actor, made, event, occurred, place),
        ]

        class _Graph:
            def get_event_centric_neighbors(self, seed_uuids, brain_id="default"):
                return neighbors

        original = retrieve_mod.graph_adapter
        retrieve_mod.graph_adapter = _Graph()
        try:
            adj = retrieve_mod._build_adjacency_from_seeds(["n1", "e1"], "brain")
        finally:
            retrieve_mod.graph_adapter = original
        e_out = {dst: w for dst, w in adj["e1"]}
        self.assertGreater(e_out.get("b1", 0), e_out.get("p1", 0))

    def test_typed_ppr_reorder_deterministic(self):
        a = _candidate(score=0.3, r_uuid="ra", r2_uuid="xa", m_uuid="ea", n_uuid="na")
        b = _candidate(score=0.3, r_uuid="rb", r2_uuid="xb", m_uuid="eb", n_uuid="nb")
        ppr = {"na": 0.2, "ea": 0.5, "nb": 0.2, "eb": 0.5}
        forward = sorted([a, b], key=lambda c: retrieve_mod._ppr_rank_key(c, ppr))
        reverse = sorted([b, a], key=lambda c: retrieve_mod._ppr_rank_key(c, ppr))
        self.assertEqual([c["key"] for c in forward], [c["key"] for c in reverse])


class PathFormatTests(unittest.TestCase):
    def test_format_labels_actor_event_target(self):
        n = _node("a", "Alice")
        r = _pred("r1", "MADE")
        m = _node("e", "Wedding", happened_at="2024-05-01", labels=["EVENT"])
        r2 = _pred("r2", "TARGETED")
        b = _node("b", "Bob")
        line = retrieve_mod._format_event_fact(n, r, m, r2, b)
        self.assertIn("Actor:", line)
        self.assertIn("Event:", line)
        self.assertIn("Target:", line)
        self.assertIn("@2024-05-01", line)
        self.assertNotIn("hub→hub", line.lower())

    def test_format_labels_context_for_occurred_within(self):
        n = _node("a", "Alice")
        r = _pred("r1", "MADE")
        m = _node("e", "Picnic", labels=["EVENT"])
        r2 = _pred("r2", "OCCURRED_WITHIN")
        b = _node("p", "Park", labels=["LOCATION"])
        line = retrieve_mod._format_event_fact(n, r, m, r2, b)
        self.assertIn("Context:", line)
        self.assertNotIn("Target:", line)


class TemporalConflictMetaTests(unittest.TestCase):
    def test_recency_prefer_and_emit_conflicts(self):
        older = _candidate(
            score=0.2,
            r_uuid="ro",
            r2_uuid="xo",
            flow_key="h-old",
            m_uuid="e-old",
            n_uuid="alice",
            happened_at="01/01/2020",
        )
        newer = _candidate(
            score=0.2,
            r_uuid="rn",
            r2_uuid="xn",
            flow_key="h-new",
            m_uuid="e-new",
            n_uuid="alice",
            happened_at="01/01/2024",
        )
        ranked = retrieve_mod._rank_facts_with_completeness([older, newer])
        self.assertEqual(ranked[0]["key"], ("rn", "xn"))
        conflicts = retrieve_mod._temporal_conflict_meta([older, newer])
        self.assertTrue(conflicts)
        self.assertEqual(conflicts[0]["entity_uuid"], "alice")


if __name__ == "__main__":
    unittest.main()

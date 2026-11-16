import unittest
from unittest.mock import MagicMock, patch

from src.constants.kg import Node, Predicate
from src.core.saving.conflicts import (
    annotation_payload,
    classify_disagreement,
    conflict_detect_on_ingest_enabled,
    detect_conflicts,
    dismiss_conflict,
    find_conflict_candidates,
    intervals_overlap,
    is_functional_predicate,
    judge_semantic_pairs,
    maybe_detect_after_ingest,
    open_conflicts_by_fact_ids,
    pair_key_for,
    parse_quantity,
    resolve_by_verified_fact,
    resolve_conflict,
    same_source_revision,
    snapshot_from_triple,
    values_equivalent,
)
from src.core.saving.conflict_report import build_report, render_markdown
from src.services.mcp.conflicts import (
    conflict_report_tool,
    dismiss_conflict_tool,
    read_conflict,
    read_conflicts,
    resolve_conflict_tool,
)
from src.services.mcp.prompt import MCP_TOOL_SPECS


def _node(uuid, name, labels=None, polarity="neutral"):
    return Node(
        uuid=uuid,
        name=name,
        labels=labels or ["ENTITY"],
        properties={"polarity": polarity},
        polarity=polarity,
    )


def _pred(uuid, name, value_props=None, description=""):
    props = dict(value_props or {})
    return Predicate(uuid=uuid, name=name, description=description, properties=props)


def _triple(subject, pred, obj):
    return (subject, pred, obj)


class MemoryConflictStore:
    def __init__(self, chunks=None):
        self.rows = {}
        self.chunks = chunks or {}

    def save_conflict(self, conflict, brain_id):
        self.rows[conflict.id] = conflict
        return conflict

    def update_conflict(self, conflict, brain_id):
        self.rows[conflict.id] = conflict
        return conflict

    def get_conflict_by_id(self, id, brain_id):
        return self.rows.get(id)

    def get_conflict_by_pair_key(self, pair_key, brain_id):
        for row in self.rows.values():
            if row.pair_key == pair_key:
                return row
        return None

    def get_conflicts_list(
        self,
        brain_id,
        limit=10,
        skip=0,
        status=None,
        source=None,
        subject=None,
        since=None,
        until=None,
    ):
        items = list(self.rows.values())
        if status:
            items = [item for item in items if item.status == status]
        if source:
            items = [item for item in items if source in item.source_keys()]
        if subject:
            needle = subject.lower()
            items = [
                item
                for item in items
                if needle in (item.subject.name or "").lower()
            ]
        total = len(items)
        return items[skip : skip + limit], total

    def get_open_conflicts_for_facts(self, fact_ids, brain_id):
        wanted = set(fact_ids)
        return [
            item
            for item in self.rows.values()
            if item.status == "open" and wanted.intersection(item.fact_ids())
        ]

    def get_text_chunks_by_ids(self, ids, with_observations, brain_id):
        return [self.chunks[item] for item in ids if item in self.chunks], []


class FakeGraph:
    def __init__(self, triples):
        self.triples = triples

    def list_validity_relationships(
        self,
        brain_id,
        skip=0,
        limit=100,
        since=None,
        until=None,
        subject_uuids=None,
    ):
        items = self.triples
        if subject_uuids:
            allowed = set(subject_uuids)
            items = [item for item in items if item[0].uuid in allowed]
        return items[skip : skip + limit], len(items)


class QuantityAndMatchTests(unittest.TestCase):
    def test_duration_normalization(self):
        self.assertEqual(parse_quantity("30 days")[0], 30)
        self.assertEqual(parse_quantity("two weeks")[0], 14)
        self.assertTrue(values_equivalent("30 days", "one month"))
        self.assertFalse(values_equivalent("30 days", "14 days"))

    def test_functional_vs_multivalued(self):
        self.assertTrue(is_functional_predicate("HAS_WINDOW", ["30 days", "14 days"]))
        self.assertFalse(is_functional_predicate("HAS_FEATURE", ["chat", "email"]))
        self.assertFalse(is_functional_predicate("MADE", ["event-a", "event-b"]))

    def test_same_source_revision_not_a_conflict_slot(self):
        left = {"source_key": "help/refunds", "fact_id": "a", "superseded_by": "b"}
        right = {"source_key": "help/refunds", "fact_id": "b"}
        self.assertTrue(same_source_revision(left, right))

    def test_overlapping_validity(self):
        live = {"valid_from": "2024-01-01T00:00:00Z"}
        closed = {
            "valid_from": "2023-01-01T00:00:00Z",
            "valid_to": "2024-01-01T00:00:00Z",
        }
        self.assertFalse(intervals_overlap(live, closed))
        concurrent = {"valid_from": "2024-06-01T00:00:00Z"}
        self.assertTrue(intervals_overlap(live, concurrent))

    def test_classify_numeric_and_polarity(self):
        kind, _ = classify_disagreement(
            {"value": "30 days", "polarity": "neutral"},
            {"value": "14 days", "polarity": "neutral"},
        )
        self.assertEqual(kind, "numeric_mismatch")
        kind, _ = classify_disagreement(
            {"value": "restocking fee", "polarity": "positive"},
            {"value": "restocking fee", "polarity": "negative"},
        )
        self.assertEqual(kind, "polarity_conflict")
        self.assertIsNone(
            classify_disagreement(
                {"value": "warehouse A", "polarity": "neutral"},
                {"value": "warehouse A", "polarity": "neutral"},
            )
        )


class DetectionEngineTests(unittest.TestCase):
    def _refund_triples(self):
        policy = _node("policy-1", "RefundPolicy", ["POLICY"])
        days30 = _node("win-30", "30 days", ["DURATION"])
        days14 = _node("win-14", "14 days", ["DURATION"])
        days60 = _node("win-60", "60 days", ["DURATION"])
        chat = _node("feat-chat", "chat", ["FEATURE"])
        email = _node("feat-email", "email", ["FEATURE"])
        fee = _node("fee", "restocking fee", ["FEE"])
        ship = _node("ship", "free over $50", ["OFFER"])
        t1 = "2024-01-01T00:00:00Z"
        t2 = "2025-01-01T00:00:00Z"
        return [
            _triple(
                policy,
                _pred(
                    "refund-v1",
                    "HAS_WINDOW",
                    {
                        "valid_from": t1,
                        "valid_to": t2,
                        "superseded_by": "refund-v2",
                        "source_key": "help/refunds",
                        "source_chunk_ids": ["chunk-refunds-v1"],
                    },
                ),
                days30,
            ),
            _triple(
                policy,
                _pred(
                    "refund-v2",
                    "HAS_WINDOW",
                    {
                        "valid_from": t2,
                        "source_key": "help/refunds",
                        "source_chunk_ids": ["chunk-refunds-v2"],
                    },
                ),
                days60,
            ),
            _triple(
                policy,
                _pred(
                    "refund-faq",
                    "HAS_WINDOW",
                    {
                        "valid_from": t1,
                        "source_key": "help/faq",
                        "source_chunk_ids": ["chunk-faq"],
                    },
                ),
                days14,
            ),
            _triple(
                policy,
                _pred(
                    "feat-chat",
                    "HAS_FEATURE",
                    {"valid_from": t1, "source_key": "help/features"},
                ),
                chat,
            ),
            _triple(
                policy,
                _pred(
                    "feat-email",
                    "HAS_FEATURE",
                    {"valid_from": t1, "source_key": "help/features"},
                ),
                email,
            ),
            _triple(
                policy,
                _pred(
                    "fee-yes",
                    "APPLIES",
                    {
                        "valid_from": t1,
                        "source_key": "help/fees",
                        "polarity": "positive",
                    },
                ),
                Node(
                    uuid="fee",
                    name="restocking fee",
                    labels=["FEE"],
                    properties={"polarity": "positive"},
                    polarity="positive",
                ),
            ),
            _triple(
                policy,
                _pred(
                    "fee-no",
                    "APPLIES",
                    {
                        "valid_from": t1,
                        "source_key": "help/returns",
                        "polarity": "negative",
                    },
                ),
                Node(
                    uuid="fee",
                    name="restocking fee",
                    labels=["FEE"],
                    properties={"polarity": "negative"},
                    polarity="negative",
                ),
            ),
            _triple(
                policy,
                _pred(
                    "ship-a",
                    "STATES",
                    {"valid_from": t1, "source_key": "help/shipping"},
                ),
                ship,
            ),
            _triple(
                policy,
                _pred(
                    "ship-b",
                    "STATES",
                    {"valid_from": t1, "source_key": "help/checkout"},
                ),
                ship,
            ),
        ]

    def test_detects_cross_source_and_skips_revisions_and_multivalue(self):
        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(self._refund_triples()), "brain")
        pairs = {tuple(sorted(item.fact_ids())) for item in found}
        self.assertIn(("refund-faq", "refund-v1"), pairs)
        self.assertIn(("refund-faq", "refund-v2"), pairs)
        self.assertIn(("fee-no", "fee-yes"), pairs)
        self.assertNotIn(("refund-v1", "refund-v2"), pairs)
        self.assertNotIn(("feat-chat", "feat-email"), pairs)
        self.assertNotIn(("ship-a", "ship-b"), pairs)
        types = {item.disagreement_type for item in found}
        self.assertIn("numeric_mismatch", types)
        self.assertIn("polarity_conflict", types)

    def test_candidate_partition_sends_non_numeric_to_semantic(self):
        snapshots = [
            snapshot_from_triple(item)
            for item in [
                _triple(
                    _node("p", "Policy"),
                    _pred(
                        "a",
                        "HAS_POLICY",
                        {"valid_from": "2024-01-01T00:00:00Z", "source_key": "s1"},
                    ),
                    _node("full", "full refund"),
                ),
                _triple(
                    _node("p", "Policy"),
                    _pred(
                        "b",
                        "HAS_POLICY",
                        {"valid_from": "2024-01-01T00:00:00Z", "source_key": "s2"},
                    ),
                    _node("credit", "store credit only"),
                ),
            ]
        ]
        definite, semantic = find_conflict_candidates(snapshots)
        self.assertEqual(definite, [])
        self.assertEqual(len(semantic), 1)

    def test_llm_judge_uses_injected_client_and_can_drop_false_positives(self):
        left = {"subject_name": "Policy", "predicate": "HAS_POLICY", "value": "a month"}
        right = {"subject_name": "Policy", "predicate": "HAS_POLICY", "value": "30 days"}
        client = MagicMock()
        client.generate_json.return_value = {
            "results": [
                {
                    "index": 0,
                    "contradicts": False,
                    "confidence": 0.1,
                    "type": "semantic_contradiction",
                }
            ]
        }
        result = judge_semantic_pairs([(left, right)], client=client)
        self.assertEqual(result[0][0], False)
        prompt = client.generate_json.call_args[0][0]
        self.assertIn("contradict", prompt.lower())

    def test_detect_conflicts_llm_path_mocked(self):
        triples = [
            _triple(
                _node("p", "Policy"),
                _pred(
                    "a",
                    "HAS_POLICY",
                    {"valid_from": "2024-01-01T00:00:00Z", "source_key": "s1"},
                ),
                _node("full", "full refund"),
            ),
            _triple(
                _node("p", "Policy"),
                _pred(
                    "b",
                    "HAS_POLICY",
                    {"valid_from": "2024-01-01T00:00:00Z", "source_key": "s2"},
                ),
                _node("credit", "store credit only"),
            ),
        ]
        client = MagicMock()
        client.generate_json.return_value = {
            "results": [
                {
                    "index": 0,
                    "contradicts": True,
                    "confidence": 0.91,
                    "type": "semantic_contradiction",
                }
            ]
        }
        found = detect_conflicts(
            MemoryConflictStore(),
            FakeGraph(triples),
            "brain",
            use_llm=True,
            llm_client=client,
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].detection_method, "llm")
        self.assertEqual(found[0].disagreement_type, "semantic_contradiction")

    def test_resolve_and_verified_side_hook(self):
        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(self._refund_triples()), "brain")
        numeric = next(
            item for item in found if item.disagreement_type == "numeric_mismatch"
        )
        resolved = resolve_conflict(
            store,
            "brain",
            numeric.id,
            winning_fact_id=numeric.sides[0].fact_id,
            method="verified_side",
        )
        self.assertEqual(resolved.status, "resolved")
        self.assertEqual(resolved.resolution.method, "verified_side")
        self.assertEqual(resolved.sides[0].verification_state, "unverified")
        self.assertEqual(resolved.sides[1].verification_state, "unverified")
        leftover = next(
            item for item in found if item.status == "open" and "fee-yes" in item.fact_ids()
        )
        hooked = resolve_by_verified_fact(store, "brain", "fee-yes")
        self.assertEqual(len(hooked), 1)
        self.assertEqual(hooked[0].id, leftover.id)
        self.assertEqual(hooked[0].status, "resolved")
        self.assertEqual(hooked[0].resolution.method, "verified_side")
        self.assertEqual(hooked[0].sides[0].verification_state, "unverified")

    def test_report_json_and_markdown(self):
        store = MemoryConflictStore()
        detect_conflicts(store, FakeGraph(self._refund_triples()), "brain")
        report = build_report(store, "brain", fmt="markdown")
        self.assertIn("markdown", report)
        self.assertIn("Conflict report", report["markdown"])
        self.assertGreater(report["summary"]["open"], 0)
        text = render_markdown([], {"generated_at": "now", "open": 0, "total": 0})
        self.assertIn("No conflicts matched", text)

    def test_ingest_flag_defaults_off(self):
        with patch.dict("os.environ", {"CONFLICT_DETECT_ON_INGEST": "false"}):
            self.assertFalse(conflict_detect_on_ingest_enabled())
            fake_tasks = MagicMock()
            with patch.dict("sys.modules", {"src.workers.tasks.conflicts": fake_tasks}):
                maybe_detect_after_ingest("brain", ["fact-1"])
                fake_tasks.detect_conflicts_incremental.apply_async.assert_not_called()

    def test_maybe_detect_enqueues_when_flag_on(self):
        fake_tasks = MagicMock()
        with patch.dict("os.environ", {"CONFLICT_DETECT_ON_INGEST": "true"}):
            with patch.dict("sys.modules", {"src.workers.tasks.conflicts": fake_tasks}):
                maybe_detect_after_ingest("brain", ["fact-1"])
        fake_tasks.detect_conflicts_incremental.apply_async.assert_called_once()

    def test_verified_side_asks_verification_adapter(self):
        from src.core.saving.conflict_verification import apply_verified_side_to_facts

        actions: list[tuple[str, str]] = []

        def fake_apply(graph, *, fact_id, brain_id, action, verified_by=None, note=None):
            actions.append((fact_id, action))
            return True

        applied = apply_verified_side_to_facts(
            object(),
            "brain",
            winning_fact_id="a",
            losing_fact_ids=["b"],
            apply_action=fake_apply,
        )
        self.assertTrue(applied)
        self.assertEqual(actions, [("a", "verify"), ("b", "reject")])

    def test_hydrate_sides_from_verification_reader(self):
        from src.constants.conflicts import ConflictEntity, ConflictSide, FactConflict
        from src.core.saving.conflict_verification import hydrate_conflict_sides

        conflict = FactConflict(
            id="c",
            pair_key="p",
            claim_key="k",
            disagreement_type="value_mismatch",
            subject=ConflictEntity(uuid="s", name="S"),
            predicate="HAS",
            sides=[
                ConflictSide(
                    fact_id="a",
                    subject=ConflictEntity(),
                    predicate="HAS",
                    object=ConflictEntity(),
                ),
                ConflictSide(
                    fact_id="b",
                    subject=ConflictEntity(),
                    predicate="HAS",
                    object=ConflictEntity(),
                ),
            ],
        )
        states = {"a": "verified", "b": "rejected"}
        hydrate_conflict_sides(
            conflict,
            graph=object(),
            brain_id="brain",
            read_state=lambda fact_id, **kwargs: states[fact_id],
        )
        self.assertEqual(conflict.sides[0].verification_state, "verified")
        self.assertEqual(conflict.sides[1].verification_state, "rejected")

    def test_verification_adapter_absent_is_noop(self):
        from src.core.saving.conflict_verification import apply_fact_verification_action

        self.assertFalse(
            apply_fact_verification_action(
                None, fact_id="a", brain_id="brain", action="verify"
            )
        )
        self.assertFalse(
            apply_fact_verification_action(
                object(), fact_id="a", brain_id="brain", action="verify"
            )
        )

    def test_resolve_verified_side_calls_adapter_hook_does_not(self):
        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(self._refund_triples()), "brain")
        numeric = next(
            item for item in found if item.disagreement_type == "numeric_mismatch"
        )
        with patch(
            "src.core.saving.conflict_verification.apply_verified_side_to_facts"
        ) as apply_mock:
            apply_mock.return_value = True
            resolve_conflict(
                store,
                "brain",
                numeric.id,
                winning_fact_id=numeric.sides[0].fact_id,
                method="verified_side",
            )
            apply_mock.assert_called_once()
            leftover = next(
                item
                for item in found
                if item.status == "open" and "fee-yes" in item.fact_ids()
            )
            apply_mock.reset_mock()
            hooked = resolve_by_verified_fact(store, "brain", "fee-yes")
            apply_mock.assert_not_called()
        self.assertEqual(len(hooked), 1)
        self.assertEqual(hooked[0].id, leftover.id)

    def test_open_conflict_annotation_payload(self):
        store = MemoryConflictStore()
        detect_conflicts(store, FakeGraph(self._refund_triples()), "brain")
        mapping = open_conflicts_by_fact_ids(store, "brain", ["refund-faq"])
        self.assertTrue(mapping["refund-faq"])
        payload = annotation_payload(mapping["refund-faq"][0], "refund-faq")
        self.assertEqual(payload["id"], mapping["refund-faq"][0].id)
        self.assertIn(payload["disagreement_type"], {"numeric_mismatch", "value_mismatch"})
        self.assertTrue(payload["other_value"])
        empty = open_conflicts_by_fact_ids(store, "brain", ["feat-chat"])
        self.assertEqual(empty["feat-chat"], [])


class McpConflictToolTests(unittest.TestCase):
    def test_unauthorized_and_filters(self):
        store = MemoryConflictStore()
        detect_conflicts(
            store,
            FakeGraph(
                [
                    _triple(
                        _node("p", "RefundPolicy"),
                        _pred(
                            "a",
                            "HAS_WINDOW",
                            {
                                "valid_from": "2024-01-01T00:00:00Z",
                                "source_key": "help/faq",
                            },
                        ),
                        _node("x", "14 days"),
                    ),
                    _triple(
                        _node("p", "RefundPolicy"),
                        _pred(
                            "b",
                            "HAS_WINDOW",
                            {
                                "valid_from": "2024-01-01T00:00:00Z",
                                "source_key": "help/refunds",
                            },
                        ),
                        _node("y", "30 days"),
                    ),
                ]
            ),
            "brain",
        )
        self.assertEqual(
            read_conflicts(store, "brain", authorized=False), "Unauthorized"
        )
        page = read_conflicts(
            store, "brain", authorized=True, status="open", source="help/faq"
        )
        self.assertEqual(page["count"], 1)
        conflict_id = page["conflicts"][0]["id"]
        detail = read_conflict(store, "brain", conflict_id, authorized=True)
        self.assertEqual(detail["conflict"]["id"], conflict_id)
        self.assertEqual(len(detail["conflict"]["sides"]), 2)
        resolved = resolve_conflict_tool(
            store,
            "brain",
            conflict_id,
            authorized=True,
            winning_fact_id="a",
            method="verified_side",
        )
        self.assertEqual(resolved["conflict"]["status"], "resolved")
        report = conflict_report_tool(
            store, "brain", authorized=True, fmt="markdown"
        )
        self.assertIn("markdown", report)

    def test_dismiss_and_report_tools_registered(self):
        names = {spec["name"] for spec in MCP_TOOL_SPECS}
        for name in (
            "list_conflicts",
            "get_conflict",
            "resolve_conflict",
            "dismiss_conflict",
            "conflict_report",
        ):
            self.assertIn(name, names)
        store = MemoryConflictStore()
        detect_conflicts(
            store,
            FakeGraph(
                [
                    _triple(
                        _node("p", "RefundPolicy"),
                        _pred(
                            "a",
                            "HAS_WINDOW",
                            {
                                "valid_from": "2024-01-01T00:00:00Z",
                                "source_key": "s1",
                            },
                        ),
                        _node("x", "14 days"),
                    ),
                    _triple(
                        _node("p", "RefundPolicy"),
                        _pred(
                            "b",
                            "HAS_WINDOW",
                            {
                                "valid_from": "2024-01-01T00:00:00Z",
                                "source_key": "s2",
                            },
                        ),
                        _node("y", "30 days"),
                    ),
                ]
            ),
            "brain",
        )
        conflict_id = next(iter(store.rows))
        dismissed = dismiss_conflict_tool(
            store, "brain", conflict_id, authorized=True, notes="noise"
        )
        self.assertEqual(dismissed["conflict"]["status"], "dismissed")


class ConfigDefaultTests(unittest.TestCase):
    def test_config_flags_default_off(self):
        from src.config import config

        self.assertFalse(config.conflict_detect_on_ingest)
        self.assertFalse(config.conflict_detect_llm)


if __name__ == "__main__":
    unittest.main()

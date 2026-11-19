import unittest
from unittest.mock import MagicMock, patch

from src.constants.kg import Node, Predicate, _coerce_datetime
from src.core.auth.scopes import VERIFICATION_PERMISSION_DENIED
from src.core.saving.conflicts import (
    VerificationPermissionDenied,
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
    parse_date_value,
    parse_quantity,
    resolve_by_verified_fact,
    resolve_conflict,
    resolve_conflict_authorized,
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
        self.calls = []

    def list_validity_relationships(
        self,
        brain_id,
        skip=0,
        limit=100,
        since=None,
        until=None,
        subject_uuids=None,
        relationship_uuids=None,
    ):
        self.calls.append(
            {
                "brain_id": brain_id,
                "skip": skip,
                "limit": limit,
                "subject_uuids": list(subject_uuids) if subject_uuids else None,
                "relationship_uuids": (
                    list(relationship_uuids) if relationship_uuids else None
                ),
            }
        )
        items = self.triples
        if subject_uuids:
            allowed = set(subject_uuids)
            items = [item for item in items if item[0].uuid in allowed]
        if relationship_uuids:
            allowed_rels = set(relationship_uuids)
            items = [item for item in items if item[1].uuid in allowed_rels]
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
        with patch(
            "src.core.saving.conflict_verification.apply_verified_side_to_facts",
            return_value="applied",
        ):
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
                maybe_detect_after_ingest(
                    "brain", ["fact-1"], subject_uuids=["policy-1"]
                )
        fake_tasks.detect_conflicts_incremental.apply_async.assert_called_once()
        args = fake_tasks.detect_conflicts_incremental.apply_async.call_args.kwargs.get(
            "args"
        ) or fake_tasks.detect_conflicts_incremental.apply_async.call_args[0][0]
        payload = args[0] if args else {}
        self.assertEqual(payload["fact_ids"], ["fact-1"])
        self.assertEqual(payload["subject_uuids"], ["policy-1"])

    def test_verified_side_asks_verification_adapter(self):
        from src.core.saving.conflict_verification import (
            VerificationSnapshot,
            apply_verified_side_to_facts,
        )

        actions: list[tuple[str, str]] = []

        def fake_apply(
            graph, *, fact_id, brain_id, action, actor=None, verified_by=None, note=None
        ):
            actions.append((fact_id, action, actor or verified_by))
            return "applied"

        applied = apply_verified_side_to_facts(
            object(),
            "brain",
            winning_fact_id="a",
            losing_fact_ids=["b"],
            actor="brainpat:demo",
            apply_action=fake_apply,
            read_snapshot=lambda fact_id, **kwargs: VerificationSnapshot(
                "unverified", kind="relationship"
            ),
        )
        self.assertEqual(applied, "applied")
        self.assertEqual(
            actions,
            [("a", "verify", "brainpat:demo"), ("b", "reject", "brainpat:demo")],
        )

    def test_verified_side_skips_reject_when_verify_fails(self):
        from src.core.saving.conflict_verification import (
            VerificationSnapshot,
            apply_verified_side_to_facts,
        )

        actions: list[tuple[str, str]] = []

        def fake_apply(
            graph, *, fact_id, brain_id, action, actor=None, verified_by=None, note=None
        ):
            actions.append((fact_id, action))
            return "failed"

        result = apply_verified_side_to_facts(
            object(),
            "brain",
            winning_fact_id="a",
            losing_fact_ids=["b"],
            apply_action=fake_apply,
            read_snapshot=lambda fact_id, **kwargs: VerificationSnapshot(
                "unverified", kind="relationship"
            ),
        )
        self.assertEqual(result, "failed")
        self.assertEqual(actions, [("a", "verify")])

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

        self.assertEqual(
            apply_fact_verification_action(
                None, fact_id="a", brain_id="brain", action="verify"
            ),
            "absent",
        )
        with patch.dict("sys.modules", {"src.core.verification.service": None}):
            self.assertEqual(
                apply_fact_verification_action(
                    object(), fact_id="a", brain_id="brain", action="verify"
                ),
                "absent",
            )

    def test_verified_side_requires_winner(self):
        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(self._refund_triples()), "brain")
        conflict_id = found[0].id
        with self.assertRaises(ValueError):
            resolve_conflict(store, "brain", conflict_id, method="verified_side")
        self.assertEqual(store.rows[conflict_id].status, "open")
        manual = resolve_conflict(store, "brain", conflict_id, method="manual")
        self.assertEqual(manual.status, "resolved")
        self.assertEqual(manual.resolution.method, "manual")

    def test_verified_side_failure_keeps_conflict_open(self):
        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(self._refund_triples()), "brain")
        numeric = next(
            item for item in found if item.disagreement_type == "numeric_mismatch"
        )
        with patch(
            "src.core.saving.conflict_verification.apply_verified_side_to_facts"
        ) as apply_mock:
            apply_mock.return_value = "failed"
            with self.assertRaises(ValueError):
                resolve_conflict(
                    store,
                    "brain",
                    numeric.id,
                    winning_fact_id=numeric.sides[0].fact_id,
                    method="verified_side",
                )
        self.assertEqual(store.rows[numeric.id].status, "open")

    def test_resolve_verified_side_calls_adapter_hook_does_not(self):
        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(self._refund_triples()), "brain")
        numeric = next(
            item for item in found if item.disagreement_type == "numeric_mismatch"
        )
        with patch(
            "src.core.saving.conflict_verification.apply_verified_side_to_facts"
        ) as apply_mock:
            apply_mock.return_value = "applied"
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
        missing_winner = resolve_conflict_tool(
            store,
            "brain",
            conflict_id,
            authorized=True,
            method="verified_side",
            actor="brainpat:brain",
            has_verification=True,
        )
        self.assertIn("winning_fact_id", str(missing_winner))
        with patch(
            "src.core.saving.conflict_verification.apply_verified_side_to_facts",
            return_value="applied",
        ):
            resolved = resolve_conflict_tool(
                store,
                "brain",
                conflict_id,
                authorized=True,
                winning_fact_id="a",
                method="verified_side",
                actor="brainpat:brain",
                has_verification=True,
            )
        self.assertEqual(resolved["conflict"]["status"], "resolved")
        self.assertEqual(resolved["conflict"]["resolution"]["resolved_by"], "brainpat:brain")
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

    def test_verified_side_denied_without_verification_scope(self):
        store = MemoryConflictStore()
        detect_conflicts(
            store,
            FakeGraph(
                [
                    _triple(
                        _node("p", "RefundPolicy"),
                        _pred("a", "HAS_WINDOW", {"source_key": "s1"}),
                        _node("x", "14 days"),
                    ),
                    _triple(
                        _node("p", "RefundPolicy"),
                        _pred("b", "HAS_WINDOW", {"source_key": "s2"}),
                        _node("y", "30 days"),
                    ),
                ]
            ),
            "brain",
        )
        conflict_id = next(iter(store.rows))
        denied = resolve_conflict_tool(
            store,
            "brain",
            conflict_id,
            authorized=True,
            winning_fact_id="a",
            method="verified_side",
            actor="brainpat:brain",
            has_verification=False,
        )
        self.assertIn(VERIFICATION_PERMISSION_DENIED, str(denied))
        self.assertEqual(store.rows[conflict_id].status, "open")
        manual = resolve_conflict_tool(
            store,
            "brain",
            conflict_id,
            authorized=True,
            winning_fact_id="a",
            method="manual",
            actor="brainpat:brain",
            has_verification=False,
        )
        self.assertEqual(manual["conflict"]["status"], "resolved")
        self.assertEqual(manual["conflict"]["resolution"]["resolved_by"], "brainpat:brain")


class AuthorizedResolveTests(unittest.TestCase):
    def test_authorized_resolve_requires_scope_for_verified_side(self):
        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(DetectionEngineTests()._refund_triples()), "brain")
        numeric = next(
            item for item in found if item.disagreement_type == "numeric_mismatch"
        )
        with self.assertRaises(VerificationPermissionDenied):
            resolve_conflict_authorized(
                store,
                "brain",
                numeric.id,
                authorized=True,
                actor="brainpat:brain",
                has_verification=False,
                winning_fact_id=numeric.sides[0].fact_id,
                method="verified_side",
            )
        self.assertEqual(store.rows[numeric.id].status, "open")

    def test_authorized_resolve_records_actor_when_allowed(self):
        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(DetectionEngineTests()._refund_triples()), "brain")
        numeric = next(
            item for item in found if item.disagreement_type == "numeric_mismatch"
        )
        with patch(
            "src.core.saving.conflict_verification.apply_verified_side_to_facts",
            return_value="applied",
        ):
            resolved = resolve_conflict_authorized(
                store,
                "brain",
                numeric.id,
                authorized=True,
                actor="oauth:client-1",
                has_verification=True,
                winning_fact_id=numeric.sides[0].fact_id,
                method="verified_side",
            )
        self.assertEqual(resolved.status, "resolved")
        self.assertEqual(resolved.resolution.resolved_by, "oauth:client-1")

    def test_reject_failure_resets_winner_and_keeps_conflict_open(self):
        from src.core.saving.conflict_verification import (
            VerificationSnapshot,
            apply_verified_side_to_facts,
        )

        actions: list[tuple[str, str]] = []
        restores: list[tuple[str, VerificationSnapshot]] = []
        snapshots = {
            "a": VerificationSnapshot("unverified", kind="relationship"),
            "b": VerificationSnapshot("unverified", kind="relationship"),
        }

        def fake_apply(
            graph, *, fact_id, brain_id, action, actor=None, verified_by=None, note=None
        ):
            actions.append((fact_id, action))
            if action == "reject":
                return "failed"
            return "applied"

        def fake_restore(
            graph, *, fact_id, brain_id, snapshot, from_state=None, actor=None
        ):
            restores.append((fact_id, snapshot))
            return "applied"

        result = apply_verified_side_to_facts(
            object(),
            "brain",
            winning_fact_id="a",
            losing_fact_ids=["b"],
            actor="brainpat:demo",
            apply_action=fake_apply,
            read_snapshot=lambda fact_id, **kwargs: snapshots[fact_id],
            restore_snapshot=fake_restore,
        )
        self.assertEqual(result, "failed")
        self.assertFalse(result.rollback_failed)
        self.assertEqual(actions, [("a", "verify"), ("b", "reject")])
        self.assertEqual(restores, [("a", snapshots["a"])])

        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(DetectionEngineTests()._refund_triples()), "brain")
        numeric = next(
            item for item in found if item.disagreement_type == "numeric_mismatch"
        )
        with patch(
            "src.core.saving.conflict_verification.apply_verified_side_to_facts",
            return_value="failed",
        ):
            with self.assertRaises(ValueError):
                resolve_conflict(
                    store,
                    "brain",
                    numeric.id,
                    winning_fact_id=numeric.sides[0].fact_id,
                    method="verified_side",
                    resolved_by="brainpat:demo",
                )
        self.assertEqual(store.rows[numeric.id].status, "open")

    def test_reject_failure_restores_expert_verified_winner(self):
        from src.core.saving.conflict_verification import (
            VerificationSnapshot,
            apply_verified_side_to_facts,
        )

        actions: list[tuple[str, str]] = []
        restores: list[tuple[str, VerificationSnapshot, str | None, str | None]] = []
        expert = VerificationSnapshot(
            "verified",
            verified_by="expert:alice",
            verified_at="2024-06-01T12:00:00Z",
            verification_source="manual",
            reason="checked against source",
            kind="relationship",
        )
        snapshots = {
            "a": expert,
            "b": VerificationSnapshot("unverified", kind="relationship"),
        }

        def fake_apply(
            graph, *, fact_id, brain_id, action, actor=None, verified_by=None, note=None
        ):
            actions.append((fact_id, action))
            if fact_id == "b" and action == "reject":
                return "failed"
            return "applied"

        def fake_restore(
            graph, *, fact_id, brain_id, snapshot, from_state=None, actor=None
        ):
            restores.append((fact_id, snapshot, from_state, actor))
            return "applied"

        result = apply_verified_side_to_facts(
            object(),
            "brain",
            winning_fact_id="a",
            losing_fact_ids=["b"],
            actor="brainpat:demo",
            apply_action=fake_apply,
            read_snapshot=lambda fact_id, **kwargs: snapshots[fact_id],
            restore_snapshot=fake_restore,
        )
        self.assertEqual(result, "failed")
        self.assertFalse(result.rollback_failed)
        self.assertEqual(actions, [("a", "verify"), ("b", "reject")])
        self.assertEqual(len(restores), 1)
        self.assertEqual(restores[0][0], "a")
        self.assertEqual(restores[0][1].verified_by, "expert:alice")
        self.assertEqual(restores[0][1].verified_at, "2024-06-01T12:00:00Z")
        self.assertEqual(restores[0][1].verification_source, "manual")
        self.assertEqual(restores[0][1].reason, "checked against source")
        self.assertEqual(restores[0][2], "verified")
        self.assertNotEqual(restores[0][1].verified_by, "brainpat:demo")

    def test_reject_failure_restores_earlier_losers_when_later_reject_fails(self):
        from src.core.saving.conflict_verification import (
            VerificationSnapshot,
            apply_verified_side_to_facts,
        )

        actions: list[tuple[str, str]] = []
        restores: list[str] = []
        snapshots = {
            "w": VerificationSnapshot("unverified", kind="relationship"),
            "l1": VerificationSnapshot("unverified", kind="relationship"),
            "l2": VerificationSnapshot("rejected", kind="relationship"),
        }

        def fake_apply(
            graph, *, fact_id, brain_id, action, actor=None, verified_by=None, note=None
        ):
            actions.append((fact_id, action))
            if fact_id == "l2" and action == "reject":
                return "failed"
            return "applied"

        def fake_restore(
            graph, *, fact_id, brain_id, snapshot, from_state=None, actor=None
        ):
            restores.append(fact_id)
            return "applied"

        result = apply_verified_side_to_facts(
            object(),
            "brain",
            winning_fact_id="w",
            losing_fact_ids=["l1", "l2"],
            actor="brainpat:demo",
            apply_action=fake_apply,
            read_snapshot=lambda fact_id, **kwargs: snapshots[fact_id],
            restore_snapshot=fake_restore,
        )
        self.assertEqual(result, "failed")
        self.assertFalse(result.rollback_failed)
        self.assertEqual(actions, [("w", "verify"), ("l1", "reject"), ("l2", "reject")])
        self.assertEqual(restores, ["l1", "w"])

    def test_preread_none_aborts_before_writes(self):
        from src.core.saving.conflict_verification import apply_verified_side_to_facts

        actions: list[tuple[str, str]] = []
        restores: list[str] = []

        def fake_apply(
            graph, *, fact_id, brain_id, action, actor=None, verified_by=None, note=None
        ):
            actions.append((fact_id, action))
            return "applied"

        result = apply_verified_side_to_facts(
            object(),
            "brain",
            winning_fact_id="a",
            losing_fact_ids=["b"],
            actor="brainpat:demo",
            apply_action=fake_apply,
            read_snapshot=lambda fact_id, **kwargs: None,
            restore_snapshot=lambda **kwargs: restores.append(kwargs["fact_id"]) or "applied",
        )
        self.assertEqual(result, "failed")
        self.assertFalse(result.rollback_failed)
        self.assertEqual(result.rollback_errors, ("a pre-read returned none",))
        self.assertEqual(actions, [])
        self.assertEqual(restores, [])

        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(DetectionEngineTests()._refund_triples()), "brain")
        numeric = next(
            item for item in found if item.disagreement_type == "numeric_mismatch"
        )
        with patch(
            "src.core.saving.conflict_verification.apply_verified_side_to_facts",
            return_value=result,
        ):
            with self.assertRaises(ValueError) as raised:
                resolve_conflict(
                    store,
                    "brain",
                    numeric.id,
                    winning_fact_id=numeric.sides[0].fact_id,
                    method="verified_side",
                    resolved_by="brainpat:demo",
                )
        self.assertIn("pre-read returned none", str(raised.exception))
        self.assertEqual(store.rows[numeric.id].status, "open")

    def test_rollback_failure_is_surfaced_and_conflict_stays_open(self):
        from src.core.saving.conflict_verification import VerifiedSideApplyResult

        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(DetectionEngineTests()._refund_triples()), "brain")
        numeric = next(
            item for item in found if item.disagreement_type == "numeric_mismatch"
        )
        with patch(
            "src.core.saving.conflict_verification.apply_verified_side_to_facts",
            return_value=VerifiedSideApplyResult(
                "failed",
                rollback_failed=True,
                rollback_errors=("a restore returned failed",),
            ),
        ):
            with self.assertRaises(ValueError) as raised:
                resolve_conflict(
                    store,
                    "brain",
                    numeric.id,
                    winning_fact_id=numeric.sides[0].fact_id,
                    method="verified_side",
                    resolved_by="brainpat:demo",
                )
        self.assertIn("rollback failed", str(raised.exception))
        self.assertIn("a restore returned failed", str(raised.exception))
        self.assertEqual(store.rows[numeric.id].status, "open")

    def test_restore_checks_each_result_when_rollback_fails(self):
        from src.core.saving.conflict_verification import (
            VerificationSnapshot,
            apply_verified_side_to_facts,
        )

        actions: list[tuple[str, str]] = []

        def fake_apply(
            graph, *, fact_id, brain_id, action, actor=None, verified_by=None, note=None
        ):
            actions.append((fact_id, action))
            if action == "reject":
                return "failed"
            return "applied"

        result = apply_verified_side_to_facts(
            object(),
            "brain",
            winning_fact_id="w",
            losing_fact_ids=["l1", "l2"],
            actor="brainpat:demo",
            apply_action=fake_apply,
            read_snapshot=lambda fact_id, **kwargs: VerificationSnapshot(
                "unverified", kind="relationship"
            ),
            restore_snapshot=lambda *args, **kwargs: "failed",
        )
        self.assertEqual(result, "failed")
        self.assertTrue(result.rollback_failed)
        self.assertEqual(result.rollback_errors, ("w restore returned failed",))
        self.assertEqual(actions, [("w", "verify"), ("l1", "reject")])

    def test_graph_restore_keeps_original_verifier(self):
        from src.core.saving.conflict_verification import apply_verified_side_to_facts
        from src.core.verification.facts import parse_history

        class MemoryVerificationGraph:
            def __init__(self, rows):
                self.rows = {row["uuid"]: row for row in rows}
                self.fail_on = None

            def get_fact_record(self, fact_id, brain_id):
                return self.rows.get(fact_id)

            def set_fact_verification(
                self, fact_id, brain_id, *, kind, properties, history_entry=None
            ):
                from src.core.verification.facts import serialize_history_entry

                if self.fail_on == fact_id:
                    raise RuntimeError("injected write failure")
                row = self.rows[fact_id]
                props = dict(row.get("properties") or {})
                props.update(properties)
                entry = serialize_history_entry(history_entry)
                if entry:
                    history = list(props.get("verification_history") or [])
                    history.append(entry)
                    props["verification_history"] = history
                row["properties"] = props
                row["kind"] = kind
                return row

        winner = {
            "kind": "relationship",
            "uuid": "a",
            "name": "HAS_WINDOW",
            "description": "",
            "properties": {
                "uuid": "a",
                "verification_state": "verified",
                "verified_by": "expert:alice",
                "verified_at": "2024-06-01T12:00:00Z",
                "verification_source": "manual",
                "verification_note": "checked against source",
            },
            "labels": [],
        }
        loser = {
            "kind": "relationship",
            "uuid": "b",
            "name": "HAS_WINDOW",
            "description": "",
            "properties": {"uuid": "b", "verification_state": "unverified"},
            "labels": [],
        }
        graph = MemoryVerificationGraph([winner, loser])
        graph.fail_on = "b"
        result = apply_verified_side_to_facts(
            graph,
            "brain",
            winning_fact_id="a",
            losing_fact_ids=["b"],
            actor="brainpat:demo",
        )
        self.assertEqual(result, "failed")
        self.assertFalse(result.rollback_failed)
        restored = graph.rows["a"]["properties"]
        self.assertEqual(restored["verification_state"], "verified")
        self.assertEqual(restored["verified_by"], "expert:alice")
        self.assertEqual(restored["verified_at"], "2024-06-01T12:00:00Z")
        self.assertEqual(restored["verification_source"], "manual")
        self.assertEqual(restored["verification_note"], "checked against source")
        history = parse_history(restored.get("verification_history"))
        self.assertEqual(history[-1].verification_source, "rollback")
        self.assertEqual(history[-1].note, "rollback")
        self.assertEqual(history[-1].to_state, "verified")
        self.assertNotEqual(restored["verified_by"], "brainpat:demo")


class RestConflictResolveAuthTests(unittest.TestCase):
    def _seed(self):
        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(DetectionEngineTests()._refund_triples()), "brain")
        numeric = next(
            item for item in found if item.disagreement_type == "numeric_mismatch"
        )
        return store, numeric

    def test_rest_verified_side_denied_without_scope(self):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient

        from src.core.auth.scopes import VERIFICATION_PERMISSION_DENIED
        from src.services.api.errors import install_error_handlers
        from src.services.api.routes.retrieve import retrieve_router

        store, numeric = self._seed()
        app = FastAPI()
        install_error_handlers(app)

        @app.middleware("http")
        async def inject_brain(request: Request, call_next):
            request.state.brain_id = "brain"
            request.state.principal = "brainpat:brain"
            request.state.scopes = ["brainapi"]
            return await call_next(request)

        app.include_router(retrieve_router)
        api = TestClient(app)
        with patch(
            "src.services.api.controllers.conflicts.data_adapter", store
        ), patch(
            "src.core.saving.conflict_verification.apply_verified_side_to_facts",
            return_value="applied",
        ):
            denied = api.post(
                f"/retrieve/conflicts/{numeric.id}/resolve",
                json={
                    "winning_fact_id": numeric.sides[0].fact_id,
                    "method": "verified_side",
                },
            )
        self.assertEqual(denied.status_code, 403)
        self.assertEqual(denied.json()["error"]["code"], VERIFICATION_PERMISSION_DENIED)
        self.assertEqual(store.rows[numeric.id].status, "open")

    def test_rest_verified_side_allowed_with_scope_records_actor(self):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient

        from src.services.api.errors import install_error_handlers
        from src.services.api.routes.retrieve import retrieve_router

        store, numeric = self._seed()
        app = FastAPI()
        install_error_handlers(app)

        @app.middleware("http")
        async def inject_brain(request: Request, call_next):
            request.state.brain_id = "brain"
            request.state.principal = "brainpat:brain"
            request.state.scopes = ["brainapi", "verification"]
            return await call_next(request)

        app.include_router(retrieve_router)
        api = TestClient(app)
        with patch(
            "src.services.api.controllers.conflicts.data_adapter", store
        ), patch(
            "src.core.saving.conflict_verification.apply_verified_side_to_facts",
            return_value="applied",
        ) as apply_mock:
            allowed = api.post(
                f"/retrieve/conflicts/{numeric.id}/resolve",
                json={
                    "winning_fact_id": numeric.sides[0].fact_id,
                    "method": "verified_side",
                    "verified_by": "attacker",
                },
            )
        self.assertEqual(allowed.status_code, 200, allowed.text)
        body = allowed.json()["conflict"]
        self.assertEqual(body["status"], "resolved")
        self.assertEqual(body["resolution"]["resolved_by"], "brainpat:brain")
        self.assertEqual(apply_mock.call_args.kwargs.get("actor"), "brainpat:brain")


class ReviewGapTests(unittest.TestCase):
    def test_ingest_scan_is_scoped_to_touched_subjects(self):
        policy = _node("policy-1", "RefundPolicy")
        other = _node("other-1", "OtherPolicy")
        triples = [
            _triple(
                policy,
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
                policy,
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
            _triple(
                other,
                _pred(
                    "c",
                    "HAS_WINDOW",
                    {
                        "valid_from": "2024-01-01T00:00:00Z",
                        "source_key": "s3",
                    },
                ),
                _node("z", "7 days"),
            ),
            _triple(
                other,
                _pred(
                    "d",
                    "HAS_WINDOW",
                    {
                        "valid_from": "2024-01-01T00:00:00Z",
                        "source_key": "s4",
                    },
                ),
                _node("w", "90 days"),
            ),
        ]
        graph = FakeGraph(triples)
        found = detect_conflicts(
            MemoryConflictStore(),
            graph,
            "brain",
            fact_ids=["a"],
            subject_uuids=["policy-1"],
        )
        self.assertTrue(graph.calls)
        self.assertEqual(graph.calls[0]["subject_uuids"], ["policy-1"])
        pairs = {tuple(sorted(item.fact_ids())) for item in found}
        self.assertEqual(pairs, {("a", "b")})
        self.assertFalse(found.truncated)
        self.assertEqual(found.scanned_relationships, 2)

    def test_ingest_scan_looks_up_subjects_from_fact_ids(self):
        policy = _node("policy-1", "RefundPolicy")
        other = _node("other-1", "OtherPolicy")
        triples = [
            _triple(
                policy,
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
                policy,
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
            _triple(
                other,
                _pred(
                    "c",
                    "HAS_WINDOW",
                    {
                        "valid_from": "2024-01-01T00:00:00Z",
                        "source_key": "s3",
                    },
                ),
                _node("z", "7 days"),
            ),
            _triple(
                other,
                _pred(
                    "d",
                    "HAS_WINDOW",
                    {
                        "valid_from": "2024-01-01T00:00:00Z",
                        "source_key": "s4",
                    },
                ),
                _node("w", "90 days"),
            ),
        ]
        graph = FakeGraph(triples)
        found = detect_conflicts(
            MemoryConflictStore(),
            graph,
            "brain",
            fact_ids=["a"],
        )
        self.assertTrue(
            any(call.get("relationship_uuids") == ["a"] for call in graph.calls)
        )
        self.assertTrue(
            any(call.get("subject_uuids") == ["policy-1"] for call in graph.calls)
        )
        pairs = {tuple(sorted(item.fact_ids())) for item in found}
        self.assertEqual(pairs, {("a", "b")})

    def test_scan_pages_all_relationships_and_reports_cap(self):
        policy = _node("policy-1", "RefundPolicy")
        triples = [
            _triple(
                policy,
                _pred(
                    f"f{i}",
                    "HAS_WINDOW",
                    {
                        "valid_from": "2024-01-01T00:00:00Z",
                        "source_key": f"s{i}",
                    },
                ),
                _node(f"v{i}", f"{i} days"),
            )
            for i in range(5)
        ]
        uncapped = detect_conflicts(
            MemoryConflictStore(), FakeGraph(triples), "brain"
        )
        self.assertEqual(uncapped.scanned_relationships, 5)
        self.assertEqual(uncapped.total_relationships, 5)
        self.assertFalse(uncapped.truncated)
        capped = detect_conflicts(
            MemoryConflictStore(),
            FakeGraph(triples),
            "brain",
            max_relationships=2,
        )
        self.assertTrue(capped.truncated)
        self.assertEqual(capped.scanned_relationships, 2)
        self.assertEqual(capped.total_relationships, 5)

    def test_heuristic_requires_incompatible_value_and_normalizes(self):
        self.assertTrue(values_equivalent("Warehouse A", "warehouse a"))
        self.assertTrue(values_equivalent("30 Days", "30 days"))
        self.assertTrue(values_equivalent("2024-01-01", "January 1, 2024"))
        self.assertFalse(values_equivalent("2024-01-01", "2024-06-01"))
        self.assertIsNotNone(parse_date_value("January 1, 2024"))
        snapshots = [
            snapshot_from_triple(item)
            for item in [
                _triple(
                    _node("p", "Policy"),
                    _pred(
                        "a",
                        "HAS_POLICY",
                        {
                            "valid_from": "2024-01-01T00:00:00Z",
                            "source_key": "s1",
                        },
                    ),
                    _node("full", "full refund"),
                ),
                _triple(
                    _node("p", "Policy"),
                    _pred(
                        "b",
                        "HAS_POLICY",
                        {
                            "valid_from": "2024-01-01T00:00:00Z",
                            "source_key": "s2",
                        },
                    ),
                    _node("credit", "store credit only"),
                ),
            ]
        ]
        definite, semantic = find_conflict_candidates(snapshots)
        self.assertEqual(definite, [])
        self.assertEqual(len(semantic), 1)
        found = detect_conflicts(
            MemoryConflictStore(),
            FakeGraph(
                [
                    _triple(
                        _node("p", "Policy"),
                        _pred(
                            "a",
                            "HAS_POLICY",
                            {
                                "valid_from": "2024-01-01T00:00:00Z",
                                "source_key": "s1",
                            },
                        ),
                        _node("full", "full refund"),
                    ),
                    _triple(
                        _node("p", "Policy"),
                        _pred(
                            "b",
                            "HAS_POLICY",
                            {
                                "valid_from": "2024-01-01T00:00:00Z",
                                "source_key": "s2",
                            },
                        ),
                        _node("credit", "store credit only"),
                    ),
                ]
            ),
            "brain",
            use_llm=False,
        )
        self.assertEqual(len(found), 0)
        dates = detect_conflicts(
            MemoryConflictStore(),
            FakeGraph(
                [
                    _triple(
                        _node("p", "Policy"),
                        _pred(
                            "a",
                            "DEADLINE",
                            {
                                "valid_from": "2024-01-01T00:00:00Z",
                                "source_key": "s1",
                            },
                        ),
                        _node("d1", "2024-01-01"),
                    ),
                    _triple(
                        _node("p", "Policy"),
                        _pred(
                            "b",
                            "DEADLINE",
                            {
                                "valid_from": "2024-01-01T00:00:00Z",
                                "source_key": "s2",
                            },
                        ),
                        _node("d2", "January 1, 2024"),
                    ),
                    _triple(
                        _node("p", "Policy"),
                        _pred(
                            "c",
                            "DEADLINE",
                            {
                                "valid_from": "2024-01-01T00:00:00Z",
                                "source_key": "s3",
                            },
                        ),
                        _node("d3", "2024-06-01"),
                    ),
                ]
            ),
            "brain",
            use_llm=False,
        )
        pairs = {tuple(sorted(item.fact_ids())) for item in dates}
        self.assertIn(("a", "c"), pairs)
        self.assertIn(("b", "c"), pairs)
        self.assertNotIn(("a", "b"), pairs)

    def test_llm_truncation_is_reported(self):
        triples = []
        policy = _node("p", "Policy")
        values = ("full refund", "store credit only", "exchange only")
        for i, value in enumerate(values):
            triples.append(
                _triple(
                    policy,
                    _pred(
                        f"a{i}",
                        "HAS_POLICY",
                        {
                            "valid_from": "2024-01-01T00:00:00Z",
                            "source_key": f"s{i}",
                        },
                    ),
                    _node(f"v{i}", value),
                )
            )
        client = MagicMock()
        client.generate_json.return_value = {"results": []}
        with patch.dict("os.environ", {"CONFLICT_DETECT_LLM_MAX_PAIRS": "1"}):
            found = detect_conflicts(
                MemoryConflictStore(),
                FakeGraph(triples),
                "brain",
                use_llm=True,
                llm_client=client,
            )
        self.assertTrue(found.llm_truncated)
        self.assertEqual(found.llm_pairs_total, 3)
        self.assertEqual(found.llm_pairs_judged, 1)
        self.assertEqual(client.generate_json.call_count, 1)

    def test_llm_unlimited_budget_batches_all_pairs(self):
        triples = []
        policy = _node("p", "Policy")
        values = ("full refund", "store credit only", "exchange only")
        for i, value in enumerate(values):
            triples.append(
                _triple(
                    policy,
                    _pred(
                        f"a{i}",
                        "HAS_POLICY",
                        {
                            "valid_from": "2024-01-01T00:00:00Z",
                            "source_key": f"s{i}",
                        },
                    ),
                    _node(f"v{i}", value),
                )
            )
        client = MagicMock()
        client.generate_json.return_value = {"results": []}
        with patch.dict("os.environ", {"CONFLICT_DETECT_LLM_MAX_PAIRS": "0"}):
            found = detect_conflicts(
                MemoryConflictStore(),
                FakeGraph(triples),
                "brain",
                use_llm=True,
                llm_client=client,
            )
        self.assertFalse(found.llm_truncated)
        self.assertEqual(found.llm_pairs_total, 3)
        self.assertEqual(found.llm_pairs_judged, 3)
        self.assertEqual(client.generate_json.call_count, 1)

    def test_stale_conflicts_auto_resolve(self):
        store = MemoryConflictStore()
        live = [
            _triple(
                _node("p", "Policy"),
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
                _node("p", "Policy"),
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
        found = detect_conflicts(store, FakeGraph(live), "brain")
        self.assertEqual(len(found), 1)
        conflict_id = found[0].id
        remaining = [live[0]]
        again = detect_conflicts(store, FakeGraph(remaining), "brain")
        self.assertEqual(store.rows[conflict_id].status, "resolved")
        self.assertEqual(store.rows[conflict_id].resolution.method, "stale")
        self.assertIn("side_removed", store.rows[conflict_id].resolution.notes)
        self.assertEqual(again.stale_resolved, 1)

    def test_stale_conflicts_when_values_become_compatible(self):
        store = MemoryConflictStore()
        live = [
            _triple(
                _node("p", "Policy"),
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
                _node("p", "Policy"),
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
        found = detect_conflicts(store, FakeGraph(live), "brain")
        conflict_id = found[0].id
        reconciled = [
            live[0],
            _triple(
                _node("p", "Policy"),
                _pred(
                    "b",
                    "HAS_WINDOW",
                    {
                        "valid_from": "2024-01-01T00:00:00Z",
                        "source_key": "s2",
                    },
                ),
                _node("y", "14 days"),
            ),
        ]
        again = detect_conflicts(store, FakeGraph(reconciled), "brain")
        self.assertEqual(store.rows[conflict_id].status, "resolved")
        self.assertEqual(store.rows[conflict_id].resolution.method, "stale")
        self.assertIn("no_longer_holds", store.rows[conflict_id].resolution.notes)
        self.assertEqual(again.stale_resolved, 1)

    def test_truncated_scan_does_not_auto_resolve_missing_sides(self):
        store = MemoryConflictStore()
        live = [
            _triple(
                _node("p", "Policy"),
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
                _node("p", "Policy"),
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
        found = detect_conflicts(store, FakeGraph(live), "brain")
        conflict_id = found[0].id
        truncated = detect_conflicts(
            store, FakeGraph(live), "brain", max_relationships=1
        )
        self.assertTrue(truncated.truncated)
        self.assertEqual(store.rows[conflict_id].status, "open")
        self.assertEqual(truncated.stale_resolved, 0)

    def test_coerce_datetime_does_not_invent_now(self):
        from datetime import datetime

        self.assertIsNone(_coerce_datetime(None))
        self.assertIsNone(_coerce_datetime(""))
        self.assertIsNone(_coerce_datetime("not-a-timestamp"))
        self.assertIsNone(_coerce_datetime(object()))
        parsed = _coerce_datetime("2024-01-02T03:04:05+00:00")
        self.assertIsInstance(parsed, datetime)
        node = Node(
            uuid="n1",
            name="N",
            labels=["ENTITY"],
            last_updated="bogus",
        )
        self.assertIsNone(node.last_updated)
        pred = Predicate(uuid="p1", name="HAS", description="", last_updated="")
        self.assertIsNone(pred.last_updated)


class ConfigDefaultTests(unittest.TestCase):
    def test_config_flags_default_off(self):
        from src.config import config

        self.assertFalse(config.conflict_detect_on_ingest)
        self.assertFalse(config.conflict_detect_llm)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch

from src.constants.conflicts import ConflictEntity, ConflictSide, ConflictSource, FactConflict
from src.constants.kg import Node, Predicate, _coerce_datetime
from src.core.auth.scopes import VERIFICATION_PERMISSION_DENIED
from src.core.saving.conflicts import (
    VerificationPermissionDenied,
    annotation_payload,
    antonym_or_negation,
    claim_key_for,
    classify_disagreement,
    conflict_detect_max_relationships,
    conflict_detect_on_ingest_enabled,
    conflict_matches_source_filter,
    detect_conflicts,
    dismiss_conflict,
    find_conflict_candidates,
    intervals_overlap,
    is_functional_predicate,
    judge_semantic_pairs,
    maybe_detect_after_ingest,
    normalize_predicate,
    open_conflicts_by_fact_ids,
    pair_key_for,
    parse_date_value,
    parse_quantity,
    present_conflict,
    reopen_conflict,
    resolve_by_verified_fact,
    resolve_conflict,
    resolve_conflict_authorized,
    same_source_revision,
    snapshot_from_triple,
    values_equivalent,
)
from src.core.saving.conflict_report import build_report, render_markdown
from src.lib.neo4j.client import cypher_normalize_predicate
from src.services.mcp.conflicts import (
    conflict_report_tool,
    dismiss_conflict_tool,
    read_conflict,
    read_conflicts,
    resolve_conflict_tool,
)
from src.services.mcp.prompt import MCP_TOOL_SPECS

PREDICATE_NORMALIZER_CASES = (
    ("", ""),
    ("HAS_WINDOW", "HAS_WINDOW"),
    ("has_window", "HAS_WINDOW"),
    ("Has_Window", "HAS_WINDOW"),
    ("HAS-A", "HAS_A"),
    ("HAS A", "HAS_A"),
    ("HAS.A", "HAS_A"),
    ("HAS__A", "HAS_A"),
    ("__HAS_A__", "HAS_A"),
    ("HAS---A", "HAS_A"),
    ("  has a  ", "HAS_A"),
    ("HAS.A-B C", "HAS_A_B_C"),
    ("has-a.b", "HAS_A_B"),
    ("A1", "A1"),
    ("1A", "1A"),
    ("v2.0", "V2_0"),
    ("café", "CAF"),
    ("HAS-é", "HAS"),
    ("naïve", "NA_VE"),
    ("ß", "SS"),
    ("HAS😀A", "HAS_A"),
    ("___", ""),
    ("a.b.c", "A_B_C"),
    ("foo/bar", "FOO_BAR"),
    ("foo\\bar", "FOO_BAR"),
    ("foo:bar", "FOO_BAR"),
    ("(HAS)", "HAS"),
    ("HAS[A]", "HAS_A"),
    ("has a-b.c__d", "HAS_A_B_C_D"),
)


def _node(uuid, name, labels=None, polarity="neutral"):
    return Node(
        uuid=uuid,
        name=name,
        labels=labels or ["ENTITY"],
        properties={"polarity": polarity},
        polarity=polarity,
    )


def _pred(uuid, name, value_props=None, description="", amount=None):
    props = dict(value_props or {})
    return Predicate(
        uuid=uuid,
        name=name,
        description=description,
        properties=props,
        amount=amount,
    )


def _triple(subject, pred, obj):
    return (subject, pred, obj)


class MemoryConflictStore:
    def __init__(self, chunks=None):
        self.rows = {}
        self.chunks = chunks or {}
        self.cursors = {}

    def save_conflict(self, conflict, brain_id):
        self.rows[conflict.id] = conflict
        return conflict

    def update_conflict(
        self, conflict, brain_id, if_status=None, if_resolution_method=None
    ):
        existing = self.rows.get(conflict.id)
        if if_status is not None and (existing is None or existing.status != if_status):
            return None
        if if_resolution_method is not None:
            method = ""
            if existing is not None and existing.resolution is not None:
                method = existing.resolution.method or ""
            if method != if_resolution_method:
                return None
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
            from src.core.saving.conflicts import conflict_matches_source_filter

            items = [
                item for item in items if conflict_matches_source_filter(item, source)
            ]
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

    def get_triples_by_uuid(self, uuids, brain_id=None, **kwargs):
        return []

    def get_conflict_scan_cursor(self, brain_id):
        return self.cursors.get(brain_id)

    def set_conflict_scan_cursor(self, brain_id, cursor):
        if not cursor:
            self.cursors.pop(brain_id, None)
        else:
            self.cursors[brain_id] = dict(cursor)


class FakeGraph:
    def __init__(self, triples):
        self.triples = triples
        self.calls = []

    def get_triples_by_uuid(self, uuids, brain_id=None, **kwargs):
        wanted = {str(item) for item in uuids or []}
        found = []
        for triple in self.triples:
            pred = triple[1]
            uid = str(getattr(pred, "uuid", "") or "")
            if uid in wanted:
                found.append(triple)
        return found

    def list_validity_relationships(
        self,
        brain_id,
        skip=0,
        limit=100,
        since=None,
        until=None,
        subject_uuids=None,
        relationship_uuids=None,
        after_subject_uuid=None,
        after_predicate=None,
        after_relationship_uuid=None,
    ):
        from src.core.saving.conflicts import (
            _sort_validity_triples,
            _triples_after_cursor,
        )

        self.calls.append(
            {
                "brain_id": brain_id,
                "skip": skip,
                "limit": limit,
                "subject_uuids": list(subject_uuids) if subject_uuids else None,
                "relationship_uuids": (
                    list(relationship_uuids) if relationship_uuids else None
                ),
                "after_subject_uuid": after_subject_uuid,
                "after_predicate": after_predicate,
                "after_relationship_uuid": after_relationship_uuid,
            }
        )
        items = self.triples
        if subject_uuids:
            allowed = set(subject_uuids)
            items = [item for item in items if item[0].uuid in allowed]
        if relationship_uuids:
            allowed_rels = set(relationship_uuids)
            items = [item for item in items if item[1].uuid in allowed_rels]
        ordered = _triples_after_cursor(
            _sort_validity_triples(items),
            after_subject_uuid=after_subject_uuid,
            after_predicate=after_predicate,
            after_relationship_uuid=after_relationship_uuid,
        )
        return ordered[skip : skip + limit], len(items)


class SearchOnlyGraph:
    def __init__(self, triples):
        self.triples = triples

    def search_relationships(self, brain_id, limit=100, skip=0):
        class Item:
            def __init__(self, triple):
                self.subject, self.predicate, self.object = triple

        class Result:
            def __init__(self, results, total=None):
                self.results = results
                if total is not None:
                    self.total = total

        items = [Item(item) for item in self.triples]
        kwargs = {}
        if getattr(self, "include_total", True):
            kwargs["total"] = len(items)
        return Result(items[skip : skip + limit], **kwargs)


class QuantityAndMatchTests(unittest.TestCase):
    def test_duration_normalization(self):
        self.assertEqual(parse_quantity("30 days")[0], 30)
        self.assertEqual(parse_quantity("two weeks")[0], 14)
        self.assertTrue(values_equivalent("30 days", "one month"))
        self.assertFalse(values_equivalent("30 days", "14 days"))
        self.assertEqual(parse_quantity("12,400")[0], 12400)
        self.assertEqual(parse_quantity("price 14,900")[0], 14900)
        self.assertFalse(values_equivalent("Founded in 2011", "Founded in 2013"))
        self.assertFalse(values_equivalent("price 12,400", "price 14,900"))
        self.assertTrue(values_equivalent("Founded in 2011", "Founded in 2011"))

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

    def test_antonym_and_negation_boundaries_and_direct_negation(self):
        # Direct negation of the same claim is a deterministic polarity conflict.
        self.assertTrue(antonym_or_negation("active", "not active"))
        self.assertTrue(antonym_or_negation("restocking fee", "no restocking fee"))
        self.assertTrue(antonym_or_negation("refundable", "non-refundable"))
        kind, _ = classify_disagreement(
            {"value": "active", "polarity": "neutral"},
            {"value": "not active", "polarity": "neutral"},
        )
        self.assertEqual(kind, "polarity_conflict")

        # Agreeing negated phrases, unrelated substrings, and unrelated terms
        # with one negated side must not trigger deterministic polarity conflicts.
        self.assertFalse(
            antonym_or_negation("non-refundable deposit", "non-refundable fee")
        )
        self.assertFalse(antonym_or_negation("non-refundable", "nonrefundable"))
        self.assertFalse(antonym_or_negation("yesterday", "north"))
        self.assertFalse(antonym_or_negation("untrue", "false"))
        self.assertFalse(antonym_or_negation("green", "no fee"))
        for left_val, right_val in (
            ("non-refundable deposit", "non-refundable fee"),
            ("yesterday", "north"),
            ("green", "no fee"),
        ):
            kind, _ = classify_disagreement(
                {"value": left_val, "polarity": "neutral"},
                {"value": right_val, "polarity": "neutral"},
            )
            self.assertNotEqual(kind, "polarity_conflict")
            snaps = [
                snapshot_from_triple(
                    _triple(
                        _node("p", "Policy"),
                        _pred(
                            "a",
                            "HAS_POLICY",
                            {"valid_from": "2024-01-01T00:00:00Z", "source_key": "s1"},
                        ),
                        _node("v1", left_val),
                    )
                ),
                snapshot_from_triple(
                    _triple(
                        _node("p", "Policy"),
                        _pred(
                            "b",
                            "HAS_POLICY",
                            {"valid_from": "2024-01-01T00:00:00Z", "source_key": "s2"},
                        ),
                        _node("v2", right_val),
                    )
                ),
            ]
            definite, _ = find_conflict_candidates(snaps)
            self.assertEqual(definite, [])

        neg_snaps = [
            snapshot_from_triple(
                _triple(
                    _node("p", "Policy"),
                    _pred(
                        "a",
                        "HAS_STATUS",
                        {"valid_from": "2024-01-01T00:00:00Z", "source_key": "s1"},
                    ),
                    _node("v1", "active"),
                )
            ),
            snapshot_from_triple(
                _triple(
                    _node("p", "Policy"),
                    _pred(
                        "b",
                        "HAS_STATUS",
                        {"valid_from": "2024-01-01T00:00:00Z", "source_key": "s2"},
                    ),
                    _node("v2", "not active"),
                )
            ),
        ]
        definite, semantic = find_conflict_candidates(neg_snaps)
        self.assertEqual(len(definite), 1)
        self.assertEqual(definite[0][2], "polarity_conflict")
        self.assertEqual(semantic, [])


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
        self.assertEqual(found.llm_pairs_attempted, 1)
        self.assertEqual(found.llm_pairs_judged, 1)
        self.assertEqual(found.llm_pairs_failed, 0)

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
            "reopen_conflict",
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
        triples = [
            _triple(
                _node(f"policy-{i}", f"Policy{i}"),
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
        self.assertEqual(found.llm_pairs_attempted, 1)
        self.assertEqual(found.llm_pairs_judged, 0)
        self.assertEqual(found.llm_pairs_failed, 1)
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
        self.assertEqual(found.llm_pairs_attempted, 3)
        self.assertEqual(found.llm_pairs_judged, 0)
        self.assertEqual(found.llm_pairs_failed, 3)
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
        extra = [
            _triple(
                _node("q", "Other"),
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
                _node("q", "Other"),
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
        truncated = detect_conflicts(
            store, FakeGraph(live + extra), "brain", max_relationships=2
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

    def test_coerce_datetime_falls_back_to_isoformat(self):
        from datetime import datetime

        class Temporal:
            def to_native(self):
                raise RuntimeError("native unavailable")

            def isoformat(self):
                return "2024-03-04T05:06:07+00:00"

        parsed = _coerce_datetime(Temporal())
        self.assertIsInstance(parsed, datetime)
        self.assertEqual(parsed.date().isoformat(), "2024-03-04")

    def test_time_bearing_deadlines_compare_full_instant(self):
        self.assertFalse(
            values_equivalent("2024-01-01T09:00:00Z", "2024-01-01T17:00:00Z")
        )
        self.assertTrue(
            values_equivalent("2024-01-01T09:00:00Z", "2024-01-01T09:00:00+00:00")
        )
        self.assertFalse(values_equivalent("20240101T090000", "20240101T170000"))
        self.assertTrue(values_equivalent("20240101T090000", "20240101T090000"))
        self.assertFalse(values_equivalent("2024-01-01T090000", "2024-01-01T170000"))
        self.assertTrue(values_equivalent("2024-01-01T090000", "2024-01-01T090000"))

    def test_empty_scoped_scan_does_not_stale_other_conflicts(self):
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
        detect_conflicts(
            store,
            FakeGraph(live),
            "brain",
            subject_uuids=["ghost-subject"],
        )
        self.assertEqual(store.rows[conflict_id].status, "open")

    def test_deprecated_side_is_marked_stale(self):
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
        deprecated_b = Predicate(
            uuid="b",
            name="HAS_WINDOW",
            description="",
            properties={
                "valid_from": "2024-01-01T00:00:00Z",
                "source_key": "s2",
            },
            deprecated=True,
        )
        remaining = [live[0], _triple(_node("p", "Policy"), deprecated_b, _node("y", "30 days"))]
        again = detect_conflicts(store, FakeGraph(remaining), "brain")
        self.assertEqual(store.rows[conflict_id].status, "resolved")
        self.assertEqual(store.rows[conflict_id].resolution.method, "stale")
        self.assertIn("side_removed", store.rows[conflict_id].resolution.notes)
        self.assertEqual(again.stale_resolved, 1)

    def test_stale_resolution_reopens_when_contradiction_returns(self):
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
        compatible = [
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
        detect_conflicts(store, FakeGraph(compatible), "brain")
        self.assertEqual(store.rows[conflict_id].status, "resolved")
        again = detect_conflicts(store, FakeGraph(live), "brain")
        self.assertEqual(store.rows[conflict_id].status, "open")
        self.assertEqual(again[0].id, conflict_id)

    def test_semantic_conflict_stays_open_without_verdict(self):
        store = MemoryConflictStore()
        triples = [
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
        client = MagicMock()
        client.generate_json.return_value = {
            "results": [
                {
                    "index": 0,
                    "contradicts": True,
                    "confidence": 0.9,
                    "type": "semantic_contradiction",
                }
            ]
        }
        found = detect_conflicts(
            store, FakeGraph(triples), "brain", use_llm=True, llm_client=client
        )
        self.assertEqual(len(found), 1)
        conflict_id = found[0].id
        again = detect_conflicts(store, FakeGraph(triples), "brain", use_llm=False)
        self.assertEqual(store.rows[conflict_id].status, "open")
        self.assertEqual(again.stale_resolved, 0)

    def test_stale_write_does_not_overwrite_manual_resolution(self):
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
        store.rows[conflict_id].status = "resolved"
        store.rows[conflict_id].resolution.method = "manual"
        detect_conflicts(store, FakeGraph([live[0]]), "brain")
        self.assertEqual(store.rows[conflict_id].status, "resolved")
        self.assertEqual(store.rows[conflict_id].resolution.method, "manual")

    def test_scan_pages_past_default_page_size(self):
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
            for i in range(250)
        ]
        graph = FakeGraph(triples)
        found = detect_conflicts(MemoryConflictStore(), graph, "brain")
        self.assertGreaterEqual(len(graph.calls), 2)
        self.assertTrue(
            any(call.get("after_relationship_uuid") for call in graph.calls)
        )
        self.assertTrue(all(call["skip"] == 0 for call in graph.calls))
        self.assertEqual(found.scanned_relationships, 250)
        pairs = {tuple(sorted(item.fact_ids())) for item in found}
        self.assertIn(("f0", "f249"), pairs)

    def test_relationship_filter_is_applied_when_backend_lacks_kwarg(self):
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
        ]

        class LegacyGraph:
            def list_validity_relationships(
                self, brain_id, skip=0, limit=100, subject_uuids=None
            ):
                items = triples
                if subject_uuids:
                    allowed = set(subject_uuids)
                    items = [item for item in items if item[0].uuid in allowed]
                return items[skip : skip + limit], len(items)

        found = detect_conflicts(
            MemoryConflictStore(),
            LegacyGraph(),
            "brain",
            fact_ids=["a"],
        )
        pairs = {tuple(sorted(item.fact_ids())) for item in found}
        self.assertEqual(pairs, {("a", "b")})

    def _semantic_policy_triples(self, extra=()):
        policy = _node("p", "Policy")
        triples = [
            _triple(
                policy,
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
                policy,
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
        triples.extend(extra)
        return triples

    def _llm_client(self, contradicts=True):
        client = MagicMock()
        client.generate_json.return_value = {
            "results": [
                {
                    "index": 0,
                    "contradicts": contradicts,
                    "confidence": 0.9,
                    "type": "semantic_contradiction",
                }
            ]
        }
        return client

    def test_llm_conflict_stays_open_when_llm_off(self):
        store = MemoryConflictStore()
        triples = self._semantic_policy_triples()
        found = detect_conflicts(
            store, FakeGraph(triples), "brain", use_llm=True, llm_client=self._llm_client()
        )
        conflict_id = found[0].id
        again = detect_conflicts(store, FakeGraph(triples), "brain", use_llm=False)
        self.assertEqual(store.rows[conflict_id].status, "open")
        self.assertEqual(again.stale_resolved, 0)

    def test_llm_conflict_stays_open_when_budget_truncated(self):
        store = MemoryConflictStore()
        triples = self._semantic_policy_triples()
        found = detect_conflicts(
            store, FakeGraph(triples), "brain", use_llm=True, llm_client=self._llm_client()
        )
        conflict_id = found[0].id
        extra = _triple(
            _node("p", "Policy"),
            _pred(
                "c",
                "HAS_POLICY",
                {
                    "valid_from": "2024-01-01T00:00:00Z",
                    "source_key": "s3",
                },
            ),
            _node("exchange", "exchange only"),
        )
        client = MagicMock()
        client.generate_json.side_effect = RuntimeError("batch failed")
        with patch.dict("os.environ", {"CONFLICT_DETECT_LLM_MAX_PAIRS": "1"}):
            again = detect_conflicts(
                store,
                FakeGraph(triples + [extra]),
                "brain",
                use_llm=True,
                llm_client=client,
            )
        self.assertTrue(again.llm_truncated)
        self.assertEqual(store.rows[conflict_id].status, "open")
        self.assertEqual(again.stale_resolved, 0)

    def test_llm_conflict_stays_open_when_batch_fails(self):
        store = MemoryConflictStore()
        triples = self._semantic_policy_triples()
        found = detect_conflicts(
            store, FakeGraph(triples), "brain", use_llm=True, llm_client=self._llm_client()
        )
        conflict_id = found[0].id
        client = MagicMock()
        client.generate_json.side_effect = RuntimeError("judge down")
        again = detect_conflicts(
            store, FakeGraph(triples), "brain", use_llm=True, llm_client=client
        )
        self.assertEqual(again.llm_pairs_attempted, 1)
        self.assertEqual(again.llm_pairs_judged, 0)
        self.assertEqual(again.llm_pairs_failed, 1)
        self.assertEqual(store.rows[conflict_id].status, "open")
        self.assertEqual(again.stale_resolved, 0)

    def test_llm_compatible_verdict_marks_stale(self):
        store = MemoryConflictStore()
        triples = self._semantic_policy_triples()
        found = detect_conflicts(
            store, FakeGraph(triples), "brain", use_llm=True, llm_client=self._llm_client()
        )
        conflict_id = found[0].id
        again = detect_conflicts(
            store,
            FakeGraph(triples),
            "brain",
            use_llm=True,
            llm_client=self._llm_client(contradicts=False),
        )
        self.assertEqual(store.rows[conflict_id].status, "resolved")
        self.assertEqual(store.rows[conflict_id].resolution.method, "stale")
        self.assertIn("llm_compatible", store.rows[conflict_id].resolution.notes)
        self.assertEqual(again.stale_resolved, 1)

    def test_upsert_does_not_overwrite_concurrent_manual_resolution(self):
        from src.constants.conflicts import ConflictResolution
        from src.core.saving.conflicts import upsert_conflict

        store = MemoryConflictStore()
        triples = [
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
        found = detect_conflicts(store, FakeGraph(triples), "brain")
        conflict = found[0]
        conflict.status = "resolved"
        conflict.resolution = ConflictResolution(method="stale", notes="side_removed")
        store.save_conflict(conflict, "brain")

        class RacingStore(MemoryConflictStore):
            def update_conflict(
                self, item, brain_id, if_status=None, if_resolution_method=None
            ):
                existing = self.rows[item.id]
                existing.status = "resolved"
                existing.resolution = ConflictResolution(
                    method="manual", notes="operator"
                )
                return super().update_conflict(
                    item,
                    brain_id,
                    if_status=if_status,
                    if_resolution_method=if_resolution_method,
                )

        racing = RacingStore()
        racing.rows = store.rows
        reopened = found[0].model_copy(deep=True)
        reopened.status = "open"
        reopened.resolution = ConflictResolution()
        saved = upsert_conflict(racing, "brain", reopened)
        self.assertEqual(saved.status, "resolved")
        self.assertEqual(saved.resolution.method, "manual")
        self.assertEqual(racing.rows[conflict.id].resolution.method, "manual")

    def test_llm_conflict_still_closes_when_side_removed(self):
        store = MemoryConflictStore()
        triples = self._semantic_policy_triples()
        found = detect_conflicts(
            store, FakeGraph(triples), "brain", use_llm=True, llm_client=self._llm_client()
        )
        conflict_id = found[0].id
        again = detect_conflicts(
            store, FakeGraph(triples[:1]), "brain", use_llm=False
        )
        self.assertEqual(store.rows[conflict_id].status, "resolved")
        self.assertIn("side_removed", store.rows[conflict_id].resolution.notes)
        self.assertEqual(again.stale_resolved, 1)

    def test_stale_when_validity_windows_no_longer_overlap(self):
        store = MemoryConflictStore()
        live = [
            _triple(
                _node("p", "Policy"),
                _pred(
                    "a",
                    "HAS_WINDOW",
                    {
                        "valid_from": "2024-01-01T00:00:00Z",
                        "valid_to": "2024-06-01T00:00:00Z",
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
                        "valid_from": "2024-03-01T00:00:00Z",
                        "source_key": "s2",
                    },
                ),
                _node("y", "30 days"),
            ),
        ]
        found = detect_conflicts(store, FakeGraph(live), "brain")
        conflict_id = found[0].id
        split = [
            live[0],
            _triple(
                _node("p", "Policy"),
                _pred(
                    "b",
                    "HAS_WINDOW",
                    {
                        "valid_from": "2024-07-01T00:00:00Z",
                        "source_key": "s2",
                    },
                ),
                _node("y", "30 days"),
            ),
        ]
        again = detect_conflicts(store, FakeGraph(split), "brain")
        self.assertEqual(store.rows[conflict_id].status, "resolved")
        self.assertIn("validity_no_overlap", store.rows[conflict_id].resolution.notes)
        self.assertEqual(again.stale_resolved, 1)

    def test_stale_when_side_is_superseded(self):
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
        superseded = [
            _triple(
                _node("p", "Policy"),
                _pred(
                    "a",
                    "HAS_WINDOW",
                    {
                        "valid_from": "2024-01-01T00:00:00Z",
                        "source_key": "s1",
                        "superseded_by": "b",
                    },
                ),
                _node("x", "14 days"),
            ),
            live[1],
        ]
        again = detect_conflicts(store, FakeGraph(superseded), "brain")
        self.assertEqual(store.rows[conflict_id].status, "resolved")
        self.assertIn("side_superseded", store.rows[conflict_id].resolution.notes)
        self.assertEqual(again.stale_resolved, 1)

    def test_config_max_relationships_caps_full_scan(self):
        triples = [
            _triple(
                _node(f"policy-{i}", f"Policy{i}"),
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
        with patch.dict("os.environ", {"CONFLICT_DETECT_MAX_RELATIONSHIPS": "2"}):
            capped = detect_conflicts(MemoryConflictStore(), FakeGraph(triples), "brain")
        self.assertTrue(capped.truncated)
        self.assertEqual(capped.scanned_relationships, 2)
        with patch.dict("os.environ", {"CONFLICT_DETECT_MAX_RELATIONSHIPS": "0"}):
            unlimited = detect_conflicts(
                MemoryConflictStore(), FakeGraph(triples), "brain"
            )
        self.assertFalse(unlimited.truncated)
        self.assertEqual(unlimited.scanned_relationships, 5)
        self.assertEqual(conflict_detect_max_relationships(), 10000)


def _window_triple(subject, fact_id, value, source=None):
    return _triple(
        _node(subject, subject),
        _pred(
            fact_id,
            "HAS_WINDOW",
            {
                "valid_from": "2024-01-01T00:00:00Z",
                "source_key": source or fact_id,
            },
        ),
        _node(fact_id + "-obj", value),
    )


class ConflictScanCursorTests(unittest.TestCase):
    def test_repeated_capped_scans_cover_every_relationship(self):
        triples = [
            _window_triple(f"s{i}", f"f{i}", f"{i + 1} days") for i in range(5)
        ]
        store = MemoryConflictStore()
        graph = FakeGraph(triples)
        seen = set()
        for _ in range(4):
            result = detect_conflicts(store, graph, "brain", max_relationships=2)
            snapshots = [
                snapshot_from_triple(item) for item in graph.triples
            ]
            scanned_ids = {
                item["fact_id"]
                for item in snapshots
                if claim_key_for(item["subject_uuid"], item["predicate"])
                in set(result.scanned_claim_keys)
            }
            seen.update(scanned_ids)
        self.assertEqual(seen, {f"f{i}" for i in range(5)})

    def test_group_at_the_cut_is_never_split(self):
        triples = [
            _window_triple("a", "a1", "14 days"),
            _window_triple("a", "a2", "30 days"),
            _window_triple("b", "b1", "7 days"),
            _window_triple("b", "b2", "14 days"),
            _window_triple("b", "b3", "21 days"),
        ]
        store = MemoryConflictStore()
        first = detect_conflicts(
            store, FakeGraph(triples), "brain", max_relationships=4
        )
        self.assertTrue(first.truncated)
        self.assertEqual(first.scanned_relationships, 2)
        self.assertEqual(
            set(first.scanned_claim_keys), {claim_key_for("a", "HAS_WINDOW")}
        )
        second = detect_conflicts(
            store, FakeGraph(triples), "brain", max_relationships=4
        )
        self.assertEqual(second.scanned_relationships, 3)
        self.assertEqual(
            set(second.scanned_claim_keys), {claim_key_for("b", "HAS_WINDOW")}
        )

    def test_cursor_wraparound_restarts_at_first_group(self):
        triples = [
            _window_triple("a", "a1", "14 days"),
            _window_triple("b", "b1", "30 days"),
            _window_triple("c", "c1", "7 days"),
        ]
        store = MemoryConflictStore()
        graph = FakeGraph(triples)
        first = detect_conflicts(store, graph, "brain", max_relationships=2)
        self.assertTrue(first.truncated)
        self.assertEqual(store.cursors["brain"]["after_subject"], "b")
        second = detect_conflicts(store, graph, "brain", max_relationships=2)
        self.assertTrue(second.truncated)
        scanned = set(second.scanned_claim_keys)
        self.assertIn(claim_key_for("c", "HAS_WINDOW"), scanned)
        self.assertIn(claim_key_for("a", "HAS_WINDOW"), scanned)
        self.assertNotIn(claim_key_for("b", "HAS_WINDOW"), scanned)
        third = detect_conflicts(store, graph, "brain", max_relationships=2)
        scanned = set(third.scanned_claim_keys)
        self.assertIn(claim_key_for("b", "HAS_WINDOW"), scanned)

    def test_stale_close_only_applies_to_scanned_groups(self):
        live = [
            _window_triple("a", "a1", "14 days"),
            _window_triple("a", "a2", "30 days"),
            _window_triple("b", "b1", "14 days"),
            _window_triple("b", "b2", "30 days"),
        ]
        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(live), "brain")
        self.assertEqual(len(found), 2)
        by_subject = {item.subject.uuid: item for item in found}
        reconciled = [
            live[0],
            live[1],
            _window_triple("b", "b1", "14 days"),
            _window_triple("b", "b2", "14 days"),
        ]
        truncated = detect_conflicts(
            store, FakeGraph(reconciled), "brain", max_relationships=2
        )
        self.assertTrue(truncated.truncated)
        self.assertEqual(store.rows[by_subject["a"].id].status, "open")
        self.assertEqual(store.rows[by_subject["b"].id].status, "open")
        self.assertEqual(truncated.stale_resolved, 0)
        again = detect_conflicts(
            store, FakeGraph(reconciled), "brain", max_relationships=2
        )
        self.assertEqual(store.rows[by_subject["a"].id].status, "open")
        self.assertEqual(store.rows[by_subject["b"].id].status, "resolved")
        self.assertEqual(again.stale_resolved, 1)

    def test_truncated_scan_can_stale_close_within_scanned_group(self):
        live = [
            _window_triple("a", "a1", "14 days"),
            _window_triple("a", "a2", "30 days"),
            _window_triple("b", "b1", "14 days"),
            _window_triple("b", "b2", "30 days"),
        ]
        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(live), "brain")
        by_subject = {item.subject.uuid: item for item in found}
        remaining = [live[0], live[2], live[3]]
        truncated = detect_conflicts(
            store, FakeGraph(remaining), "brain", max_relationships=1
        )
        self.assertTrue(truncated.truncated)
        self.assertEqual(store.rows[by_subject["a"].id].status, "resolved")
        self.assertEqual(store.rows[by_subject["b"].id].status, "open")
        self.assertEqual(truncated.stale_resolved, 1)

    def test_search_relationships_fallback_respects_cap(self):
        triples = [
            _window_triple(f"s{i}", f"f{i}", f"{i + 1} days") for i in range(5)
        ]
        result = detect_conflicts(
            MemoryConflictStore(),
            SearchOnlyGraph(triples),
            "brain",
            max_relationships=2,
        )
        self.assertTrue(result.truncated)
        self.assertEqual(result.scanned_relationships, 2)
        self.assertEqual(result.total_relationships, 5)

    def test_search_fallback_keeps_interleaved_group_together(self):
        triples = (
            [_window_triple("a", "a1", "14 days")]
            + [
                _window_triple("b", f"b{i}", f"{i + 1} days")
                for i in range(199)
            ]
            + [_window_triple("a", "a2", "30 days")]
        )
        result = detect_conflicts(
            MemoryConflictStore(),
            SearchOnlyGraph(triples),
            "brain",
            max_relationships=2,
        )
        self.assertEqual(result.scanned_relationships, 2)
        self.assertEqual(
            set(result.scanned_claim_keys),
            {claim_key_for("a", "HAS_WINDOW")},
        )

    def test_search_fallback_treats_empty_results_as_end(self):
        triples = [
            _window_triple(f"s{i}", f"f{i}", f"{i + 1} days") for i in range(200)
        ]
        graph = SearchOnlyGraph(triples)
        graph.include_total = False
        result = detect_conflicts(MemoryConflictStore(), graph, "brain")
        self.assertEqual(result.scanned_relationships, 200)
        self.assertFalse(result.truncated)

    def test_graph_adapter_fallback_finds_uuid_past_first_page(self):
        from src.adapters.graph import GraphAdapter

        triples = [
            _window_triple(f"s{i}", f"f{i}", f"{i + 1} days") for i in range(201)
        ]
        adapter = GraphAdapter()
        adapter.graph = SearchOnlyGraph(triples)
        found, _total = adapter.list_validity_relationships(
            "brain", skip=0, limit=2, relationship_uuids=["f200"]
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][1].uuid, "f200")

    def test_search_fallback_stale_lookup_honors_uuid_filter(self):
        live = [
            _window_triple("a", "a1", "14 days"),
            _window_triple("a", "a2", "30 days"),
            _window_triple("b", "b1", "14 days"),
            _window_triple("b", "b2", "30 days"),
            _window_triple("c", "c1", "7 days"),
            _window_triple("c", "c2", "21 days"),
        ]
        store = MemoryConflictStore()
        found = detect_conflicts(store, SearchOnlyGraph(live), "brain")
        by_subject = {item.subject.uuid: item for item in found}
        truncated = detect_conflicts(
            store, SearchOnlyGraph(live), "brain", max_relationships=2
        )
        self.assertTrue(truncated.truncated)
        self.assertEqual(store.rows[by_subject["a"].id].status, "open")
        self.assertEqual(store.rows[by_subject["b"].id].status, "open")
        self.assertEqual(store.rows[by_subject["c"].id].status, "open")
        self.assertEqual(truncated.stale_resolved, 0)

    def test_legacy_list_validity_without_cursor_still_pages(self):
        from src.adapters.graph import GraphAdapter

        triples = [
            _window_triple(f"s{i}", f"f{i}", f"{i + 1} days") for i in range(250)
        ]

        class SkipOnlyGraph:
            def list_validity_relationships(self, brain_id, skip=0, limit=100):
                return triples[skip : skip + limit], len(triples)

        adapter = GraphAdapter()
        adapter.graph = SkipOnlyGraph()
        found = detect_conflicts(MemoryConflictStore(), adapter, "brain")
        self.assertEqual(found.scanned_relationships, 250)
        self.assertFalse(found.truncated)

    def test_truncated_scan_stale_closes_removed_group(self):
        live = [
            _window_triple("a", "a1", "14 days"),
            _window_triple("a", "a2", "30 days"),
            _window_triple("b", "b1", "14 days"),
            _window_triple("b", "b2", "30 days"),
            _window_triple("c", "c1", "7 days"),
            _window_triple("c", "c2", "21 days"),
        ]
        store = MemoryConflictStore()
        found = detect_conflicts(store, FakeGraph(live), "brain")
        by_subject = {item.subject.uuid: item for item in found}
        remaining = live[2:]
        truncated = detect_conflicts(
            store, FakeGraph(remaining), "brain", max_relationships=2
        )
        self.assertTrue(truncated.truncated)
        self.assertEqual(store.rows[by_subject["a"].id].status, "resolved")
        self.assertEqual(store.rows[by_subject["b"].id].status, "open")
        self.assertEqual(store.rows[by_subject["c"].id].status, "open")

    def test_cypher_predicate_key_matches_normalize_predicate(self):
        self.assertEqual(cypher_normalize_predicate(None), "")
        self.assertEqual(normalize_predicate(None), "")
        for raw, expected in PREDICATE_NORMALIZER_CASES:
            self.assertEqual(normalize_predicate(raw), expected, raw)
            self.assertEqual(
                cypher_normalize_predicate(raw),
                normalize_predicate(raw),
                raw,
            )

    def test_punctuation_predicate_aliases_stay_in_one_group(self):
        def named(subject, fact_id, predicate, value):
            return _triple(
                _node(subject, subject),
                _pred(
                    fact_id,
                    predicate,
                    {
                        "valid_from": "2024-01-01T00:00:00Z",
                        "source_key": fact_id,
                    },
                ),
                _node(fact_id + "-obj", value),
            )

        triples = [
            named("s", "a1", "HAS-A", "14 days"),
            named("s", "b1", "HAS_B", "7 days"),
            named("s", "a2", "HAS A", "30 days"),
            named("s", "a3", "HAS.A", "21 days"),
            named("s", "a4", "has_a", "10 days"),
            named("s", "b2", "HAS_B", "3 days"),
        ]
        store = MemoryConflictStore()
        graph = FakeGraph(triples)

        def fact_ids_for(result):
            wanted = set(result.scanned_claim_keys)
            ids = set()
            for item in triples:
                snap = snapshot_from_triple(item)
                if claim_key_for(snap["subject_uuid"], snap["predicate"]) in wanted:
                    ids.add(snap["fact_id"])
            return ids

        first = detect_conflicts(
            store, graph, "brain", max_relationships=4
        )
        self.assertEqual(first.scanned_relationships, 4)
        self.assertEqual(
            set(first.scanned_claim_keys), {claim_key_for("s", "HAS_A")}
        )
        self.assertTrue(first.truncated)
        first_ids = fact_ids_for(first)
        self.assertEqual(first_ids, {"a1", "a2", "a3", "a4"})
        second = detect_conflicts(
            store, graph, "brain", max_relationships=4
        )
        second_ids = fact_ids_for(second)
        self.assertEqual(second_ids, {"b1", "b2"})
        self.assertEqual(first_ids & second_ids, set())
        self.assertEqual(first_ids | second_ids, {item[1].uuid for item in triples})

    def test_normalized_predicate_aliases_stay_in_one_group(self):
        def named(subject, fact_id, predicate, value):
            return _triple(
                _node(subject, subject),
                _pred(
                    fact_id,
                    predicate,
                    {
                        "valid_from": "2024-01-01T00:00:00Z",
                        "source_key": fact_id,
                    },
                ),
                _node(fact_id + "-obj", value),
            )

        triples = [
            named("s", "a1", "HAS_A", "14 days"),
            named("s", "b1", "HAS_B", "7 days"),
            named("s", "a2", "HAS__A", "30 days"),
        ]
        store = MemoryConflictStore()
        result = detect_conflicts(
            store,
            FakeGraph(triples),
            "brain",
            max_relationships=2,
        )
        self.assertEqual(result.scanned_relationships, 2)
        self.assertEqual(
            set(result.scanned_claim_keys), {claim_key_for("s", "HAS_A")}
        )
        self.assertEqual(result.scan_cursor["after_predicate"], "HAS_A")
        self.assertEqual(store.cursors["brain"]["after_predicate"], "HAS_A")

    def test_stored_raw_alias_cursor_is_normalized_on_read(self):
        from src.core.saving.conflicts import _read_scan_cursor

        store = MemoryConflictStore()
        store.cursors["brain"] = {
            "after_subject": "s",
            "after_predicate": "HAS__A",
        }
        self.assertEqual(
            _read_scan_cursor(store, "brain"),
            {"after_subject": "s", "after_predicate": "HAS_A"},
        )

    def test_cursor_write_failure_is_raised(self):
        class BrokenStore(MemoryConflictStore):
            def set_conflict_scan_cursor(self, brain_id, cursor):
                raise RuntimeError("disk full")

        triples = [
            _window_triple(f"s{i}", f"f{i}", f"{i + 1} days") for i in range(3)
        ]
        with self.assertRaises(RuntimeError):
            detect_conflicts(
                BrokenStore(),
                FakeGraph(triples),
                "brain",
                max_relationships=1,
            )

    def test_rest_scan_request_cannot_override_cap(self):
        from src.services.api.constants.requests import ConflictScanRequest

        self.assertNotIn("max_relationships", ConflictScanRequest.model_fields)
        self.assertEqual(
            set(ConflictScanRequest.model_fields), {"use_llm"}
        )


class ConfigDefaultTests(unittest.TestCase):
    def test_config_flags_default_off(self):
        from src.config import config

        self.assertFalse(config.conflict_detect_on_ingest)
        self.assertFalse(config.conflict_detect_llm)
        self.assertEqual(config.conflict_detect_max_relationships, 10000)


class NumericFactAndProvenanceTests(unittest.TestCase):
    def test_snapshot_keeps_amount_on_value_and_detects_year_conflict(self):
        org = _node("org-1", "Halvorsen Dynamics", ["ORGANIZATION"])
        founding = _node("evt-founding", "Halvorsen Dynamics founding", ["EVENT"])
        left = snapshot_from_triple(
            _triple(
                org,
                _pred(
                    "f-2011",
                    "FOUNDED_IN",
                    {
                        "source_key": "text:profile.txt:aaaaaaaaaaaaaaaa",
                        "source_span": "Founded in 2011",
                        "source_chunk_ids": ["chunk-a"],
                    },
                    description="Halvorsen Dynamics was founded",
                    amount=2011,
                ),
                founding,
            )
        )
        right = snapshot_from_triple(
            _triple(
                org,
                _pred(
                    "f-2013",
                    "FOUNDED_IN",
                    {
                        "source_key": "text:counter.txt:bbbbbbbbbbbbbbbb",
                        "source_span": "Founded in 2013",
                    },
                    description="Halvorsen Dynamics was founded",
                    amount=2013,
                ),
                founding,
            )
        )
        self.assertEqual(left["amount"], 2011.0)
        self.assertIn("2011", left["value"])
        self.assertEqual(left["source_key"], "text:profile.txt:aaaaaaaaaaaaaaaa")
        self.assertEqual(left["source_text"], "Founded in 2011")
        kind, _ = classify_disagreement(left, right)
        self.assertEqual(kind, "numeric_mismatch")

    def test_price_literals_with_commas_conflict(self):
        kind, _ = classify_disagreement(
            {"value": "price 12,400", "polarity": "neutral"},
            {"value": "price 14,900", "polarity": "neutral"},
        )
        self.assertEqual(kind, "numeric_mismatch")

    def test_present_conflict_fills_empty_subject_sides_and_provenance(self):
        org = _node("org-1", "Halvorsen Dynamics", ["ORGANIZATION"])
        founding = _node("evt-founding", "founding 2011", ["EVENT"])
        pred = _pred(
            "f-2011",
            "FOUNDED_IN",
            {
                "source_key": "text:profile.txt:aaaaaaaaaaaaaaaa",
                "source_span": "Founded in 2011",
                "source_timestamp": "2026-11-26T00:00:00Z",
            },
            description="Halvorsen Dynamics was founded",
            amount=2011,
        )
        graph = FakeGraph([_triple(org, pred, founding)])
        empty = FactConflict(
            id="c1",
            pair_key="pair",
            claim_key="claim",
            disagreement_type="numeric_mismatch",
            subject=ConflictEntity(),
            predicate="FOUNDED_IN",
            sides=[
                ConflictSide(
                    fact_id="f-2011",
                    subject=ConflictEntity(),
                    predicate="FOUNDED_IN",
                    object=ConflictEntity(),
                    value="",
                    source=ConflictSource(),
                ),
                ConflictSide(
                    fact_id="missing",
                    subject=ConflictEntity(uuid="org-1", name="Halvorsen Dynamics"),
                    predicate="FOUNDED_IN",
                    object=ConflictEntity(uuid="evt-b", name="founding 2013"),
                    value="2013",
                    amount=2013,
                    source=ConflictSource(
                        source_key="text:counter.txt:bbbbbbbbbbbbbbbb",
                        document="QARC-20261126 counter-brief",
                        text="Founded in 2013",
                        timestamp="2026-11-26T00:00:00Z",
                    ),
                ),
            ],
        )
        filled = present_conflict(empty, graph=graph, brain_id="brain")
        self.assertEqual(filled.subject.name, "Halvorsen Dynamics")
        self.assertEqual(len(filled.sides), 2)
        first = filled.sides[0]
        self.assertEqual(first.subject.name, "Halvorsen Dynamics")
        self.assertIn("2011", first.value)
        self.assertEqual(first.amount, 2011.0)
        self.assertEqual(first.source.source_key, "text:profile.txt:aaaaaaaaaaaaaaaa")
        self.assertEqual(first.source.text, "Founded in 2011")
        self.assertTrue(first.source.document)
        self.assertEqual(first.source.timestamp, "2026-11-26T00:00:00Z")
        second = filled.sides[1]
        self.assertEqual(second.source.source_key, "text:counter.txt:bbbbbbbbbbbbbbbb")
        self.assertIn("QARC-20261126", second.source.document)

    def test_source_filter_matches_document_substring(self):
        conflict = FactConflict(
            id="c1",
            pair_key="pair",
            claim_key="claim",
            disagreement_type="numeric_mismatch",
            subject=ConflictEntity(name="Halvorsen Dynamics"),
            predicate="FOUNDED_IN",
            sides=[
                ConflictSide(
                    fact_id="a",
                    subject=ConflictEntity(name="Halvorsen Dynamics"),
                    predicate="FOUNDED_IN",
                    object=ConflictEntity(name="2011"),
                    value="2011",
                    source=ConflictSource(
                        source_key="text:untitled:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        document="QARC-20261126 Halvorsen Dynamics company profile",
                    ),
                )
            ],
        )
        self.assertTrue(conflict_matches_source_filter(conflict, "QARC-20261126"))
        self.assertTrue(
            conflict_matches_source_filter(
                conflict, "text:untitled:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            )
        )
        self.assertTrue(conflict_matches_source_filter(conflict, "text:untitled"))
        self.assertFalse(conflict_matches_source_filter(conflict, "text:paste"))
        self.assertFalse(conflict_matches_source_filter(conflict, "no-such-source"))
        with_source_id = FactConflict(
            id="c-id",
            pair_key="pair-id",
            claim_key="claim-id",
            disagreement_type="numeric_mismatch",
            subject=ConflictEntity(name="Halvorsen Dynamics"),
            predicate="FOUNDED_IN",
            sides=[
                ConflictSide(
                    fact_id="a",
                    subject=ConflictEntity(name="Halvorsen Dynamics"),
                    predicate="FOUNDED_IN",
                    object=ConflictEntity(name="2011"),
                    value="2011",
                    source=ConflictSource(source_key="qa-profile"),
                )
            ],
        )
        self.assertTrue(conflict_matches_source_filter(with_source_id, "qa-profile"))

    def test_dismiss_and_reopen_record_actor_and_time(self):
        store = MemoryConflictStore()
        found = detect_conflicts(
            store, FakeGraph(DetectionEngineTests()._refund_triples()), "brain"
        )
        conflict = found[0]
        dismissed = dismiss_conflict(
            store, "brain", conflict.id, notes="noise", resolved_by="user:42"
        )
        self.assertEqual(dismissed.status, "dismissed")
        self.assertEqual(dismissed.resolution.resolved_by, "user:42")
        self.assertTrue(dismissed.resolution.resolved_at)
        reopened = reopen_conflict(
            store, "brain", conflict.id, notes="check again", resolved_by="user:42"
        )
        self.assertEqual(reopened.status, "open")
        self.assertEqual(reopened.resolution.resolved_by, "user:42")
        self.assertTrue(reopened.resolution.resolved_at)
        self.assertIsNone(reopened.resolution.method)
        listed, total = store.get_conflicts_list("brain", source="QARC-missing")
        self.assertEqual(total, 0)


class RestConflictActorAndReopenTests(unittest.TestCase):
    def _app(self, principal, scopes, store):
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient

        from src.services.api.errors import install_error_handlers
        from src.services.api.routes.retrieve import retrieve_router

        app = FastAPI()
        install_error_handlers(app)

        @app.middleware("http")
        async def inject_brain(request: Request, call_next):
            request.state.brain_id = "brain"
            request.state.principal = principal
            request.state.scopes = scopes
            return await call_next(request)

        app.include_router(retrieve_router)
        client = TestClient(app)
        return client

    def _seed(self):
        store = MemoryConflictStore()
        found = detect_conflicts(
            store, FakeGraph(DetectionEngineTests()._refund_triples()), "brain"
        )
        return store, found[0]

    def test_dismiss_uses_signed_usage_context_user(self):
        import base64
        import hashlib
        import hmac
        import json
        import os
        import time

        payload = {
            "jobId": "job-1",
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
            "userId": "99",
        }
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        signature = base64.urlsafe_b64encode(
            hmac.new(
                b"test-secret", f"v1.{encoded}".encode(), hashlib.sha256
            ).digest()
        ).decode().rstrip("=")
        token = f"v1.{encoded}.{signature}"

        store, conflict = self._seed()
        api = self._app("system", ["brainapi"], store)
        with patch(
            "src.services.api.controllers.conflicts.data_adapter", store
        ), patch.dict(os.environ, {"USAGE_CONTEXT_SECRET": "test-secret"}):
            dismissed = api.post(
                f"/retrieve/conflicts/{conflict.id}/dismiss",
                json={"notes": "dup"},
                headers={"X-Usage-Context": token},
            )
        self.assertEqual(dismissed.status_code, 200, dismissed.text)
        body = dismissed.json()["conflict"]
        self.assertEqual(body["status"], "dismissed")
        self.assertEqual(body["resolution"]["resolved_by"], "user:99")
        self.assertTrue(body["resolution"]["resolved_at"])

        with patch(
            "src.services.api.controllers.conflicts.data_adapter", store
        ):
            reopened = api.post(
                f"/retrieve/conflicts/{conflict.id}/reopen",
                json={"notes": "look again"},
            )
        self.assertEqual(reopened.status_code, 200, reopened.text)
        opened = reopened.json()["conflict"]
        self.assertEqual(opened["status"], "open")
        self.assertEqual(opened["resolution"]["resolved_by"], "system")
        self.assertTrue(opened["resolution"]["resolved_at"])
        self.assertGreaterEqual(len(opened["sides"]), 2)
        self.assertTrue(opened["subject"]["name"])


if __name__ == "__main__":
    unittest.main()


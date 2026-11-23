from __future__ import annotations

import unittest

from evals.adapters.base import QueryResult
from evals.evaluate import (
    _aggregate,
    _conflict_headline,
    _record,
    _score_one,
    compare_systems,
)
from evals.judge import _parse_bool_payload


class CompareSystemsTests(unittest.TestCase):
    def test_capability_pending_is_not_a_loss(self) -> None:
        result = compare_systems(
            {
                "hybrid_baseline": {
                    "metrics": {
                        "n": 150,
                        "accuracy": 0.9,
                        "wilson_lo": 0.84,
                        "wilson_hi": 0.94,
                    }
                },
                "brainapi": {
                    "capabilities": {"as_of": False, "conflict_pairs": False},
                    "metrics": {
                        "n": 150,
                        "accuracy": 0.2,
                        "wilson_lo": 0.14,
                        "wilson_hi": 0.27,
                    },
                },
            },
            use_case="versioned_rules",
        )
        self.assertEqual(result["verdict"], "capability_pending")
        self.assertEqual(result["missing_capability"], "as_of")

    def test_mismatched_question_ids_are_incomplete(self) -> None:
        records = [
            {"system": "hybrid_baseline", "question_id": "q1", "correct": True},
            {"system": "hybrid_baseline", "question_id": "q2", "correct": True},
            {"system": "brainapi", "question_id": "q1", "correct": True},
        ]
        result = compare_systems(
            {
                "hybrid_baseline": {
                    "metrics": {"n": 2, "accuracy": 1.0, "wilson_lo": 0.3, "wilson_hi": 1.0}
                },
                "brainapi": {
                    "capabilities": {},
                    "metrics": {"n": 1, "accuracy": 1.0, "wilson_lo": 0.2, "wilson_hi": 1.0},
                },
            },
            use_case="expert_knowhow",
            records=records,
        )
        self.assertEqual(result["verdict"], "incomplete")

    def test_errors_count_as_incorrect(self) -> None:
        rows = [
            {"correct": True},
            {"correct": False, "error": "timeout"},
        ]
        headline = _aggregate(rows)
        self.assertEqual(headline["n"], 2)
        self.assertEqual(headline["n_errors"], 1)
        self.assertEqual(headline["accuracy"], 0.5)

    def test_degraded_rows_excluded_from_accuracy(self) -> None:
        rows = [
            {"correct": True, "degraded": None},
            {"correct": False, "degraded": ["data.keyword_search"]},
        ]
        headline = _aggregate(rows)
        self.assertEqual(headline["n"], 2)
        self.assertEqual(headline["n_degraded"], 1)
        self.assertEqual(headline["degraded_rate"], 0.5)
        self.assertEqual(headline["accuracy"], 1.0)
        self.assertEqual(headline["strict_accuracy"], 0.5)
        self.assertTrue(headline["exceeded_max_degraded_rate"])
        included = _aggregate(rows, include_degraded=True)
        self.assertEqual(included["n_degraded"], 1)
        self.assertEqual(included["accuracy"], 0.5)
        self.assertEqual(included["strict_accuracy"], 0.5)
        under_cap = _aggregate(rows, max_degraded_rate=0.5)
        self.assertFalse(under_cap["exceeded_max_degraded_rate"])

    def test_strict_accuracy_counts_correct_degraded_as_wrong(self) -> None:
        rows = [
            {"correct": True, "degraded": None},
            {"correct": True, "degraded": ["data.keyword_search"]},
        ]
        headline = _aggregate(rows)
        self.assertEqual(headline["accuracy"], 1.0)
        self.assertEqual(headline["strict_accuracy"], 0.5)
        included = _aggregate(rows, include_degraded=True)
        self.assertEqual(included["accuracy"], 1.0)
        self.assertEqual(included["strict_accuracy"], 0.5)
        result = QueryResult(
            answer="ok",
            raw={"context": {"degraded": ["vector.search_data"]}},
        )
        recorded = _record("expert_knowhow", "brainapi", {"id": "q1"}, result, {"correct": True})
        self.assertEqual(recorded["degraded"], ["vector.search_data"])

    def test_unsupported_conflict_metric_is_not_zero(self) -> None:
        records = [
            {
                "notes": ["conflict_mapping_unsupported"],
                "conflict_pairs": [],
                "unsupported": True,
                "scoring": "conflict_pairs_unsupported",
                "correct": None,
            }
        ]
        headline = _conflict_headline(records, [("stmt-a", "stmt-b")])
        self.assertTrue(headline["unsupported"])
        self.assertIsNone(headline["f1"])
        agg = _aggregate(records)
        self.assertEqual(agg["n_unsupported"], 1)
        self.assertIsNone(agg["accuracy"])

    def test_score_one_marks_unmapped_conflicts_unsupported(self) -> None:
        result = QueryResult(
            answer="",
            conflict_pairs=[],
            notes=["conflict_mapping_unsupported"],
        )
        scored = _score_one(
            {"task": "conflict_audit"},
            result,
            gold_pairs=[("stmt-a", "stmt-b")],
        )
        self.assertTrue(scored["unsupported"])
        self.assertIsNone(scored["correct"])
        self.assertEqual(scored["scoring"], "conflict_pairs_unsupported")

    def test_score_one_marks_scan_failure_as_error(self) -> None:
        result = QueryResult(
            answer="",
            conflict_pairs=[],
            notes=["conflict_scan_error"],
        )
        scored = _score_one(
            {"task": "conflict_audit"},
            result,
            gold_pairs=[("stmt-a", "stmt-b")],
        )
        self.assertFalse(scored.get("unsupported"))
        self.assertEqual(scored["scoring"], "conflict_scan_error")
        self.assertEqual(scored["correct"], False)
        self.assertIn("scan failed", scored["error"])

    def test_score_one_marks_scan_unavailable_as_skipped(self) -> None:
        result = QueryResult(
            answer="",
            conflict_pairs=[],
            notes=["conflict_scan_unavailable"],
        )
        scored = _score_one(
            {"task": "conflict_audit"},
            result,
            gold_pairs=[("stmt-a", "stmt-b")],
        )
        self.assertTrue(scored["unsupported"])
        self.assertIsNone(scored["correct"])
        self.assertIn("not available", scored["reason"])

    def test_score_one_marks_unavailable_producer_unsupported(self) -> None:
        result = QueryResult(
            answer="",
            conflict_pairs=[],
            notes=["conflict_capability_pending"],
        )
        scored = _score_one(
            {"task": "conflict_audit"},
            result,
            gold_pairs=[("stmt-a", "stmt-b")],
        )
        self.assertTrue(scored["unsupported"])
        self.assertIn("producer is not available", scored["reason"])

    def test_empty_conflict_prediction_is_scored(self) -> None:
        result = QueryResult(
            answer="",
            conflict_pairs=[],
            notes=["conflict_producer_empty"],
        )
        scored = _score_one(
            {"task": "conflict_audit"},
            result,
            gold_pairs=[("stmt-a", "stmt-b")],
        )
        self.assertFalse(scored.get("unsupported"))
        self.assertEqual(scored["scoring"], "conflict_pairs")
        self.assertEqual(scored["f1"], 0.0)
        self.assertFalse(scored["correct"])

    def test_mixed_mapped_and_unmapped_headline_is_unsupported(self) -> None:
        records = [
            {
                "notes": ["conflict_ids_mapped"],
                "conflict_pairs": [("stmt-a", "stmt-b")],
                "unsupported": False,
                "scoring": "conflict_pairs",
            },
            {
                "notes": ["conflict_mapping_unsupported"],
                "conflict_pairs": [],
                "unsupported": True,
                "scoring": "conflict_pairs_unsupported",
            },
        ]
        headline = _conflict_headline(records, [("stmt-a", "stmt-b"), ("stmt-c", "stmt-d")])
        self.assertTrue(headline["unsupported"])
        self.assertIsNone(headline["f1"])

    def test_mean_token_f1_keeps_timeouts_in_denominator(self) -> None:
        rows = [
            {"token_f1": 1.0, "correct": True},
            {"correct": False, "error": "timeout"},
        ]
        headline = _aggregate(rows)
        self.assertEqual(headline["mean_token_f1"], 0.5)

    def test_compare_systems_uses_gradable_ids(self) -> None:
        records = [
            {"system": "hybrid_baseline", "question_id": "q1", "correct": True},
            {"system": "hybrid_baseline", "question_id": "q2", "correct": True},
            {"system": "brainapi", "question_id": "q1", "correct": True},
            {
                "system": "brainapi",
                "question_id": "q2",
                "unsupported": True,
                "correct": None,
            },
        ]
        result = compare_systems(
            {
                "hybrid_baseline": {
                    "metrics": {
                        "n": 2,
                        "n_unsupported": 0,
                        "accuracy": 1.0,
                        "wilson_lo": 0.3,
                        "wilson_hi": 1.0,
                    }
                },
                "brainapi": {
                    "capabilities": {},
                    "metrics": {
                        "n": 2,
                        "n_unsupported": 1,
                        "accuracy": 1.0,
                        "wilson_lo": 0.2,
                        "wilson_hi": 1.0,
                    },
                },
            },
            use_case="expert_knowhow",
            records=records,
        )
        self.assertEqual(result["verdict"], "incomplete")
        self.assertEqual(result["reason"], "Systems did not score the same question ids.")

    def test_compare_systems_keeps_timeout_rows(self) -> None:
        records = [
            {"system": "hybrid_baseline", "question_id": "q1", "correct": True},
            {
                "system": "hybrid_baseline",
                "question_id": "q2",
                "correct": False,
                "error": "timeout",
            },
            {"system": "brainapi", "question_id": "q1", "correct": True},
            {
                "system": "brainapi",
                "question_id": "q2",
                "correct": False,
            },
        ]
        result = compare_systems(
            {
                "hybrid_baseline": {
                    "metrics": {
                        "n": 2,
                        "n_unsupported": 0,
                        "accuracy": 0.5,
                        "wilson_lo": 0.09,
                        "wilson_hi": 0.91,
                    }
                },
                "brainapi": {
                    "capabilities": {},
                    "metrics": {
                        "n": 2,
                        "n_unsupported": 0,
                        "accuracy": 0.5,
                        "wilson_lo": 0.09,
                        "wilson_hi": 0.91,
                    },
                },
            },
            use_case="expert_knowhow",
            records=records,
        )
        self.assertEqual(result["verdict"], "undecided")

    def test_include_history_is_not_a_versioned_rules_win(self) -> None:
        result = compare_systems(
            {
                "hybrid_baseline": {
                    "metrics": {
                        "n": 150,
                        "accuracy": 0.9,
                        "wilson_lo": 0.84,
                        "wilson_hi": 0.94,
                    }
                },
                "brainapi": {
                    "capabilities": {"as_of": True, "as_of_mode": "include_history"},
                    "metrics": {
                        "n": 150,
                        "accuracy": 0.2,
                        "wilson_lo": 0.14,
                        "wilson_hi": 0.27,
                    },
                },
            },
            use_case="versioned_rules",
        )
        self.assertEqual(result["verdict"], "capability_pending")
        self.assertEqual(result["missing_capability"], "as_of")

    def test_score_one_marks_unsupported_changelog(self) -> None:
        result = QueryResult(
            answer="something",
            notes=["as_of_mode=unsupported"],
            raw={"as_of_mode": "unsupported"},
        )
        scored = _score_one(
            {"task": "changelog", "date_from": "2019-01-01", "date_to": "2024-01-01"},
            result,
        )
        self.assertTrue(scored["unsupported"])
        self.assertEqual(scored["scoring"], "as_of_unsupported")

    def test_headline_uses_producer_unavailable_reason(self) -> None:
        records = [
            {
                "notes": ["conflict_capability_pending"],
                "conflict_pairs": [],
                "unsupported": True,
                "scoring": "conflict_pairs_unsupported",
                "reason": "BrainAPI conflict pair producer is not available on this deployment",
            }
        ]
        headline = _conflict_headline(records, [("stmt-a", "stmt-b")])
        self.assertTrue(headline["unsupported"])
        self.assertIn("producer is not available", headline["reason"])

    def test_score_one_partial_mapping_is_unsupported(self) -> None:
        result = QueryResult(
            answer="",
            conflict_pairs=[("stmt-a", "stmt-b")],
            notes=["conflict_mapping_unsupported"],
        )
        scored = _score_one(
            {"task": "conflict_audit"},
            result,
            gold_pairs=[("stmt-a", "stmt-b"), ("stmt-c", "stmt-d")],
        )
        self.assertTrue(scored["unsupported"])
        self.assertIsNone(scored["correct"])


class JudgeParseTests(unittest.TestCase):
    def test_requires_json_boolean(self) -> None:
        correct, _reason = _parse_bool_payload('{"correct": true, "reason": "ok"}', "correct")
        self.assertTrue(correct)
        wrong, reason = _parse_bool_payload('{"correct": "false"}', "correct")
        self.assertFalse(wrong)
        self.assertIn("boolean", reason)
        nulls, _ = _parse_bool_payload('{"correct": null}', "correct")
        self.assertFalse(nulls)
        arrays, _ = _parse_bool_payload('{"correct": ["yes"]}', "correct")
        self.assertFalse(arrays)

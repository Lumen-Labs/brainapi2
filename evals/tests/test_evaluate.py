from __future__ import annotations

import unittest

from evals.evaluate import _aggregate, compare_systems
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

from __future__ import annotations

import unittest

from evals.metrics import (
    cohens_kappa,
    conflict_prf,
    exact_match,
    must_include,
    pair_key,
    percentile,
    score_prediction,
    token_f1,
    wilson_ci,
)


class MetricsTests(unittest.TestCase):
    def test_exact_match_normalizes(self) -> None:
        self.assertTrue(exact_match("Net 30 days.", "net 30 days", ["30 days"]))

    def test_must_include(self) -> None:
        self.assertTrue(must_include("changed from 10 days to 5 days", ["10", "5"]))
        self.assertFalse(must_include("changed to 5 days", ["10", "5"]))

    def test_token_f1(self) -> None:
        self.assertGreater(token_f1("drain shard-b", "isolate/drain the failing shard"), 0.2)

    def test_conflict_prf_order_invariant(self) -> None:
        prf = conflict_prf([("b", "a")], [("a", "b")])
        self.assertEqual(prf["tp"], 1)
        self.assertEqual(prf["f1"], 1.0)

    def test_pair_key(self) -> None:
        self.assertEqual(pair_key("z", "a"), ("a", "z"))

    def test_wilson_and_percentile(self) -> None:
        lo, hi = wilson_ci(8, 10)
        self.assertIsNotNone(lo)
        self.assertLess(lo, hi)
        self.assertEqual(percentile([1.0], 95), 1.0)
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)

    def test_cohens_kappa_perfect(self) -> None:
        labels = [True, False, True, False]
        self.assertEqual(cohens_kappa(labels, labels), 1.0)

    def test_score_prediction_modes(self) -> None:
        q = {
            "gold": "Net 30 days",
            "must_include": ["30"],
            "scoring": "must_include",
        }
        scored = score_prediction("payment is net 30", q)
        self.assertTrue(scored["correct"])
        self.assertTrue(scored["must_include"])

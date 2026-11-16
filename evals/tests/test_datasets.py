from __future__ import annotations

import unittest

from evals.config import USE_CASES
from evals.datasets.loader import load_judge_labels, load_use_case


class DatasetTests(unittest.TestCase):
    def test_every_use_case_has_splits(self) -> None:
        for name in USE_CASES:
            bundle = load_use_case(name)
            self.assertGreater(len(bundle["corpus"]), 0, name)
            for split in ("smoke", "dev", "test"):
                self.assertGreater(
                    len(bundle["questions"][split]),
                    0,
                    f"{name} missing {split}",
                )
            ids = [doc["id"] for doc in bundle["corpus"]]
            self.assertEqual(len(ids), len(set(ids)), name)

    def test_policy_gold_pairs_reference_statements(self) -> None:
        bundle = load_use_case("policy_conflicts")
        statements = {
            doc.get("statement_id") or doc["id"] for doc in bundle["corpus"]
        }
        self.assertGreaterEqual(len(bundle["gold_pairs"]), 4)
        for left, right in bundle["gold_pairs"]:
            self.assertIn(left, statements)
            self.assertIn(right, statements)
            self.assertNotEqual(left, right)

    def test_versioned_rules_have_intervals(self) -> None:
        bundle = load_use_case("versioned_rules")
        for doc in bundle["corpus"]:
            self.assertTrue(doc.get("valid_from"), doc["id"])
            self.assertTrue(doc.get("clause_id"), doc["id"])

    def test_judge_sample_has_both_labels(self) -> None:
        rows = load_judge_labels()
        self.assertGreaterEqual(len(rows), 8)
        labels = {bool(row["human_correct"]) for row in rows}
        self.assertEqual(labels, {True, False})

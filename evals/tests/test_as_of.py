from __future__ import annotations

import unittest

from evals.baseline.dates import filter_as_of, in_force, parse_date
from evals.baseline.index import build_index
from evals.datasets.loader import load_corpus, load_questions
from evals.metrics import must_include


class AsOfTests(unittest.TestCase):
    def test_parse_and_interval(self) -> None:
        self.assertEqual(str(parse_date("2021-06-15")), "2021-06-15")
        doc = {"valid_from": "2018-04-01", "valid_to": "2021-12-31"}
        self.assertTrue(in_force(doc, "2021-06-15"))
        self.assertFalse(in_force(doc, "2022-01-01"))

    def test_open_ended_valid_to(self) -> None:
        doc = {"valid_from": "2023-10-01", "valid_to": None}
        self.assertTrue(in_force(doc, "2024-01-10"))
        self.assertFalse(in_force(doc, "2020-01-01"))

    def test_versioned_rules_filter_picks_v1_payment(self) -> None:
        docs = load_corpus("versioned_rules")
        as_of = filter_as_of(docs, "2021-06-15")
        pay = [doc for doc in as_of if doc["clause_id"] == "FAMR 12.4"]
        self.assertEqual(len(pay), 1)
        self.assertEqual(pay[0]["version"], "2018.4")
        self.assertIn("Net 30", pay[0]["text"])

    def test_hybrid_as_of_does_not_return_future_version(self) -> None:
        docs = load_corpus("versioned_rules")
        index = build_index(docs, {"retrieve_k": 8, "rerank_depth": 8})
        hits = index.search("FAMR 12.4 payment period Net", k=8, as_of="2021-06-15")
        versions = {hit.get("version") for hit in hits if hit.get("clause_id") == "FAMR 12.4"}
        self.assertTrue(versions)
        self.assertNotIn("2023.4", versions)
        blob = " ".join(hit.get("text") or "" for hit in hits)
        self.assertTrue(must_include(blob, ["30"]))

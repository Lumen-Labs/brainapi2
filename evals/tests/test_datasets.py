from __future__ import annotations

import unittest

from evals.baseline.dates import (
    filter_as_of,
    require_document_source_timestamp,
)
from evals.baseline.index import DEFAULT_HYBRID_CONFIG
from evals.config import (
    CHEAP_CROSS_ENCODER,
    CHEAP_ST_EMBEDDING_MODEL,
    DEFAULT_CROSS_ENCODER,
    DEFAULT_ST_EMBEDDING_MODEL,
)
from evals.config import REQUIRED_USE_CASES, USE_CASES
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

    def test_required_test_splits_are_large_enough(self) -> None:
        for name in REQUIRED_USE_CASES:
            bundle = load_use_case(name)
            self.assertGreaterEqual(
                len(bundle["questions"]["test"]),
                150,
                f"{name} test n={len(bundle['questions']['test'])}",
            )
            smoke_docs = [doc for doc in bundle["corpus"] if doc.get("ci_smoke")]
            self.assertLessEqual(len(smoke_docs), 20, name)

    def test_policy_gold_pairs_reference_statements(self) -> None:
        bundle = load_use_case("policy_conflicts")
        statements = {
            doc.get("statement_id") or doc["id"] for doc in bundle["corpus"]
        }
        self.assertGreaterEqual(len(bundle["gold_pairs"]), 150)
        for left, right in bundle["gold_pairs"]:
            self.assertIn(left, statements)
            self.assertIn(right, statements)
            self.assertNotEqual(left, right)

    def test_versioned_rules_have_intervals(self) -> None:
        bundle = load_use_case("versioned_rules")
        for doc in bundle["corpus"]:
            self.assertTrue(doc.get("valid_from"), doc["id"])
            self.assertTrue(doc.get("clause_id"), doc["id"])
            stamp = require_document_source_timestamp(doc, doc["id"])
            self.assertTrue(stamp)

    def test_versioned_loader_rejects_missing_or_unparseable_dates(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            require_document_source_timestamp(
                {"id": "famr-x", "text": "hello", "clause_id": "FAMR 1"},
                "corpus.jsonl",
            )
        self.assertIn("missing parseable", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            require_document_source_timestamp(
                {
                    "id": "famr-x",
                    "text": "hello",
                    "valid_from": "not-a-date",
                },
                "corpus.jsonl",
            )
        self.assertIn("unparseable", str(ctx.exception))
        stamp = require_document_source_timestamp(
            {
                "id": "famr-ok",
                "text": "hello",
                "source_timestamp": "2021-06-15",
            }
        )
        self.assertEqual(stamp, "2021-06-15")
        with self.assertRaises(ValueError) as ctx:
            require_document_source_timestamp(
                {
                    "id": "famr-x",
                    "text": "hello",
                    "valid_from": "2021-06-15xyz",
                },
                "corpus.jsonl",
            )
        self.assertIn("unparseable", str(ctx.exception))

    def test_versioned_test_gold_is_in_force_text(self) -> None:
        bundle = load_use_case("versioned_rules")
        for question in bundle["questions"]["test"]:
            required = list(question.get("must_include") or [])
            if not required:
                continue
            if question.get("task") == "as_of":
                blob = " ".join(
                    str(doc.get("text") or "")
                    for doc in filter_as_of(bundle["corpus"], question.get("as_of"))
                    if doc.get("clause_id") == question.get("clause_id")
                )
            else:
                d1 = filter_as_of(bundle["corpus"], question.get("date_from"))
                d2 = filter_as_of(bundle["corpus"], question.get("date_to"))
                blob = " ".join(
                    str(doc.get("text") or "")
                    for doc in d1 + d2
                    if doc.get("clause_id") == question.get("clause_id")
                )
            missing = [phrase for phrase in required if phrase.lower() not in blob.lower()]
            self.assertFalse(missing, question.get("id"))

    def test_knowhow_test_gold_in_corpus(self) -> None:
        bundle = load_use_case("expert_knowhow")
        blob = " ".join(str(doc.get("text") or "") for doc in bundle["corpus"]).lower()
        for question in bundle["questions"]["test"]:
            for phrase in question.get("must_include") or []:
                self.assertIn(phrase.lower(), blob, question.get("id"))

    def test_knowhow_gold_is_scoped_to_hops(self) -> None:
        bundle = load_use_case("expert_knowhow")
        by_id = {str(doc.get("id")): doc for doc in bundle["corpus"]}
        for question in bundle["questions"]["test"]:
            hops = [str(item) for item in (question.get("hops") or [])]
            phrases = list(question.get("must_include") or [])
            if len(hops) < 2 or len(phrases) < 2:
                continue
            for hop_id, phrase in zip(hops, phrases):
                doc = by_id.get(hop_id)
                self.assertIsNotNone(doc, question.get("id"))
                self.assertIn(
                    phrase.lower(),
                    str(doc.get("text") or "").lower(),
                    question.get("id"),
                )

    def test_policy_natural_pairs_share_unit_family(self) -> None:
        from evals.datasets.build_public_corpora import is_natural_clash

        bundle = load_use_case("policy_conflicts")
        by_sid = {
            str(doc.get("statement_id") or doc["id"]): doc for doc in bundle["corpus"]
        }
        natural_questions = [
            q
            for q in bundle["questions"]["test"]
            if q.get("gold_method") == "natural_numeric_clash"
        ]
        for question in natural_questions:
            pairs = question.get("gold_pairs") or []
            self.assertTrue(pairs, question.get("id"))
            left, right = pairs[0]
            self.assertTrue(
                is_natural_clash(by_sid[left], by_sid[right]),
                question.get("id"),
            )

    def test_default_hybrid_is_strong(self) -> None:
        self.assertEqual(DEFAULT_HYBRID_CONFIG["dense"], "sentence-transformers")
        self.assertEqual(DEFAULT_HYBRID_CONFIG["reranker"], "cross-encoder")
        self.assertEqual(DEFAULT_HYBRID_CONFIG["dense_model"], DEFAULT_ST_EMBEDDING_MODEL)
        self.assertEqual(DEFAULT_ST_EMBEDDING_MODEL, "BAAI/bge-large-en-v1.5")
        self.assertEqual(DEFAULT_CROSS_ENCODER, "BAAI/bge-reranker-v2-m3")
        self.assertEqual(
            CHEAP_ST_EMBEDDING_MODEL, "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.assertEqual(
            CHEAP_CROSS_ENCODER, "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def test_judge_sample_has_both_labels(self) -> None:
        rows = load_judge_labels()
        self.assertGreaterEqual(len(rows), 8)
        labels = {bool(row["human_correct"]) for row in rows}
        self.assertEqual(labels, {True, False})

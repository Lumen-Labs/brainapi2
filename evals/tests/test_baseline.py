from __future__ import annotations

import unittest

from evals.baseline.bm25 import BM25Index
from evals.baseline.chunk import chunk_document
from evals.baseline.dense import cosine, hashed_ngram_vector, make_embedder
from evals.baseline.fusion import fuse, reciprocal_rank_fusion, weighted_fusion
from evals.baseline.index import HASH_INDEX_CONFIG, build_index
from evals.baseline.rerank import lexical_rerank


class BaselineTests(unittest.TestCase):
    def test_bm25_ranks_overlap_first(self) -> None:
        index = BM25Index()
        index.add("hot", "drain the failing shard from the serving pool")
        index.add("cold", "office coffee machine descaling instructions")
        hits = index.search("failing shard drain")
        self.assertEqual(hits[0][0], "hot")

    def test_rrf_and_weighted(self) -> None:
        rrf = reciprocal_rank_fusion([["x", "b", "c"], ["b", "a"]], k=10)
        self.assertEqual(rrf[0][0], "b")
        weighted = weighted_fusion(
            [("a", 1.0), ("b", 0.1)],
            [("b", 10.0), ("a", 0.0)],
            dense_weight=0.9,
        )
        self.assertEqual(weighted[0][0], "b")
        fused = fuse([("a", 1.0)], [("a", 1.0)], method="rrf")
        self.assertEqual(fused[0][0], "a")

    def test_make_embedder_hash_does_not_need_sentence_transformers(self) -> None:
        embedder = make_embedder("hash", dim=32)
        vectors, cost = embedder.embed(["inspection within 10 days"])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 32)
        self.assertEqual(cost.get("embedding_backend"), 0)

    def test_sentence_transformers_provider_errors_without_package(self) -> None:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError):
                make_embedder("sentence-transformers")
        else:
            self.skipTest("sentence-transformers is installed")

    def test_hash_embed_cosine(self) -> None:
        a = hashed_ngram_vector("inspection within 10 calendar days")
        b = hashed_ngram_vector("inspection within 10 calendar days after delivery")
        c = hashed_ngram_vector("unrelated baking recipe with chocolate")
        self.assertGreater(cosine(a, b), cosine(a, c))

    def test_chunk_respects_size(self) -> None:
        doc = {"id": "d1", "text": "Sentence one. Sentence two. Sentence three."}
        chunks = chunk_document(doc, chunk_size=20, chunk_overlap=5)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(all(chunk["doc_id"] == "d1" for chunk in chunks))

    def test_lexical_rerank(self) -> None:
        cands = [
            {"id": "1", "title": "coffee", "text": "beans"},
            {"id": "2", "title": "drain shard", "text": "failing shard drain"},
        ]
        ranked = lexical_rerank("drain failing shard", cands, depth=2)
        self.assertEqual(ranked[0]["id"], "2")

    def test_index_search_returns_k(self) -> None:
        docs = [
            {"id": "a", "title": "pay", "text": "Net 30 days after invoice"},
            {"id": "b", "title": "inspect", "text": "inspect supplies in 10 days"},
        ]
        index = build_index(
            docs, {**HASH_INDEX_CONFIG, "retrieve_k": 1, "rerank_depth": 1}
        )
        hits = index.search("Net 30 payment", k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["doc_id"], "a")

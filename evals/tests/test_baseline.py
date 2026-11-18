from __future__ import annotations

import unittest

from evals.baseline.bm25 import BM25Index
from evals.baseline.chunk import chunk_document
from evals.baseline.dense import cosine, hashed_ngram_vector, make_embedder
from evals.baseline.fusion import fuse, reciprocal_rank_fusion, weighted_fusion
from evals.baseline.index import HASH_INDEX_CONFIG, build_index, numeric_cost
from evals.baseline.rerank import lexical_rerank
from evals.baseline.tune import SMOKE_GRID, SOTA_TUNE_GRID, TUNE_GRID, evaluate_config, tune_hybrid


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

    def test_numeric_cost_drops_model_metadata(self) -> None:
        cost = numeric_cost(
            {
                "embedding_calls": 1,
                "embedding_tokens": 12,
                "embedding_backend": 0,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            }
        )
        self.assertEqual(cost["embedding_calls"], 1)
        self.assertNotIn("embedding_model", cost)

    def test_build_index_ignores_non_int_embed_cost(self) -> None:
        class TaggedHash:
            name = "hash"

            def embed(self, texts, *, is_query=False):
                del is_query
                vectors, cost = make_embedder("hash", dim=32).embed(texts)
                cost["embedding_model"] = "should-not-crash"
                return vectors, cost

        docs = [{"id": "a", "title": "pay", "text": "Net 30 days after invoice"}]
        index = build_index(
            docs,
            {**HASH_INDEX_CONFIG, "dense_dim": 32},
            embedder=TaggedHash(),
        )
        self.assertIsInstance(index.cost.get("embedding_tokens"), int)
        self.assertNotIn("embedding_model", index.cost)
        hits = index.search("Net 30")
        self.assertGreaterEqual(len(hits), 1)

    def test_evaluate_config_merges_partial_over_index(self) -> None:
        docs = [{"id": "a", "title": "pay", "text": "Net 30 days after invoice"}]
        questions = [
            {
                "id": "q",
                "question": "net 30",
                "gold": "Net 30",
                "must_include": ["Net 30"],
            }
        ]
        index = build_index(docs, HASH_INDEX_CONFIG)
        result = evaluate_config(
            docs, questions, {"fusion": "weighted"}, index=index
        )
        self.assertEqual(result["config"]["dense"], "hash")
        self.assertEqual(result["config"]["fusion"], "weighted")
        self.assertIn("retrieve_k", result["config"])

    def test_tune_records_lexical_without_cross_encoder(self) -> None:
        docs = [
            {"id": "a", "title": "pay", "text": "Net 30 days after invoice"},
            {"id": "b", "title": "inspect", "text": "inspect supplies in 10 days"},
        ]
        questions = [
            {
                "id": "q",
                "question": "net 30 payment",
                "gold": "Net 30",
                "must_include": ["Net 30"],
            }
        ]
        result = tune_hybrid(
            docs,
            questions,
            grid=SMOKE_GRID,
            embedder=make_embedder("hash", dim=32),
        )
        self.assertEqual(result["best_config"]["reranker"], "lexical")
        self.assertEqual(result["best_config"]["dense_dim"], 32)

    def test_sota_grid_is_smaller_than_minilm_grid(self) -> None:
        def n_cells(grid: dict) -> int:
            n = 1
            for values in grid.values():
                n *= len(values)
            return n

        self.assertLess(n_cells(SOTA_TUNE_GRID), n_cells(TUNE_GRID))
        self.assertEqual(n_cells(SOTA_TUNE_GRID), 16)
        self.assertIn(16, SOTA_TUNE_GRID["rerank_depth"])

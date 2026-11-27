import os
import unittest

ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "EMBEDDINGS_LOCAL_MODEL": "local-model",
    "EMBEDDINGS_SMALL_MODEL": "small-model",
    "EMBEDDING_NODES_DIMENSION": "3",
    "EMBEDDING_TRIPLETS_DIMENSION": "3",
    "EMBEDDING_OBSERVATIONS_DIMENSION": "3",
    "EMBEDDING_DATA_DIMENSION": "3",
    "EMBEDDING_RELATIONSHIPS_DIMENSION": "3",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
    "CELERY_WORKER_CONCURRENCY": "1",
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_LLM_SMALL_MODEL": "small",
    "OLLAMA_LLM_LARGE_MODEL": "large",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "postgres",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from src.core.saving.identity import merge_source_chunk_ids, stamp_provenance
from src.core.search.fact_filter import (
    filter_relevant_facts,
    personalized_pagerank,
    reciprocal_rank_fusion,
)
from src.utils.dates import normalize_date_string, resolve_relative_date


class ProvenanceHelpersTests(unittest.TestCase):
    def test_merge_source_chunk_ids_dedupes(self):
        merged = merge_source_chunk_ids(
            ["a", "b"],
            "b",
            ["c"],
            None,
            "a",
        )
        self.assertEqual(merged, ["a", "b", "c"])

    def test_stamp_provenance_accumulates(self):
        props = stamp_provenance(
            {"source_chunk_ids": ["chunk-1"]},
            source_chunk_id="chunk-2",
            source_timestamp="01/05/2023",
            existing_properties={"source_chunk_ids": ["chunk-0"]},
        )
        self.assertEqual(props["source_chunk_ids"], ["chunk-0", "chunk-1", "chunk-2"])
        self.assertEqual(props["source_timestamp"], "01/05/2023")
        keyed = stamp_provenance(
            {"source_chunk_ids": ["chunk-1"]},
            source_key="text:profile.txt:abc",
        )
        self.assertEqual(keyed["source_key"], "text:profile.txt:abc")


class RelativeDateTests(unittest.TestCase):
    def test_normalize_absolute(self):
        self.assertEqual(normalize_date_string("2023-05-15"), "15/05/2023")

    def test_resolve_yesterday(self):
        self.assertEqual(
            resolve_relative_date("yesterday", "15/05/2023"),
            "14/05/2023",
        )

    def test_resolve_last_tuesday(self):
        # 15/05/2023 is a Monday
        self.assertEqual(
            resolve_relative_date("last Tuesday", "15/05/2023"),
            "09/05/2023",
        )


class RrfTests(unittest.TestCase):
    def test_rrf_prefers_items_ranked_high_in_multiple_lists(self):
        fused = reciprocal_rank_fusion(
            [
                ["a", "b", "c"],
                ["b", "a", "d"],
            ]
        )
        ranked = [item for item, _ in fused]
        self.assertEqual(ranked[0], "a")
        self.assertIn("b", ranked[:2])


class FactFilterTests(unittest.TestCase):
    def test_fallback_preserves_all_candidates(self):
        keep = filter_relevant_facts(
            "What books has Melanie read?",
            [f"fact-{i}" for i in range(12)],
            llm_adapter=None,
            max_keep=4,
        )
        self.assertEqual(keep, list(range(12)))


class PersonalizedPageRankTests(unittest.TestCase):
    def test_ppr_concentrates_on_seed_neighborhood(self):
        adjacency = {
            "seed": ["a", "b"],
            "a": ["seed", "c"],
            "b": ["seed"],
            "c": ["a"],
            "far": ["x"],
            "x": ["far"],
        }
        scores = personalized_pagerank(adjacency, {"seed": 1.0}, iterations=30)
        self.assertGreater(scores["seed"], scores["far"])
        self.assertGreater(scores["a"], scores["far"])


class ScoutChunkingTests(unittest.TestCase):
    def test_chunk_text_splits_long_input(self):
        from src.utils.text_chunking import chunk_text

        text = ("paragraph one.\n\n" * 50) + ("x" * 7000)
        chunks = chunk_text(text, max_chars=1000)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 1000 or c.count("x") > 0 for c in chunks))


if __name__ == "__main__":
    unittest.main()

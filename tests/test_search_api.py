import os
import unittest
from unittest.mock import MagicMock, patch

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
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from fastapi import HTTPException

from src.constants.embeddings import Vector
from src.constants.data import TextChunk
from src.services.api.constants.requests import SearchRequestBody


class SearchApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_404_when_search_disabled(self):
        from src.services.api.controllers import search as search_mod

        with patch.object(search_mod.config, "search_enabled", False):
            with self.assertRaises(HTTPException) as ctx:
                await search_mod.search(SearchRequestBody(query="license"))
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_fused_hits_when_both_channels_on(self):
        from src.services.api.controllers import search as search_mod

        dense_vec = Vector(
            id="dense-1",
            metadata={"resource_id": "chunk-dense"},
            distance=0.1,
        )
        bm25_chunk = TextChunk(id="chunk-bm25", text="counseling license renewal")
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_data.return_value = [dense_vec]
        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [(bm25_chunk, 4.2)]
        mock_data.get_text_chunks_by_ids.return_value = (
            [TextChunk(id="chunk-dense", text="dense passage about alice")],
            [],
        )

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", True),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod.config, "search_fusion", "rrf"),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(
                SearchRequestBody(query="license", k=10, profile_stages=True)
            )

        ids = [hit.id for hit in response.hits]
        self.assertIn("chunk-dense", ids)
        self.assertIn("chunk-bm25", ids)
        self.assertTrue(any(hit.scores.rrf for hit in response.hits))
        mock_embeddings.embed_text.assert_called_once_with("license")
        self.assertIsNotNone(response.stage_timings)
        stage_names = [s["stage"] for s in response.stage_timings.get("stages", [])]
        self.assertIn("embed.query", stage_names)
        self.assertTrue(
            any(name.startswith("search.") for name in stage_names),
            stage_names,
        )

    async def test_dense_only_skips_bm25(self):
        from src.services.api.controllers import search as search_mod

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_data.return_value = [
            Vector(id="d1", metadata={"resource_id": "only-dense"}, distance=0.2)
        ]
        mock_data = MagicMock()
        mock_data.get_text_chunks_by_ids.return_value = (
            [TextChunk(id="only-dense", text="vector hit")],
            [],
        )

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", True),
            patch.object(search_mod.config, "search_use_bm25", False),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(SearchRequestBody(query="alice", k=5))

        self.assertEqual([hit.id for hit in response.hits], ["only-dense"])
        mock_data.search_bm25.assert_not_called()
        self.assertIsNotNone(response.hits[0].scores.dense)

    async def test_bm25_only_skips_embed(self):
        from src.services.api.controllers import search as search_mod

        mock_embeddings = MagicMock()
        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [
            (TextChunk(id="lex", text="license board"), 3.1)
        ]
        mock_vs = MagicMock()

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(SearchRequestBody(query="license", k=5))

        self.assertEqual([hit.id for hit in response.hits], ["lex"])
        mock_embeddings.embed_text.assert_not_called()
        mock_vs.search_data.assert_not_called()
        self.assertEqual(response.hits[0].scores.bm25, 3.1)
        self.assertIsNone(response.hits[0].node_id)
        self.assertEqual(response.node_ids, [])

    async def test_passage_docid_sets_node_id(self):
        from src.services.api.controllers import search as search_mod

        mock_embeddings = MagicMock()
        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [
            (
                TextChunk(
                    id="chunk-bed",
                    text="DOCID 0.\nTitle: solid wood platform bed\nClass: Beds",
                ),
                5.0,
            )
        ]
        mock_vs = MagicMock()

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(
                SearchRequestBody(query="platform bed", k=5)
            )

        self.assertEqual([hit.id for hit in response.hits], ["chunk-bed"])
        self.assertEqual(response.hits[0].channel, "passages")
        self.assertEqual(response.hits[0].node_id, "0")
        self.assertEqual(response.node_ids, ["0"])

    def test_default_search_mode_is_default(self):
        body = SearchRequestBody(query="license")
        self.assertEqual(body.mode, "default")
        self.assertEqual(body.k, 10)
        self.assertIsNone(body.extras)

    async def test_extras_equality_filter_and_facets(self):
        from src.services.api.controllers import search as search_mod
        from src.core.search.hybrid import (
            facet_counts_from_extras,
            hit_matches_extras,
        )

        self.assertTrue(hit_matches_extras({"locale": "it"}, {"locale": "IT"}))
        self.assertFalse(hit_matches_extras({"color": "argento"}, {"color": "nope"}))
        self.assertEqual(
            facet_counts_from_extras(
                [
                    {"locale": "it", "color": "argento", "brand": "CasaLuce"},
                    {"locale": "it", "color": "blu navy", "brand": "AtelierNord"},
                    None,
                ]
            ),
            {
                "locale": {"it": 2},
                "color": {"argento": 1, "blu navy": 1},
                "brand": {"CasaLuce": 1, "AtelierNord": 1},
            },
        )
        self.assertIsNone(facet_counts_from_extras([None, {}]))
        self.assertEqual(
            facet_counts_from_extras(
                [{"locale": "it", "uuid": "skip-me", "resource_id": "chunk"}]
            ),
            {"locale": {"it": 1}},
        )

        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [
            (
                TextChunk(
                    id="it-1",
                    text="bollitore acciaio",
                    metadata={"locale": "it", "color": "argento", "brand": "CasaLuce"},
                ),
                3.0,
            ),
            (
                TextChunk(
                    id="it-2",
                    text="divano velluto",
                    metadata={"locale": "it", "color": "blu navy", "brand": "AtelierNord"},
                ),
                2.0,
            ),
        ]
        mock_data.get_text_chunks_by_ids.return_value = ([], [])
        mock_embeddings = MagicMock()
        mock_vs = MagicMock()

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            kept = await search_mod.search(
                SearchRequestBody(query="divano", k=10, extras={"locale": "it"})
            )
            missed = await search_mod.search(
                SearchRequestBody(query="divano", k=10, extras={"color": "nope"})
            )
            unfiltered = await search_mod.search(
                SearchRequestBody(query="divano", k=10)
            )

        self.assertEqual({hit.id for hit in kept.hits}, {"it-1", "it-2"})
        self.assertEqual(kept.hits[0].extras["locale"], "it")
        self.assertEqual(kept.facets["locale"]["it"], 2)
        self.assertEqual(kept.facets["brand"]["CasaLuce"], 1)
        self.assertEqual(missed.hits, [])
        self.assertIsNone(missed.facets)
        self.assertEqual({hit.id for hit in unfiltered.hits}, {"it-1", "it-2"})
        self.assertEqual(unfiltered.hits[0].channel, "passages")
        mock_embeddings.embed_text.assert_not_called()

    def test_literal_overlap_and_frozen_head_helpers(self):
        from src.core.search.hybrid import (
            frozen_head_merge,
            fuse_passage_lists,
            literal_overlap_ids,
        )

        ranked = frozen_head_merge(
            [f"h{index}" for index in range(12)],
            [["extra", "h0"]],
            head_k=10,
            k=15,
        )
        self.assertEqual(ranked[:10], [f"h{index}" for index in range(10)])
        self.assertEqual(ranked[10], "extra")
        ids = literal_overlap_ids(
            "navy velvet sofa",
            {
                "hit": "Title: Navy velvet sofa two seats",
                "miss": "plastic kettle lid",
                "partial": "velvet throw pillow",
            },
            k=10,
        )
        self.assertEqual(ids[0], "hit")
        self.assertIn("partial", ids)
        self.assertNotIn("miss", ids)
        fused = fuse_passage_lists(
            ["d1", "d2"],
            ["b1", "d1"],
            fusion="cc",
            alpha=0.7,
            dense_similarities={"d1": 1.0, "d2": 0.2},
            bm25_scores={"b1": 4.0, "d1": 1.0},
        )
        self.assertTrue(fused)
        self.assertEqual(fused[0][0], "d1")

    async def test_fusion_alpha_request_overrides_config(self):
        from src.services.api.controllers import search as search_mod

        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [
            (TextChunk(id="lex", text="license board"), 3.1)
        ]
        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_data.return_value = [
            Vector(id="d1", metadata={"resource_id": "dense-1"}, distance=0.1)
        ]
        mock_data.get_text_chunks_by_ids.return_value = (
            [TextChunk(id="dense-1", text="vector hit")],
            [],
        )

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", True),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod.config, "search_fusion", "rrf"),
            patch.object(search_mod.config, "search_fusion_alpha", 0.1),
            patch.object(search_mod.config, "search_literal_fill", False),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(
                SearchRequestBody(
                    query="license",
                    k=5,
                    fusion="cc",
                    fusion_alpha=0.9,
                )
            )

        ids = [hit.id for hit in response.hits]
        self.assertIn("dense-1", ids)
        self.assertTrue(any(hit.scores.cc is not None for hit in response.hits))

    async def test_literal_fill_uses_frozen_head(self):
        from src.services.api.controllers import search as search_mod

        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [
            (TextChunk(id=f"h{index}", text=f"t{index}"), float(20 - index))
            for index in range(12)
        ]
        mock_data.get_text_chunks.return_value = (
            [TextChunk(id="title-hit", text="navy velvet sofa")],
            1,
        )
        mock_embeddings = MagicMock()
        mock_vs = MagicMock()

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod.config, "search_literal_fill", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            response = await search_mod.search(
                SearchRequestBody(query="navy velvet sofa", k=15)
            )

        ids = [hit.id for hit in response.hits]
        self.assertEqual(ids[:10], [f"h{index}" for index in range(10)])
        self.assertIn("title-hit", ids[10:])
        self.assertEqual(response.channel_lists["literal"], ["title-hit"])

    async def test_as_of_reports_undated_hits_behind_opt_in(self):
        from src.services.api.controllers import search as search_mod

        dated = TextChunk(
            id="dated",
            text="dated passage",
            metadata={"valid_from": "2024-01-01T00:00:00Z"},
        )
        undated = TextChunk(id="plain", text="no dates", metadata={})
        mock_embeddings = MagicMock()
        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [(dated, 3.1), (undated, 2.0)]
        mock_data.get_text_chunks_by_ids.return_value = ([dated, undated], [])
        mock_vs = MagicMock()

        with (
            patch.object(search_mod.config, "search_enabled", True),
            patch.object(search_mod.config, "search_use_dense", False),
            patch.object(search_mod.config, "search_use_bm25", True),
            patch.object(search_mod, "embeddings_adapter", mock_embeddings),
            patch.object(search_mod, "vector_search", mock_vs),
            patch.object(search_mod, "data_adapter", mock_data),
        ):
            hidden = await search_mod.search(
                SearchRequestBody(
                    query="license", k=5, as_of="2024-06-01T00:00:00Z"
                )
            )
            shown = await search_mod.search(
                SearchRequestBody(
                    query="license",
                    k=5,
                    as_of="2024-06-01T00:00:00Z",
                    include_validity_unknown=True,
                )
            )
            live = await search_mod.search(SearchRequestBody(query="license", k=5))

        self.assertEqual([hit.id for hit in hidden.hits], ["dated"])
        self.assertEqual(hidden.validity_unknown_count, 1)
        self.assertIsNone(hidden.validity_unknown)
        self.assertEqual([hit.id for hit in shown.hits], ["dated"])
        self.assertEqual(shown.validity_unknown_count, 1)
        self.assertEqual(shown.validity_unknown[0]["validity"], "unknown")
        self.assertEqual(shown.validity_unknown[0]["id"], "plain")
        self.assertEqual({hit.id for hit in live.hits}, {"dated", "plain"})
        self.assertIsNone(live.validity_unknown_count)


if __name__ == "__main__":
    unittest.main()

import asyncio
import os
import sys
import types
import unittest
from types import SimpleNamespace
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
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "postgres",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.embeddings import Vector
from src.constants.data import TextChunk
from src.constants.kg import Node, Predicate
from src.core.search.entity_info import EventSynergyRetriever
from src.services.api.constants.requests import GetContextRequestBody
from src.utils.nlp.ner import ExtractElementsResponse


def _node(uuid: str, name: str, *, happened_at: str | None = None, labels=None):
    return Node(
        uuid=uuid,
        name=name,
        labels=labels or ["PERSON"],
        description=f"desc-{name}",
        happened_at=happened_at,
    )


def _pred(uuid: str, name: str):
    return Predicate(uuid=uuid, name=name, description=f"desc-{name}")


class CollectQueryVariantsTests(unittest.TestCase):
    def test_includes_noun_chunks_deduped_and_capped(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        elements = ExtractElementsResponse(
            tokens=[{"text": "Alice", "lemma": "Alice", "pos": "PROPN"}],
            noun_chunks=[
                "closest friend",
                "Alice",
                "charity event",
                "weekend trip",
                "old school",
                "extra chunk",
                "another one",
            ],
        )
        variants = retrieve_mod._collect_query_variants(
            "What did Alice do with her closest friend?", elements
        )
        self.assertEqual(variants[0], "What did Alice do with her closest friend?")
        self.assertIn("Alice", variants)
        self.assertIn("closest friend", variants)
        self.assertIn("charity event", variants)
        noun_chunks_included = [
            chunk
            for chunk in elements.noun_chunks
            if chunk in variants and chunk != "Alice"
        ]
        self.assertLessEqual(
            len(noun_chunks_included), retrieve_mod._MAX_NOUN_CHUNKS
        )
        self.assertTrue(
            any("friend" in v.lower() for v in variants[1:]),
            "noun-chunk variants should keep the friend mention",
        )

    def test_generic_brain_question_is_not_enumeration(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        question = "what is this brain about?"
        self.assertFalse(retrieve_mod._is_enumeration_question(question))
        elements = ExtractElementsResponse(
            tokens=[
                {"text": "brain", "lemma": "brain", "pos": "NOUN"},
                {"text": "this", "lemma": "this", "pos": "DET"},
            ],
            noun_chunks=["this brain"],
        )
        extras = retrieve_mod._decompose_enumeration_queries(question, elements)
        self.assertEqual(extras, [])
        variants = retrieve_mod._collect_query_variants(question, elements)
        self.assertEqual(variants[0], question)
        self.assertNotIn("brain about", " ".join(variants[1:]).lower())

    def test_list_question_still_counts_as_enumeration(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        self.assertTrue(
            retrieve_mod._is_enumeration_question("what books has Caroline read?")
        )
        self.assertTrue(
            retrieve_mod._is_enumeration_question("list the types of events")
        )


class FormatHelpersTests(unittest.TestCase):
    def test_format_includes_happened_at(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        n = _node("a", "Alice")
        r = _pred("r1", "ATTENDED")
        m = _node("e", "Wedding", happened_at="2024-05-01", labels=["EVENT"])
        r2 = _pred("r2", "INVOLVED")
        b = _node("b", "Bob")
        line = retrieve_mod._format_event_fact(n, r, m, r2, b)
        self.assertIn("@2024-05-01", line)
        self.assertIn("Wedding", line)


class GetContextRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_retains_all_event_facts_not_just_first(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        facts = [
            (
                _node("a", "Alice"),
                _pred("r1", "ATTENDED"),
                _node("e1", "Event1", happened_at="2024-01-01", labels=["EVENT"]),
                _pred("r2", "WITH"),
                _node("b", "Bob"),
            ),
            (
                _node("a", "Alice"),
                _pred("r3", "VISITED"),
                _node("e2", "Event2", happened_at="2024-02-01", labels=["EVENT"]),
                _pred("r4", "IN"),
                _node("c", "Paris", labels=["LOCATION"]),
            ),
            (
                _node("a", "Alice"),
                _pred("r5", "BOUGHT"),
                _node("e3", "Event3", labels=["EVENT"]),
                _pred("r6", "ITEM"),
                _node("d", "Book", labels=["OBJECT"]),
            ),
            (
                _node("a", "Alice"),
                _pred("r7", "MET"),
                _node("e4", "Event4", labels=["EVENT"]),
                _pred("r8", "WITH"),
                _node("f", "Carol"),
            ),
            (
                _node("a", "Alice"),
                _pred("r9", "MOVED"),
                _node("e5", "Event5", labels=["EVENT"]),
                _pred("r10", "TO"),
                _node("g", "London", labels=["LOCATION"]),
            ),
        ]

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )

        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = [
            Vector(
                id="n1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "a", "name": "Alice"},
                distance=0.1,
            )
        ]
        mock_vs.search_relationships.return_value = []

        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = facts

        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []

        elements = ExtractElementsResponse(tokens=[], noun_chunks=[])

        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "graph_adapter", mock_graph),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(
                retrieve_mod._entity_extractor,
                "extract_elements",
                return_value=elements,
            ),
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(text="Alice", brain_id="brain-a", max_facts=40)
            )

        self.assertEqual(len(response.triples), 5)
        self.assertIn("@2024-01-01", response.text_context)
        self.assertIn("Paris", response.text_context)

    async def test_relationship_vector_search_seeds_facts(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        fact = (
            _node("tail", "Alice"),
            _pred("r1", "ORGANIZED"),
            _node("mid", "Fundraiser", happened_at="2023-11-11", labels=["EVENT"]),
            _pred("r2", "FOR"),
            _node("tip", "Charity", labels=["ORG"]),
        )

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )

        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = []
        mock_vs.search_relationships.return_value = [
            Vector(
                id="rel1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "r1", "node_ids": ["tail", "mid"]},
                distance=0.05,
            )
        ]

        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = [fact]

        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []

        elements = ExtractElementsResponse(tokens=[], noun_chunks=[])

        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "graph_adapter", mock_graph),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(
                retrieve_mod._entity_extractor,
                "extract_elements",
                return_value=elements,
            ),
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="who organized the fundraiser",
                    brain_id="brain-a",
                )
            )

        mock_graph.get_event_centric_neighbors.assert_called()
        seed_uuids = mock_graph.get_event_centric_neighbors.call_args[0][0]
        self.assertIn("tail", seed_uuids)
        self.assertIn("mid", seed_uuids)
        self.assertEqual(len(response.triples), 1)
        self.assertIn("Fundraiser", response.text_context)

    async def test_max_facts_cap_and_dedup(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        shared = (
            _node("a", "Alice"),
            _pred("r1", "ATTENDED"),
            _node("e1", "Event1", labels=["EVENT"]),
            _pred("r2", "WITH"),
            _node("b", "Bob"),
        )
        other = (
            _node("a", "Alice"),
            _pred("r3", "VISITED"),
            _node("e2", "Event2", labels=["EVENT"]),
            _pred("r4", "IN"),
            _node("c", "Paris", labels=["LOCATION"]),
        )

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = [
            Vector(
                id="n1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "a", "name": "Alice"},
                distance=0.2,
            )
        ]
        mock_vs.search_relationships.return_value = []
        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = [shared, shared, other]
        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []
        elements = ExtractElementsResponse(tokens=[], noun_chunks=[])

        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "graph_adapter", mock_graph),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(
                retrieve_mod._entity_extractor,
                "extract_elements",
                return_value=elements,
            ),
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(text="Alice", brain_id="brain-a", max_facts=1)
            )

        self.assertEqual(len(response.triples), 1)

    async def test_context_path_does_not_emit_dossier_lines(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        fact = (
            _node("a", "Alice"),
            _pred("r1", "KNOWS"),
            _node("e1", "Meeting", labels=["EVENT"]),
            _pred("r2", "WITH"),
            _node("b", "Bob"),
        )

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = [
            Vector(
                id="n1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "a", "name": "Alice"},
                distance=0.1,
            )
        ]
        mock_vs.search_relationships.return_value = []
        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = [fact]
        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []
        elements = ExtractElementsResponse(
            tokens=[{"text": "Alice", "lemma": "Alice", "pos": "PROPN"}],
            noun_chunks=[],
        )

        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "graph_adapter", mock_graph),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(
                retrieve_mod._entity_extractor,
                "extract_elements",
                return_value=elements,
            ),
            patch(
                "src.core.search.entity_info.EventSynergyRetriever"
            ) as synergy_cls,
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="Where does Alice's friend live?",
                    brain_id="brain-a",
                )
            )

        synergy_cls.assert_not_called()
        self.assertFalse(
            any(line.startswith("[dossier:") for line in response.text_context.splitlines())
        )
        self.assertEqual(len(response.triples), 1)


class EventSynergyHardeningTests(unittest.TestCase):
    def test_explores_top_k_branches_not_only_greedy_best(self):
        alice = _node("a", "Alice")
        wrong = _node("w", "Wrong")
        mid = _node("m", "Bridge")
        goal = _node("g", "Goal")
        rel_wrong = _pred("rw", "DISTRACTOR")
        rel_wrong.properties = {"v_id": "v_wrong"}
        rel_bridge = _pred("rb", "VIA")
        rel_bridge.properties = {"v_id": "v_bridge"}
        rel_goal = _pred("rg", "REACHES")
        rel_goal.properties = {"v_id": "v_goal"}

        neighbors = {
            "a": [(rel_wrong, wrong), (rel_bridge, mid)],
            "m": [(rel_goal, goal)],
            "w": [],
            "g": [],
        }

        embeddings = {
            "v_wrong": [1.0, 0.0, 0.0],
            "v_bridge": [0.8, 0.2, 0.0],
            "v_goal": [0.0, 1.0, 0.0],
        }
        query_embedding = [0.0, 1.0, 0.0]

        def fake_get_by_ids(ids, store="relationships", brain_id="default"):
            vid = ids[0]
            return [
                Vector(id=vid, embeddings=embeddings[vid], metadata={})
            ]

        with (
            patch(
                "src.core.search.entity_info.graph_adapter.get_neighbors",
                side_effect=lambda nodes, brain_id="default", **kwargs: {
                    nodes[0]: neighbors.get(nodes[0], [])
                },
            ),
            patch(
                "src.core.search.entity_info.vector_store_adapter.get_by_ids",
                side_effect=fake_get_by_ids,
            ),
            patch(
                "src.core.search.entity_info.cosine_similarity",
                side_effect=lambda a, b: sum(x * y for x, y in zip(a, b)),
            ),
        ):
            retriever = EventSynergyRetriever("brain-a")
            path = retriever._recursive_explorer(
                "a",
                query_embedding,
                depth=3,
                visited_ids=set(),
                work_counter=[0],
                branch_factor=3,
            )

        self.assertGreaterEqual(len(path), 2)
        hop_names = [hop[0][0].name for hop in path]
        self.assertIn("VIA", hop_names)
        self.assertIn("REACHES", hop_names)

    def test_work_cap_stops_exploration(self):
        nodes = {f"n{i}": _node(f"n{i}", f"N{i}") for i in range(20)}
        rels = {}
        for i in range(19):
            pred = _pred(f"r{i}", f"REL{i}")
            pred.properties = {"v_id": f"v{i}"}
            rels[f"n{i}"] = [(pred, nodes[f"n{i+1}"])]
        rels["n19"] = []

        with (
            patch(
                "src.core.search.entity_info.graph_adapter.get_neighbors",
                side_effect=lambda nodes_arg, brain_id="default", **kwargs: {
                    nodes_arg[0]: rels.get(nodes_arg[0], [])
                },
            ),
            patch(
                "src.core.search.entity_info.vector_store_adapter.get_by_ids",
                side_effect=lambda ids, store="relationships", brain_id="default": [
                    Vector(id=ids[0], embeddings=[1.0, 0.0], metadata={})
                ],
            ),
            patch(
                "src.core.search.entity_info.cosine_similarity",
                return_value=0.5,
            ),
            patch("src.core.search.entity_info._MAX_EXPLORATION_WORK", 3),
        ):
            retriever = EventSynergyRetriever("brain-a")
            work = [0]
            retriever._recursive_explorer(
                "n0",
                [1.0, 0.0],
                depth=20,
                visited_ids=set(),
                work_counter=work,
                branch_factor=1,
            )
            self.assertLessEqual(work[0], 3)


class ContextPassageModeTests(unittest.TestCase):
    def test_search_off_keeps_ilike_lexical_leg(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_data.return_value = [
            Vector(id="d", metadata={"resource_id": "dense-1"}, distance=0.2)
        ]
        ilike_chunk = TextChunk(id="ilike-1", text="ilike hit about license")
        mock_data = MagicMock()
        mock_data.search.return_value = SimpleNamespace(text_chunks=[ilike_chunk])
        mock_data.get_text_chunks_by_ids.return_value = (
            [TextChunk(id="dense-1", text="dense body")],
            [],
        )

        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(retrieve_mod.config, "search_enabled", False),
            patch.object(retrieve_mod.config, "context_passage_mode", "hybrid"),
        ):
            hits = retrieve_mod._retrieve_passages("license", "brain-a", limit=8)

        mock_data.search.assert_called()
        mock_data.search_bm25.assert_not_called()
        self.assertTrue(any(item[0] == "ilike-1" for item in hits))

    def test_search_on_hybrid_uses_bm25(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_data.return_value = [
            Vector(id="d", metadata={"resource_id": "dense-1"}, distance=0.2)
        ]
        bm25_chunk = TextChunk(id="bm25-1", text="bm25 hit about license")
        mock_data = MagicMock()
        mock_data.search_bm25.return_value = [(bm25_chunk, 5.0)]
        mock_data.get_text_chunks_by_ids.return_value = (
            [TextChunk(id="dense-1", text="dense body")],
            [],
        )

        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(retrieve_mod.config, "search_enabled", True),
            patch.object(retrieve_mod.config, "context_passage_mode", "hybrid"),
        ):
            hits = retrieve_mod._retrieve_passages("license", "brain-a", limit=8)

        mock_data.search_bm25.assert_called()
        mock_data.search.assert_not_called()
        ids = [item[0] for item in hits]
        self.assertIn("bm25-1", ids)
        self.assertIn("dense-1", ids)

    def test_search_on_ilike_freezes_old_lexical_leg(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_data.return_value = []
        ilike_chunk = TextChunk(id="ilike-2", text="legacy ilike")
        mock_data = MagicMock()
        mock_data.search.return_value = SimpleNamespace(text_chunks=[ilike_chunk])

        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(retrieve_mod.config, "search_enabled", True),
            patch.object(retrieve_mod.config, "context_passage_mode", "ilike"),
        ):
            hits = retrieve_mod._retrieve_passages("license", "brain-a", limit=8)

        mock_data.search.assert_called()
        mock_data.search_bm25.assert_not_called()
        self.assertEqual(hits[0][0], "ilike-2")


class HistoricalContextReuseTests(unittest.IsolatedAsyncioTestCase):
    async def test_historical_context_reuses_primary_passage_hits(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        mock_embeddings = MagicMock()
        mock_embeddings.embed_text.return_value = Vector(
            id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
        )
        mock_vs = MagicMock()
        mock_vs.search_nodes.return_value = []
        mock_vs.search_relationships.return_value = []
        mock_graph = MagicMock()
        mock_graph.get_event_centric_neighbors.return_value = []
        mock_data = MagicMock()
        mock_data.get_last_text_chunks.return_value = []
        mock_data.get_last_structured_data.return_value = []
        mock_data.get_text_chunks_by_ids.side_effect = RuntimeError("no lookup")
        elements = ExtractElementsResponse(tokens=[], noun_chunks=[])
        retrieve_calls: list[str] = []

        def fake_retrieve(text, brain_id, *, limit, embeddings=None):
            retrieve_calls.append(text)
            return [("chunk-1", 0.1, "cached passage")]

        with (
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings),
            patch.object(retrieve_mod, "vector_search", mock_vs),
            patch.object(retrieve_mod, "graph_adapter", mock_graph),
            patch.object(retrieve_mod, "data_adapter", mock_data),
            patch.object(
                retrieve_mod._entity_extractor,
                "extract_elements",
                return_value=elements,
            ),
            patch.object(
                retrieve_mod, "_retrieve_passages", side_effect=fake_retrieve
            ),
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="what is this brain about?",
                    brain_id="brain-a",
                )
            )

        self.assertEqual(retrieve_calls.count("what is this brain about?"), 1)
        self.assertEqual(response.historical_context, ["cached passage"])


if __name__ == "__main__":
    unittest.main()

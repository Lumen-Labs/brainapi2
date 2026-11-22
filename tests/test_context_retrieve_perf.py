import asyncio
import contextlib
import io
import logging
import os
import sys
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
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USERNAME": "postgres",
    "POSTGRES_PASSWORD": "postgres",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate
from src.core.saving.history_intent import reset_prototype_cache
from src.services.api.constants.requests import GetContextRequestBody, GetContextResponse
from src.utils.nlp.ner import ExtractElementsResponse


RESPONSE_FIELDS = {
    "text_context",
    "triples",
    "historical_context",
    "source_passages",
    "graph_session_ids",
    "temporal_conflicts",
    "paths",
    "topics",
    "stage_timings",
    "validity_unknown_count",
    "validity_unknown",
}


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


def _facts(count: int):
    return [
        (
            _node("a", "Alice"),
            _pred(f"r{i}a", f"ATTENDED_{i}"),
            _node(
                f"e{i}",
                f"Event{i}",
                happened_at=f"2024-0{i + 1}-01",
                labels=["EVENT"],
            ),
            _pred(f"r{i}b", f"WITH_{i}"),
            _node(f"b{i}", f"Person{i}"),
        )
        for i in range(count)
    ]


@contextlib.contextmanager
def _stubbed_adapters(retrieve_mod, facts, extra=(), *, embed_texts=None):
    mock_embeddings = MagicMock()
    mock_embeddings.embed_text.return_value = Vector(
        id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
    )
    if embed_texts is None:
        mock_embeddings.embed_texts.side_effect = lambda texts: [
            Vector(id=f"b{i}", embeddings=[0.1, 0.2, 0.3], metadata={})
            for i, _ in enumerate(texts)
        ]
    else:
        mock_embeddings.embed_texts.side_effect = embed_texts
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
    mock_graph.get_hub_bridges.return_value = []
    mock_graph.list_topic_memberships.return_value = []
    mock_data = MagicMock()
    mock_data.get_last_text_chunks.return_value = []
    mock_data.get_last_structured_data.return_value = []
    mock_data.search.return_value = MagicMock(text_chunks=[])
    mock_data.get_text_chunks_by_ids.return_value = ([], [])

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(retrieve_mod, "embeddings_adapter", mock_embeddings)
        )
        stack.enter_context(patch.object(retrieve_mod, "vector_search", mock_vs))
        stack.enter_context(patch.object(retrieve_mod, "graph_adapter", mock_graph))
        stack.enter_context(patch.object(retrieve_mod, "data_adapter", mock_data))
        stack.enter_context(
            patch.object(
                retrieve_mod._entity_extractor,
                "extract_elements",
                return_value=ExtractElementsResponse(
                    tokens=[{"text": "Alice", "lemma": "Alice", "pos": "PROPN"}],
                    noun_chunks=[],
                ),
            )
        )
        for context in extra:
            stack.enter_context(context)
        yield mock_embeddings, mock_vs, mock_graph, mock_data


class ContextResponseShapeTests(unittest.IsolatedAsyncioTestCase):
    def test_response_model_fields_are_locked(self):
        self.assertEqual(set(GetContextResponse.model_fields), RESPONSE_FIELDS)

    async def test_stubbed_response_dump_matches_locked_shape(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        reset_prototype_cache()
        with _stubbed_adapters(retrieve_mod, _facts(2)):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(text="Alice", brain_id="brain-a")
            )

        dumped = response.model_dump()
        self.assertEqual(set(dumped), RESPONSE_FIELDS)
        self.assertIsInstance(response.text_context, str)
        self.assertIsInstance(response.triples, list)
        self.assertIsInstance(response.historical_context, list)
        self.assertIsInstance(response.source_passages, list)
        self.assertIsNone(response.stage_timings)


class HistoryIntentPathTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        reset_prototype_cache()

    async def test_lexical_current_value_does_not_warm_prototypes(self):
        from src.core.saving import history_intent as history_mod
        from src.services.api.controllers import retrieve as retrieve_mod

        with (
            _stubbed_adapters(retrieve_mod, _facts(1)) as (embeddings, _vs, _g, _d),
            patch.object(
                history_mod, "prototype_vectors", side_effect=AssertionError("warmup")
            ),
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="what is the current value",
                    brain_id="brain-a",
                    profile_stages=True,
                )
            )

        stages = {
            stage["stage"] for stage in (response.stage_timings or {}).get("stages", [])
        }
        self.assertIn("history.intent", stages)
        self.assertNotIn("history.prototypes", stages)
        embeddings.embed_texts.assert_called()

    async def test_similarity_path_batches_prototype_embeddings(self):
        from src.core.saving.history_intent import CURRENT_EXAMPLES, HISTORY_EXAMPLES
        from src.services.api.controllers import retrieve as retrieve_mod

        reset_prototype_cache()
        with _stubbed_adapters(retrieve_mod, _facts(1)) as (embeddings, _vs, _g, _d):
            await retrieve_mod.get_context(
                GetContextRequestBody(text="who is Alice", brain_id="brain-a")
            )

        batch_sizes = [
            len(call.args[0])
            for call in embeddings.embed_texts.call_args_list
            if call.args
        ]
        prototype_n = len(HISTORY_EXAMPLES) + len(CURRENT_EXAMPLES)
        self.assertTrue(any(size >= prototype_n for size in batch_sizes))
        self.assertLess(embeddings.embed_text.call_count, 5)

    async def test_explicit_history_flag_skips_prototype_warmup(self):
        from src.core.saving import history_intent as history_mod
        from src.services.api.controllers import retrieve as retrieve_mod

        with (
            _stubbed_adapters(retrieve_mod, _facts(1)),
            patch.object(
                history_mod, "prototype_vectors", side_effect=AssertionError("warmup")
            ),
        ):
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="who is Alice",
                    brain_id="brain-a",
                    include_history=False,
                    profile_stages=True,
                )
            )

        stages = {
            stage["stage"] for stage in (response.stage_timings or {}).get("stages", [])
        }
        self.assertIn("history.intent", stages)
        self.assertNotIn("history.prototypes", stages)


class ConcurrencyRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_seed_search_on_single_io_worker_does_not_deadlock(self):
        from concurrent.futures import ThreadPoolExecutor

        from src.services.api.controllers import retrieve as retrieve_mod

        node_hits = [
            Vector(
                id="n1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "a", "name": "Alice"},
                distance=0.2,
            )
        ]
        rel_hits = [
            Vector(
                id="rel1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "r1", "node_ids": ["tail", "mid"]},
                distance=0.05,
            )
        ]
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ctx-io")
        previous = dict(retrieve_mod._io_state)
        retrieve_mod._io_state["pool"] = pool
        retrieve_mod._io_state["pid"] = os.getpid()
        try:
            with _stubbed_adapters(retrieve_mod, _facts(1)) as (_e, mock_vs, _g, _d):
                mock_vs.search_nodes.return_value = node_hits
                mock_vs.search_relationships.return_value = rel_hits
                result = retrieve_mod._submit_io(
                    retrieve_mod._seed_nodes_for_text,
                    "Alice",
                    "brain-a",
                    embeddings=[0.1, 0.2, 0.3],
                ).result(timeout=5)
        finally:
            retrieve_mod._io_state.update(previous)
            pool.shutdown(wait=False)

        self.assertEqual({item[0] for item in result}, {"a", "mid", "tail"})

    async def test_parallel_seed_search_matches_serial_uuids(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        node_hits = [
            Vector(
                id="n1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "a", "name": "Alice"},
                distance=0.2,
            )
        ]
        rel_hits = [
            Vector(
                id="rel1",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"uuid": "r1", "node_ids": ["tail", "mid"]},
                distance=0.05,
            )
        ]
        with _stubbed_adapters(retrieve_mod, _facts(1)) as (_e, mock_vs, _g, _d):
            mock_vs.search_nodes.return_value = node_hits
            mock_vs.search_relationships.return_value = rel_hits
            first = retrieve_mod._seed_nodes_for_text(
                "Alice", "brain-a", embeddings=[0.1, 0.2, 0.3]
            )
            second = retrieve_mod._seed_nodes_for_text(
                "Alice", "brain-a", embeddings=[0.1, 0.2, 0.3]
            )

        self.assertEqual(first, second)
        self.assertEqual({item[0] for item in first}, {"a", "mid", "tail"})

    async def test_current_fact_candidates_runs_once_on_the_hot_path(self):
        from src.core.saving import source_passages as passages_mod
        from src.services.api.controllers import retrieve as retrieve_mod

        calls = {"n": 0}
        original = passages_mod.current_fact_candidates

        def wrapped(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        with (
            _stubbed_adapters(retrieve_mod, _facts(2)),
            patch.object(passages_mod, "current_fact_candidates", wrapped),
        ):
            await retrieve_mod.get_context(
                GetContextRequestBody(text="Alice", brain_id="brain-a")
            )

        self.assertEqual(calls["n"], 1)


class StageLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_debug_logs_stages_without_attaching_or_query_text(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        query = "secret-question-about-alice"
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        log = logging.getLogger("src.services.api.controllers.retrieve")
        previous = log.level
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        try:
            with _stubbed_adapters(retrieve_mod, _facts(1)):
                response = await retrieve_mod.get_context(
                    GetContextRequestBody(text=query, brain_id="brain-a")
                )
        finally:
            log.removeHandler(handler)
            log.setLevel(previous)

        self.assertIsNone(response.stage_timings)
        output = stream.getvalue()
        self.assertIn("stage timings", output)
        self.assertIn("history.intent", output)
        self.assertNotIn(query, output)
        self.assertNotIn("password", output.lower())
        self.assertNotIn("api-key", output.lower())


class ContextGetRouteTests(unittest.TestCase):
    def test_get_and_post_context_are_registered(self):
        from src.services.api.routes.retrieve import retrieve_router

        found = set()
        for route in retrieve_router.routes:
            path = getattr(route, "path", "")
            methods = getattr(route, "methods", None) or set()
            if path.endswith("/context"):
                found.update(methods)
        self.assertIn("GET", found)
        self.assertIn("POST", found)

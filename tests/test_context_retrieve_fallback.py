import contextlib
import os
import sys
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

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.constants.data import TextChunk
from src.constants.embeddings import Vector
from src.constants.kg import Node, Predicate
from src.core.saving.history_intent import reset_prototype_cache
from src.services.api.constants.requests import GetContextRequestBody
from src.services.api.dependencies import get_brain_id
from src.services.api.errors import install_error_handlers
from src.services.api.routes.retrieve import retrieve_router
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
def _stubbed_adapters(retrieve_mod, facts):
    mock_embeddings = MagicMock()
    mock_embeddings.embed_text.return_value = Vector(
        id="q", embeddings=[0.1, 0.2, 0.3], metadata={}
    )
    mock_embeddings.embed_texts.side_effect = lambda texts: [
        Vector(id=f"b{i}", embeddings=[0.1, 0.2, 0.3], metadata={})
        for i, _ in enumerate(texts)
    ]
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
    mock_vs.search_data.return_value = []
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
        yield mock_embeddings, mock_vs, mock_graph, mock_data

def _keyword_chunk():
    return TextChunk(id="kw-1", text="Alice is mentioned in keyword search")


def _client():
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(retrieve_router)
    app.dependency_overrides[get_brain_id] = lambda: "brain-a"
    return TestClient(app, raise_server_exceptions=True)


class VectorTimeoutFallbackTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        reset_prototype_cache()

    async def test_seed_vector_timeout_returns_keyword_results(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        chunk = _keyword_chunk()
        with _stubbed_adapters(retrieve_mod, _facts(1)) as (_emb, mock_vs, _g, mock_data):
            mock_vs.search_nodes.side_effect = TimeoutError("vector timeout")
            mock_vs.search_relationships.return_value = []
            mock_vs.search_data.return_value = []
            mock_data.search.return_value = SimpleNamespace(text_chunks=[chunk])
            mock_data.get_text_chunks_by_ids.return_value = ([chunk], [])
            with self.assertLogs(retrieve_mod.logger, level="WARNING") as logs:
                response = await retrieve_mod.get_context(
                    GetContextRequestBody(
                        text="who is Alice",
                        brain_id="brain-a",
                        include_history=False,
                    )
                )
        self.assertTrue(any("vector.search_nodes" in line for line in logs.output))
        self.assertTrue(any("brain-a" in line for line in logs.output))
        self.assertFalse(any("who is Alice" in line for line in logs.output))
        self.assertIn("vector.search_nodes", response.degraded or [])
        self.assertTrue(
            any("keyword search" in item for item in response.source_passages)
            or "keyword search" in response.text_context
        )

    async def test_dense_passage_timeout_returns_200(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        chunk = _keyword_chunk()
        with _stubbed_adapters(retrieve_mod, _facts(1)) as (_emb, mock_vs, _g, mock_data):
            mock_vs.search_data.side_effect = TimeoutError("dense timeout")
            mock_data.search.return_value = SimpleNamespace(text_chunks=[chunk])
            mock_data.get_text_chunks_by_ids.return_value = ([chunk], [])
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="who is Alice",
                    brain_id="brain-a",
                    include_history=False,
                )
            )
        self.assertIn("vector.search_data", response.degraded or [])
        self.assertTrue(response.triples or response.source_passages)

    async def test_all_backends_failing_returns_503(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(retrieve_mod, []) as (_emb, mock_vs, mock_graph, mock_data):
            mock_vs.search_nodes.side_effect = TimeoutError("nodes")
            mock_vs.search_relationships.side_effect = TimeoutError("rels")
            mock_vs.search_data.side_effect = ConnectionError("dense")
            mock_data.search.side_effect = ConnectionError("keyword")
            mock_data.search_bm25.side_effect = ConnectionError("bm25")
            mock_data.get_last_text_chunks.side_effect = ConnectionError("chunks")
            mock_data.get_last_structured_data.side_effect = ConnectionError("structured")
            mock_graph.get_event_centric_neighbors.side_effect = TimeoutError("graph")
            with self.assertRaises(HTTPException) as ctx:
                await retrieve_mod.get_context(
                    GetContextRequestBody(
                        text="who is Alice",
                        brain_id="brain-a",
                        include_history=False,
                    )
                )
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, "Retrieval backends unavailable")

    async def test_historical_backend_errors_return_503_when_all_else_fails(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(retrieve_mod, []) as (_emb, mock_vs, mock_graph, mock_data):
            mock_vs.search_nodes.side_effect = TimeoutError("nodes")
            mock_vs.search_relationships.side_effect = TimeoutError("rels")
            mock_vs.search_data.side_effect = ConnectionError("dense")
            mock_data.search.side_effect = ConnectionError("keyword")
            mock_data.search_bm25.side_effect = ConnectionError("bm25")
            mock_data.get_last_text_chunks.side_effect = ConnectionError("chunks")
            mock_data.get_last_structured_data.side_effect = ConnectionError("structured")
            mock_graph.get_event_centric_neighbors.side_effect = TimeoutError("graph")
            with self.assertRaises(HTTPException) as ctx:
                await retrieve_mod.get_context(
                    GetContextRequestBody(
                        text="who is Alice",
                        brain_id="brain-a",
                        include_history=True,
                    )
                )
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_neo4j_driver_error_degrades_graph_not_request(self):
        from neo4j.exceptions import ServiceUnavailable

        from src.services.api.controllers import retrieve as retrieve_mod

        chunk = _keyword_chunk()
        with _stubbed_adapters(retrieve_mod, _facts(1)) as (_emb, mock_vs, mock_graph, mock_data):
            mock_graph.get_event_centric_neighbors.side_effect = ServiceUnavailable(
                "graph down"
            )
            mock_data.search.return_value = SimpleNamespace(text_chunks=[chunk])
            mock_data.get_text_chunks_by_ids.return_value = ([chunk], [])
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="who is Alice",
                    brain_id="brain-a",
                    include_history=False,
                )
            )
        self.assertIn("graph.event_neighbors", response.degraded or [])
        self.assertTrue(
            any("keyword search" in item for item in response.source_passages)
            or "keyword search" in response.text_context
        )

    async def test_late_ok_does_not_clear_failed_stage(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        health = retrieve_mod._RetrievalHealth()
        health.start("vector.search_data")
        health.error("vector.search_data")
        health.ok("vector.search_data")
        self.assertEqual(health.failed_stages, ["vector.search_data"])
        self.assertTrue(health.all_failed())
        health.start("vector.search_data")
        health.ok("vector.search_data")
        self.assertIn("vector.search_data", health.failed_stages)
        self.assertFalse(health.all_failed())

    async def test_late_ok_before_error_does_not_count_as_success(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        health = retrieve_mod._RetrievalHealth()
        health.start("vector.search_data")
        health.ok("vector.search_data")
        health.error("vector.search_data")
        self.assertEqual(health.failed_stages, ["vector.search_data"])
        self.assertTrue(health.all_failed())

    def test_wait_all_keeps_completed_sibling_on_timeout(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from src.services.api.controllers import retrieve as retrieve_mod

        block = threading.Event()

        def done():
            return "kept"

        def hang():
            block.wait(30)
            return "late"

        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ctx-io")
        previous = dict(retrieve_mod._io_state)
        retrieve_mod._io_state["pool"] = pool
        retrieve_mod._io_state["pid"] = os.getpid()
        retrieve_mod._io_state["inflight"] = 0
        try:
            futures = [
                retrieve_mod._submit_io(done),
                retrieve_mod._submit_io(hang),
            ]
            results = retrieve_mod._wait_all(futures, timeout=0.2)
            self.assertEqual(results[0], "kept")
            self.assertIs(results[1], retrieve_mod._IO_FAILED)
        finally:
            block.set()
            retrieve_mod._io_state.update(previous)
            pool.shutdown(wait=False)

    def test_parallel_io_releases_unstarted_slot_on_partial_timeout(self):
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor

        from src.services.api.controllers import retrieve as retrieve_mod

        started = threading.Event()
        block = threading.Event()

        def done():
            return "kept"

        def hang():
            started.set()
            block.wait(30)
            return "late"

        def queued():
            return "queued"

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ctx-io")
        previous = dict(retrieve_mod._io_state)
        retrieve_mod._io_state["pool"] = pool
        retrieve_mod._io_state["pid"] = os.getpid()
        retrieve_mod._io_state["inflight"] = 0
        try:
            with patch.object(retrieve_mod, "_IO_TIMEOUT_SECONDS", 0.3):
                results = retrieve_mod._parallel_io([done, hang, queued])
            self.assertTrue(started.wait(2))
            self.assertEqual(results[0], "kept")
            self.assertIs(results[1], retrieve_mod._IO_FAILED)
            self.assertIs(results[2], retrieve_mod._IO_FAILED)
            self.assertEqual(retrieve_mod._io_state["inflight"], 1)
            block.set()
            deadline = time.time() + 2
            while retrieve_mod._io_state["inflight"] != 0 and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(retrieve_mod._io_state["inflight"], 0)
        finally:
            block.set()
            retrieve_mod._io_state.update(previous)
            pool.shutdown(wait=False)

    async def test_programming_error_is_not_swallowed(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(retrieve_mod, _facts(1)) as (_emb, mock_vs, _g, _d):
            mock_vs.search_nodes.side_effect = AttributeError("broken client")
            mock_vs.search_data.return_value = []
            with self.assertRaises(AttributeError):
                await retrieve_mod.get_context(
                    GetContextRequestBody(
                        text="who is Alice",
                        brain_id="brain-a",
                        include_history=False,
                    )
                )

    async def test_keyword_backend_error_degrades(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(retrieve_mod, _facts(1)) as (_emb, mock_vs, _g, mock_data):
            mock_vs.search_data.return_value = []
            mock_data.search.side_effect = ConnectionError("keyword down")
            response = await retrieve_mod.get_context(
                GetContextRequestBody(
                    text="who is Alice",
                    brain_id="brain-a",
                    include_history=False,
                )
            )
        self.assertIn("data.keyword_search", response.degraded or [])
        self.assertTrue(response.triples or response.source_passages)

    async def test_keyword_type_error_propagates(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(retrieve_mod, _facts(1)) as (_emb, mock_vs, _g, mock_data):
            mock_vs.search_data.return_value = []
            mock_data.search.side_effect = TypeError("keyword bug")
            with self.assertRaises(TypeError):
                await retrieve_mod.get_context(
                    GetContextRequestBody(
                        text="who is Alice",
                        brain_id="brain-a",
                        include_history=False,
                    )
                )

    async def test_keyword_value_error_propagates(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(retrieve_mod, _facts(1)) as (_emb, mock_vs, _g, mock_data):
            mock_vs.search_data.return_value = []
            mock_data.search.side_effect = ValueError("keyword bug")
            with self.assertRaises(ValueError):
                await retrieve_mod.get_context(
                    GetContextRequestBody(
                        text="who is Alice",
                        brain_id="brain-a",
                        include_history=False,
                    )
                )

    async def test_bm25_backend_error_degrades(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        chunk = _keyword_chunk()
        with _stubbed_adapters(retrieve_mod, _facts(1)) as (_emb, mock_vs, _g, mock_data):
            mock_vs.search_data.return_value = []
            mock_data.search_bm25.side_effect = TimeoutError("bm25 timeout")
            mock_data.get_text_chunks_by_ids.return_value = ([chunk], [])
            with patch.object(retrieve_mod.config, "search_enabled", True), patch.object(
                retrieve_mod.config, "context_passage_mode", "hybrid"
            ):
                response = await retrieve_mod.get_context(
                    GetContextRequestBody(
                        text="who is Alice",
                        brain_id="brain-a",
                        include_history=False,
                    )
                )
        self.assertIn("data.bm25_search", response.degraded or [])
        self.assertTrue(response.triples or response.source_passages)

    async def test_bm25_type_error_propagates(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        with _stubbed_adapters(retrieve_mod, _facts(1)) as (_emb, mock_vs, _g, mock_data):
            mock_vs.search_data.return_value = []
            mock_data.search_bm25.side_effect = TypeError("bm25 bug")
            with patch.object(retrieve_mod.config, "search_enabled", True), patch.object(
                retrieve_mod.config, "context_passage_mode", "hybrid"
            ):
                with self.assertRaises(TypeError):
                    await retrieve_mod.get_context(
                        GetContextRequestBody(
                            text="who is Alice",
                            brain_id="brain-a",
                            include_history=False,
                        )
                    )


class ContextFallbackHttpTests(unittest.TestCase):
    def tearDown(self):
        reset_prototype_cache()

    def test_seed_timeout_get_context_is_200(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        chunk = _keyword_chunk()
        client = _client()
        with _stubbed_adapters(retrieve_mod, _facts(1)) as (_emb, mock_vs, _g, mock_data):
            mock_vs.search_nodes.side_effect = TimeoutError("vector timeout")
            mock_vs.search_data.return_value = []
            mock_data.search.return_value = SimpleNamespace(text_chunks=[chunk])
            mock_data.get_text_chunks_by_ids.return_value = ([chunk], [])
            response = client.get(
                "/retrieve/context",
                params={"text": "who is Alice"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("vector.search_nodes", body.get("degraded") or [])
        self.assertTrue(
            any("keyword search" in item for item in body.get("source_passages") or [])
            or "keyword search" in (body.get("text_context") or "")
        )

    def test_dense_timeout_get_context_is_200(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        chunk = _keyword_chunk()
        client = _client()
        with _stubbed_adapters(retrieve_mod, _facts(1)) as (_emb, mock_vs, _g, mock_data):
            mock_vs.search_data.side_effect = TimeoutError("dense timeout")
            mock_data.search.return_value = SimpleNamespace(text_chunks=[chunk])
            mock_data.get_text_chunks_by_ids.return_value = ([chunk], [])
            response = client.get(
                "/retrieve/context",
                params={"text": "who is Alice"},
            )
        self.assertEqual(response.status_code, 200)

    def test_all_backends_failing_get_context_is_503(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        client = _client()
        with _stubbed_adapters(retrieve_mod, []) as (_emb, mock_vs, mock_graph, mock_data):
            mock_vs.search_nodes.side_effect = TimeoutError("nodes")
            mock_vs.search_relationships.side_effect = TimeoutError("rels")
            mock_vs.search_data.side_effect = ConnectionError("dense")
            mock_data.search.side_effect = ConnectionError("keyword")
            mock_data.search_bm25.side_effect = ConnectionError("bm25")
            mock_data.get_last_text_chunks.side_effect = ConnectionError("chunks")
            mock_data.get_last_structured_data.side_effect = ConnectionError("structured")
            mock_graph.get_event_centric_neighbors.side_effect = TimeoutError("graph")
            response = client.get(
                "/retrieve/context",
                params={"text": "who is Alice"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Retrieval backends unavailable")

    def test_keyword_backend_error_get_context_is_200(self):
        from src.services.api.controllers import retrieve as retrieve_mod

        client = _client()
        with _stubbed_adapters(retrieve_mod, _facts(1)) as (_emb, mock_vs, _g, mock_data):
            mock_vs.search_data.return_value = []
            mock_data.search.side_effect = ConnectionError("keyword down")
            response = client.get(
                "/retrieve/context",
                params={"text": "who is Alice"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("data.keyword_search", body.get("degraded") or [])


if __name__ == "__main__":
    unittest.main()

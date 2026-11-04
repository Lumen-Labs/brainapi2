import json
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
    "PIPELINE_MODE": "accurate",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)


class TaskStatusMonotonicTests(unittest.TestCase):
    def test_queued_cannot_overwrite_started_or_terminal(self):
        from src.workers.tasks import ingestion as ingestion_mod

        store = {}

        def fake_get(key, brain_id="default"):
            return store.get((key, brain_id))

        def fake_set(key, value, brain_id="default", expires_in=None):
            store[(key, brain_id)] = value
            return True

        with patch.object(ingestion_mod, "cache_adapter") as cache_adapter:
            cache_adapter.get.side_effect = fake_get
            cache_adapter.set.side_effect = fake_set

            ingestion_mod.set_ingestion_task_status("t1", "b1", "queued")
            ingestion_mod.set_ingestion_task_status("t1", "b1", "started", stage="source")
            ingestion_mod.set_ingestion_task_status("t1", "b1", "queued")
            started = json.loads(store[("task:t1", "b1")])
            self.assertEqual(started["status"], "started")

            ingestion_mod.set_ingestion_task_status("t1", "b1", "completed")
            ingestion_mod.set_ingestion_task_status("t1", "b1", "started")
            completed = json.loads(store[("task:t1", "b1")])
            self.assertEqual(completed["status"], "completed")

    def test_finalizer_writes_completed(self):
        from src.workers.tasks import ingestion as ingestion_mod

        store = {}

        def fake_get(key, brain_id="default"):
            return store.get((key, brain_id))

        def fake_set(key, value, brain_id="default", expires_in=None):
            store[(key, brain_id)] = value
            return True

        with patch.object(ingestion_mod, "cache_adapter") as cache_adapter:
            cache_adapter.get.side_effect = fake_get
            cache_adapter.set.side_effect = fake_set
            ingestion_mod.set_ingestion_task_status("t2", "b1", "persisting")
            result = ingestion_mod.finalize_ingestion_task.run.__func__(
                type("Bound", (), {"request": MagicMock(id="child")})(),
                "t2",
                "b1",
                "completed",
            )
        self.assertEqual(result, "t2")
        self.assertEqual(json.loads(store[("task:t2", "b1")])["status"], "completed")


class IngestApiLifecycleTests(unittest.TestCase):
    def test_ingest_records_queued_before_publish_and_returns_202(self):
        from fastapi.testclient import TestClient

        events = []

        def fake_set_status(task_id, brain_id, status, **kwargs):
            events.append(("status", status, task_id))
            return {"status": status, "task_id": task_id}

        def fake_apply_async(*args, **kwargs):
            events.append(("publish", kwargs.get("task_id")))
            return MagicMock(id=kwargs.get("task_id"))

        with patch(
            "src.services.api.routes.ingest.set_ingestion_task_status",
            side_effect=fake_set_status,
        ), patch(
            "src.services.api.routes.ingest.ingest_data_task.apply_async",
            side_effect=fake_apply_async,
        ):
            from src.services.api.routes.ingest import ingest_router
            from src.services.api.dependencies import get_brain_id
            from fastapi import FastAPI

            app = FastAPI()
            app.dependency_overrides[get_brain_id] = lambda: "brain-a"
            app.include_router(ingest_router)
            client = TestClient(app)
            response = client.post(
                "/ingest/",
                json={
                    "data": {"data_type": "text", "text_data": "hello"},
                },
            )

        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body["message"], "Ingestion accepted")
        self.assertIn("task_id", body)
        self.assertEqual(events[0][0], "status")
        self.assertEqual(events[0][1], "queued")
        self.assertEqual(events[1][0], "publish")
        self.assertEqual(events[0][2], events[1][1])


class TasksNotFoundTests(unittest.TestCase):
    def test_unknown_task_returns_not_found(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        with patch(
            "src.services.api.routes.tasks.cache_adapter"
        ) as cache_adapter:
            cache_adapter.get_task.return_value = None
            from src.services.api.routes.tasks import tasks_router
            from src.services.api.dependencies import get_brain_id
            from fastapi import FastAPI

            app = FastAPI()
            app.dependency_overrides[get_brain_id] = lambda: "brain-a"
            app.include_router(tasks_router)
            client = TestClient(app)
            response = client.get("/tasks/missing-id")

        self.assertEqual(response.status_code, 404)


class ParentCompletionChainTests(unittest.TestCase):
    def test_parent_does_not_mark_completed_before_chain(self):
        from src.core.saving.auto_kg import EnrichmentOrchestrationResult
        from src.workers.tasks import ingestion as ingestion_mod

        request = MagicMock()
        request.id = "parent-1"
        statuses = []

        def fake_set_status(task_id, brain_id, status, **kwargs):
            statuses.append(status)
            return {"status": status, "task_id": task_id}

        chained = []

        class FakeChain:
            def __init__(self, *steps):
                chained.extend(steps)

            def apply_async(self, **kwargs):
                self_error = kwargs.get("link_error")
                assert self_error is not None
                return MagicMock(id="chain-1")

        with patch.object(
            ingestion_mod, "set_ingestion_task_status", side_effect=fake_set_status
        ), patch.object(
            ingestion_mod,
            "enrich_kg_from_input",
            return_value=EnrichmentOrchestrationResult(
                session_id="s1",
                ingestion_session_id="i1",
                brain_id="brain-a",
                enrichment_relationships=[{"uuid": "r1"}],
                should_consolidate=False,
            ),
        ), patch.object(ingestion_mod, "data_adapter") as data_adapter, patch.object(
            ingestion_mod, "embeddings_adapter"
        ) as embeddings_adapter, patch.object(
            ingestion_mod, "vector_store_adapter"
        ), patch.object(
            ingestion_mod.config, "pipeline_mode", "lightweight"
        ), patch(
            "celery.chain", FakeChain
        ):
            from src.constants.embeddings import Vector

            data_adapter.save_text_chunk.side_effect = lambda chunk, brain_id="default": chunk
            embeddings_adapter.embed_text.return_value = Vector(
                id="v1", embeddings=[0.1], metadata={}
            )
            result = ingestion_mod.ingest_data.run.__func__(
                type("Bound", (), {"request": request})(),
                {
                    "data": {"data_type": "text", "text_data": "hello"},
                    "brain_id": "brain-a",
                },
            )

        self.assertEqual(result, "parent-1")
        self.assertNotIn("completed", statuses)
        self.assertTrue(chained)


if __name__ == "__main__":
    unittest.main()

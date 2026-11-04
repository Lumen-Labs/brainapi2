import json
import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError


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


COST_A = {
    "total_llm_tokens": 10,
    "total_input_tokens": 7,
    "total_output_tokens": 3,
    "total_cached_tokens": 1,
    "source_chars": 40,
    "source_tokens": 8,
    "stages": {
        "scout": {
            "input_tokens": 7,
            "output_tokens": 3,
            "cached_tokens": 1,
            "total_tokens": 10,
            "latency_ms": 2.0,
            "calls": 1,
        }
    },
}
COST_B = {
    "total_llm_tokens": 4,
    "total_input_tokens": 3,
    "total_output_tokens": 1,
    "total_cached_tokens": 0,
    "source_chars": 12,
    "source_tokens": 3,
    "stages": {
        "architect": {
            "input_tokens": 3,
            "output_tokens": 1,
            "cached_tokens": 0,
            "total_tokens": 4,
            "latency_ms": 5.0,
            "calls": 1,
        }
    },
}


class FakeIngestGroupClient:
    def __init__(self):
        self.credits = set()
        self.meters = {}
        self.calls = []

    def credit_ingest_group(
        self, brain_id, group_tracking, task_id, status, cost=None
    ):
        from src.core.saving.ingest_groups import apply_credit_to_meter, empty_group_meter

        self.calls.append((brain_id, group_tracking, task_id, status))
        token = (brain_id, group_tracking, task_id)
        if token in self.credits:
            return False
        self.credits.add(token)
        key = (brain_id, group_tracking)
        current = self.meters.get(key) or empty_group_meter(group_tracking)
        self.meters[key] = apply_credit_to_meter(
            current, task_id=task_id, status=status, cost=cost or {}
        )
        return True

    def get_ingest_group(self, brain_id, group_tracking):
        return self.meters.get((brain_id, group_tracking))


def _cache_store():
    store = {}

    def fake_get(key, brain_id="default"):
        return store.get((key, brain_id))

    def fake_set(key, value, brain_id="default", expires_in=None):
        store[(key, brain_id)] = value
        return True

    return store, fake_get, fake_set


def _minimal_triple():
    return {
        "event": {"name": "Purchase", "type": "EVENT"},
        "event_obj": {"name": "TARGETED"},
        "object": {"name": "Widget", "type": "PRODUCT"},
    }


class GroupTrackingRequestTests(unittest.TestCase):
    def test_valid_key_is_kept_and_empty_is_omitted(self):
        from src.constants.tasks.ingestion import IngestionTaskArgs

        payload = IngestionTaskArgs(
            data={"data_type": "text", "text_data": "hello"},
            brain_id="brain-a",
            group_tracking="chapter-3",
        )
        self.assertEqual(payload.group_tracking, "chapter-3")
        dumped = payload.model_dump()
        self.assertEqual(dumped["group_tracking"], "chapter-3")

        omitted = IngestionTaskArgs(
            data={"data_type": "text", "text_data": "hello"},
            brain_id="brain-a",
        )
        self.assertIsNone(omitted.group_tracking)

        blank = IngestionTaskArgs(
            data={"data_type": "text", "text_data": "hello"},
            brain_id="brain-a",
            group_tracking="  ",
        )
        self.assertIsNone(blank.group_tracking)

    def test_invalid_key_raises(self):
        from src.constants.tasks.ingestion import IngestionTaskArgs

        with self.assertRaises(ValidationError):
            IngestionTaskArgs(
                data={"data_type": "text", "text_data": "hello"},
                brain_id="brain-a",
                group_tracking="chapter 3",
            )
        with self.assertRaises(ValidationError):
            IngestionTaskArgs(
                data={"data_type": "text", "text_data": "hello"},
                brain_id="brain-a",
                group_tracking="x" * 129,
            )

    def test_structured_request_accepts_and_rejects_keys(self):
        from src.services.api.constants.requests import IngestionStructuredRequestBody

        body = IngestionStructuredRequestBody(
            data=[_minimal_triple()],
            group_tracking="chapter-3",
        )
        self.assertEqual(body.group_tracking, "chapter-3")
        with self.assertRaises(ValidationError):
            IngestionStructuredRequestBody(
                data=[_minimal_triple()],
                group_tracking="no spaces",
            )


class TaskStampTests(unittest.TestCase):
    def test_valid_key_is_stored_and_empty_is_omitted(self):
        from src.workers.tasks import ingestion as ingestion_mod

        store, fake_get, fake_set = _cache_store()
        with patch.object(ingestion_mod, "cache_adapter") as cache_adapter:
            cache_adapter.get.side_effect = fake_get
            cache_adapter.set.side_effect = fake_set
            ingestion_mod.set_ingestion_task_status(
                "t1",
                "brain-a",
                "queued",
                stage="queued",
                group_tracking="chapter-3",
            )
            ingestion_mod.set_ingestion_task_status(
                "t2",
                "brain-a",
                "queued",
                stage="queued",
                group_tracking="",
            )
            ingestion_mod.set_ingestion_task_status(
                "t3",
                "brain-a",
                "queued",
                stage="queued",
            )

        stamped = json.loads(store[("task:t1", "brain-a")])
        self.assertEqual(stamped["group_tracking"], "chapter-3")
        self.assertNotIn("null", json.dumps(stamped))
        omitted = json.loads(store[("task:t2", "brain-a")])
        self.assertNotIn("group_tracking", omitted)
        also_omitted = json.loads(store[("task:t3", "brain-a")])
        self.assertNotIn("group_tracking", also_omitted)

    def test_ingest_route_stamps_key_and_rejects_invalid(self):
        from src.services.api.dependencies import get_brain_id
        from src.services.api.routes.ingest import ingest_router

        events = []

        def fake_set_status(task_id, brain_id, status, **kwargs):
            events.append(kwargs.get("group_tracking"))
            return {"status": status, "task_id": task_id}

        app = FastAPI()
        app.dependency_overrides[get_brain_id] = lambda: "brain-a"
        app.include_router(ingest_router)
        client = TestClient(app)
        with patch(
            "src.services.api.routes.ingest.set_ingestion_task_status",
            side_effect=fake_set_status,
        ), patch(
            "src.services.api.routes.ingest.ingest_data_task.apply_async",
            return_value=MagicMock(),
        ):
            ok = client.post(
                "/ingest/",
                json={
                    "data": {"data_type": "text", "text_data": "hello"},
                    "group_tracking": "chapter-3",
                },
            )
            bad = client.post(
                "/ingest/",
                json={
                    "data": {"data_type": "text", "text_data": "hello"},
                    "group_tracking": "chapter 3",
                },
            )
        self.assertEqual(ok.status_code, 202)
        self.assertEqual(events[0], "chapter-3")
        self.assertEqual(bad.status_code, 422)


class MeterStoreTests(unittest.TestCase):
    def test_first_credit_inserts_second_is_noop_and_two_tasks_sum(self):
        from src.core.saving.ingest_groups import apply_credit_to_meter, empty_group_meter

        store = FakeIngestGroupClient()
        first = store.credit_ingest_group(
            "brain-a", "chapter-3", "t1", "completed", COST_A
        )
        second = store.credit_ingest_group(
            "brain-a", "chapter-3", "t1", "completed", COST_A
        )
        other = store.credit_ingest_group(
            "brain-a", "chapter-3", "t2", "failed", COST_B
        )
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertTrue(other)
        meter = store.get_ingest_group("brain-a", "chapter-3")
        self.assertEqual(meter["task_count"], 2)
        self.assertEqual(meter["completed"], 1)
        self.assertEqual(meter["failed"], 1)
        self.assertEqual(meter["total_llm_tokens"], 14)
        self.assertEqual(meter["source_chars"], 52)
        self.assertEqual(meter["stages"]["scout"]["total_tokens"], 10)
        self.assertEqual(meter["stages"]["architect"]["total_tokens"], 4)
        self.assertEqual(meter["task_ids"], ["t1", "t2"])
        self.assertIsNone(store.get_ingest_group("brain-a", "missing"))

        stacked = apply_credit_to_meter(
            apply_credit_to_meter(
                empty_group_meter("chapter-3"),
                task_id="t1",
                status="completed",
                cost=COST_A,
            ),
            task_id="t2",
            status="failed",
            cost=COST_B,
        )
        self.assertEqual(stacked["total_llm_tokens"], meter["total_llm_tokens"])
        self.assertEqual(stacked["source_tokens"], meter["source_tokens"])

    def test_task_ids_are_capped_at_fifty(self):
        from src.core.saving.ingest_groups import apply_credit_to_meter, empty_group_meter

        meter = empty_group_meter("chapter-3")
        for index in range(51):
            meter = apply_credit_to_meter(
                meter,
                task_id=f"t{index}",
                status="completed",
                cost=COST_A,
            )
        self.assertEqual(len(meter["task_ids"]), 50)
        self.assertEqual(meter["task_ids"][0], "t1")
        self.assertEqual(meter["task_ids"][-1], "t50")


class CreditHookTests(unittest.TestCase):
    def test_terminal_status_credits_once_and_skips_missing_key(self):
        from src.adapters.data import DataAdapter
        from src.workers.tasks import ingestion as ingestion_mod

        store, fake_get, fake_set = _cache_store()
        fake = FakeIngestGroupClient()
        adapter = DataAdapter()
        adapter.add_client(fake)

        with patch.object(ingestion_mod, "cache_adapter") as cache_adapter, patch(
            "src.services.data.main.data_adapter", adapter
        ):
            cache_adapter.get.side_effect = fake_get
            cache_adapter.set.side_effect = fake_set
            ingestion_mod.set_ingestion_task_status(
                "t1",
                "brain-a",
                "started",
                stage="source",
                group_tracking="chapter-3",
            )
            ingestion_mod.set_ingestion_task_status(
                "t1",
                "brain-a",
                "completed",
                cost=COST_A,
            )
            ingestion_mod.set_ingestion_task_status(
                "t1",
                "brain-a",
                "completed",
                cost=COST_A,
            )
            ingestion_mod.set_ingestion_task_status(
                "t-fail",
                "brain-a",
                "failed",
                group_tracking="chapter-3",
                cost=COST_B,
            )
            ingestion_mod.set_ingestion_task_status(
                "t-none",
                "brain-a",
                "completed",
                cost=COST_A,
            )

        meter = fake.get_ingest_group("brain-a", "chapter-3")
        self.assertEqual(meter["task_count"], 2)
        self.assertEqual(meter["completed"], 1)
        self.assertEqual(meter["failed"], 1)
        self.assertEqual(meter["total_llm_tokens"], 14)
        self.assertEqual([call[2] for call in fake.calls], ["t1", "t1", "t-fail"])
        self.assertIsNone(fake.get_ingest_group("brain-a", None))
        blob = json.loads(store[("task:t-none", "brain-a")])
        self.assertNotIn("group_tracking", blob)

    def test_reads_existing_blob_from_get_task_when_get_misses(self):
        from src.adapters.data import DataAdapter
        from src.workers.tasks import ingestion as ingestion_mod

        store = {}
        fake = FakeIngestGroupClient()
        adapter = DataAdapter()
        adapter.add_client(fake)

        def fake_get(key, brain_id="default"):
            return None

        def fake_get_task(task_id, brain_id="default"):
            return store.get(("task:" + task_id, brain_id))

        def fake_set(key, value, brain_id="default", expires_in=None):
            store[(key, brain_id)] = value
            return True

        with patch.object(ingestion_mod, "cache_adapter") as cache_adapter, patch(
            "src.services.data.main.data_adapter", adapter
        ):
            cache_adapter.get.side_effect = fake_get
            cache_adapter.get_task.side_effect = fake_get_task
            cache_adapter.set.side_effect = fake_set
            ingestion_mod.set_ingestion_task_status(
                "t-hash",
                "brain-a",
                "queued",
                stage="queued",
                group_tracking="chapter-3",
            )
            ingestion_mod.set_ingestion_task_status(
                "t-hash",
                "brain-a",
                "completed",
                cost=COST_A,
            )

        meter = fake.get_ingest_group("brain-a", "chapter-3")
        self.assertEqual(meter["task_count"], 1)
        self.assertEqual(meter["total_llm_tokens"], 10)
        blob = json.loads(store[("task:t-hash", "brain-a")])
        self.assertEqual(blob["group_tracking"], "chapter-3")


class IngestGroupGetTests(unittest.TestCase):
    def _client(self, brain_id="brain-a"):
        from src.services.api.dependencies import get_brain_id
        from src.services.api.routes.meta import meta_router

        app = FastAPI()
        app.include_router(meta_router)
        app.dependency_overrides[get_brain_id] = lambda: brain_id
        return TestClient(app)

    def test_returns_rollup_and_404s_unknown_or_other_brain(self):
        fake = FakeIngestGroupClient()
        fake.credit_ingest_group("brain-a", "chapter-3", "t1", "completed", COST_A)
        adapter = type("Adapter", (), {})()
        adapter.credit_ingest_group = fake.credit_ingest_group
        adapter.get_ingest_group = fake.get_ingest_group

        with patch("src.services.data.main.data_adapter", adapter):
            found = self._client("brain-a").get("/meta/ingest-groups/chapter-3", headers={"BrainPAT":os.environ["BRAINPAT_TOKEN"]})
            missing = self._client("brain-a").get("/meta/ingest-groups/other-key", headers={"BrainPAT":os.environ["BRAINPAT_TOKEN"]})
            other = self._client("brain-b").get("/meta/ingest-groups/chapter-3", headers={"BrainPAT":os.environ["BRAINPAT_TOKEN"]})
            invalid = self._client("brain-a").get("/meta/ingest-groups/nope!", headers={"BrainPAT":os.environ["BRAINPAT_TOKEN"]})

        self.assertEqual(found.status_code, 200)
        body = found.json()
        self.assertEqual(body["group_tracking"], "chapter-3")
        self.assertEqual(body["task_count"], 1)
        self.assertEqual(body["completed"], 1)
        self.assertEqual(body["total_llm_tokens"], 10)
        self.assertEqual(body["source_chars"], 40)
        self.assertEqual(body["task_ids"], ["t1"])
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(other.status_code, 404)
        self.assertEqual(invalid.status_code, 422)


class FileAndStructuredInheritTests(unittest.TestCase):
    def test_structured_route_passes_key(self):
        from src.services.api.dependencies import get_brain_id
        from src.services.api.routes.ingest import ingest_router

        captured = {}

        def fake_apply_async(args=None, task_id=None):
            captured["args"] = args
            captured["task_id"] = task_id
            return MagicMock()

        app = FastAPI()
        app.dependency_overrides[get_brain_id] = lambda: "brain-a"
        app.include_router(ingest_router)
        client = TestClient(app)
        with patch(
            "src.services.api.routes.ingest.set_ingestion_task_status",
            return_value={},
        ), patch(
            "src.services.api.routes.ingest.ingest_structured_data_task.apply_async",
            side_effect=fake_apply_async,
        ):
            response = client.post(
                "/ingest/structured",
                json={
                    "data": [_minimal_triple()],
                    "group_tracking": "chapter-3",
                },
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(captured["args"][0]["group_tracking"], "chapter-3")

    def test_file_page_payload_inherits_key_and_omits_when_unset(self):
        from src.workers.tasks.ingestion import build_file_page_ingest_payload

        with_key = build_file_page_ingest_payload(
            "brain-a", "# page one", group_tracking="chapter-3"
        )
        without_key = build_file_page_ingest_payload("brain-a", "# page one")
        self.assertEqual(with_key["group_tracking"], "chapter-3")
        self.assertEqual(with_key["brain_id"], "brain-a")
        self.assertEqual(with_key["data"]["text_data"], "# page one")
        self.assertNotIn("group_tracking", without_key)

    def test_file_route_forwards_form_and_query_key(self):
        from src.services.api.dependencies import get_brain_id
        from src.services.api.routes.ingest import ingest_router

        captured = {}

        def fake_apply_async(args=None, task_id=None):
            captured["args"] = args
            return MagicMock()

        app = FastAPI()
        app.dependency_overrides[get_brain_id] = lambda: "brain-a"
        app.include_router(ingest_router)
        client = TestClient(app)
        with patch(
            "src.services.api.routes.ingest.config"
        ) as config, patch(
            "src.services.api.routes.ingest.set_ingestion_task_status",
            return_value={},
        ), patch(
            "src.services.api.routes.ingest.ingest_file_task.apply_async",
            side_effect=fake_apply_async,
        ):
            config.ocr_mode = "docling"
            response = client.post(
                "/ingest/file?group_tracking=chapter-3",
                files={"file": ("book.pdf", b"%PDF-1.4", "application/pdf")},
            )
            bad = client.post(
                "/ingest/file?group_tracking=nope!",
                files={"file": ("book.pdf", b"%PDF-1.4", "application/pdf")},
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(captured["args"][3], "chapter-3")
        self.assertEqual(bad.status_code, 422)

    def test_url_route_stamps_key_and_rejects_invalid(self):
        from src.services.api.dependencies import get_brain_id
        from src.services.api.routes.ingest import ingest_router

        events = []
        captured = {}

        def fake_set_status(task_id, brain_id, status, **kwargs):
            events.append(kwargs.get("group_tracking"))
            return {"status": status, "task_id": task_id}

        def fake_apply_async(args=None, task_id=None):
            captured["args"] = args
            return MagicMock()

        app = FastAPI()
        app.dependency_overrides[get_brain_id] = lambda: "brain-a"
        app.include_router(ingest_router)
        client = TestClient(app)
        with patch(
            "src.services.api.routes.ingest.config"
        ) as config, patch(
            "src.services.api.routes.ingest.set_ingestion_task_status",
            side_effect=fake_set_status,
        ), patch(
            "src.services.api.routes.ingest.ingest_url_task.apply_async",
            side_effect=fake_apply_async,
        ):
            config.crawler_is_configured = True
            ok = client.post(
                "/ingest/url",
                json={
                    "url": "https://example.com",
                    "group_tracking": "chapter-3",
                },
            )
            bad = client.post(
                "/ingest/url",
                json={
                    "url": "https://example.com",
                    "group_tracking": "chapter 3",
                },
            )
        self.assertEqual(ok.status_code, 202)
        self.assertEqual(events[0], "chapter-3")
        self.assertEqual(captured["args"][0]["group_tracking"], "chapter-3")
        self.assertEqual(bad.status_code, 422)


if __name__ == "__main__":
    unittest.main()

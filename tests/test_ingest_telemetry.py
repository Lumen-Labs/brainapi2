import json
import os
import unittest
from unittest.mock import MagicMock, patch

from celery.exceptions import Ignore

from src.lib.tracing.ingest_telemetry import (
    INGEST_DATA_OPERATION,
    INGESTION_EVENT,
    INGESTION_OUTCOMES,
    PROPERTY_KEYS,
    RUNTIME_EVENT,
    RUNTIME_OUTCOMES,
    captured_events,
    disable_event_capture,
    enable_event_capture,
    kind_for_ingest_data,
    properties_from_status,
    reset_captured_events,
    set_capture_transport,
    telemetry_extra,
)


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
    "POSTHOG_ENABLED": "false",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)
os.environ["POSTHOG_ENABLED"] = "false"


def _install_test_capture():
    enable_event_capture()
    set_capture_transport(lambda _record: None)
    reset_captured_events()


def _uninstall_test_capture():
    set_capture_transport(None)
    disable_event_capture()


SECRET_TEXT = "patient-secret-document-body"


def _cache_store():
    store = {}

    def fake_get(key, brain_id="default"):
        return store.get((key, brain_id))

    def fake_set(key, value, brain_id="default", expires_in=None):
        store[(key, brain_id)] = value
        return True

    return store, fake_get, fake_set


def _events_named(name):
    return [event for event in captured_events() if event["event"] == name]


def _assert_metric_properties(properties, *, kind, outcome, pipeline_mode="accurate"):
    assert isinstance(properties["duration_ms"], int)
    assert properties["duration_ms"] >= 0
    assert properties["kind"] == kind
    assert properties["pipeline_mode"] == pipeline_mode
    assert properties["outcome"] == outcome
    assert isinstance(properties["size_chars"], int)
    assert properties["size_chars"] >= 0
    assert isinstance(properties["size_pages"], int)
    assert properties["size_pages"] >= 0
    assert isinstance(properties["llm_call_count"], int)
    assert properties["llm_call_count"] >= 0
    assert isinstance(properties["prompt_tokens"], int)
    assert isinstance(properties["completion_tokens"], int)
    assert isinstance(properties["total_tokens"], int)
    assert properties["total_tokens"] == (
        properties["prompt_tokens"] + properties["completion_tokens"]
    )
    assert isinstance(properties["unit_count"], int)
    assert properties["unit_count"] >= 0
    assert SECRET_TEXT not in json.dumps(properties)
    assert "error" not in properties
    assert "text" not in properties
    assert "text_data" not in properties
    assert set(properties).issubset(PROPERTY_KEYS)


class IngestTelemetryHelperTests(unittest.TestCase):
    def test_kind_for_ingest_data_uses_payload_and_data_type(self):
        from src.constants.tasks.ingestion import IngestionTaskArgs

        text = IngestionTaskArgs(
            data={"data_type": "text", "text_data": "hello"},
            brain_id="b1",
        )
        file_page = IngestionTaskArgs(
            data={"data_type": "text", "text_data": "page"},
            brain_id="b1",
            kind="file",
        )
        json_payload = IngestionTaskArgs(
            data={"data_type": "json", "json_data": {"a": 1}},
            brain_id="b1",
        )
        self.assertEqual(kind_for_ingest_data(text), "text")
        self.assertEqual(kind_for_ingest_data(file_page), "file")
        self.assertEqual(kind_for_ingest_data(json_payload), "json")

    def test_properties_include_cost_tokens_and_duration(self):
        props = properties_from_status(
            {
                "started_at_ms": 1,
                "kind": "text",
                "pipeline_mode": "lightweight",
                "size_chars": 1200,
                "size_pages": 0,
                "cost": {
                    "source_chars": 1200,
                    "architect_units": 3,
                    "stages": {
                        "scout": {"input_tokens": 10, "output_tokens": 2, "calls": 1},
                        "architect": {
                            "input_tokens": 40,
                            "output_tokens": 8,
                            "calls": 2,
                        },
                        "embed": {"input_tokens": 99, "output_tokens": 0, "calls": 1},
                    },
                },
            }
        )
        self.assertGreaterEqual(props["duration_ms"], 0)
        self.assertIsInstance(props["duration_ms"], int)
        self.assertEqual(props["kind"], "text")
        self.assertEqual(props["pipeline_mode"], "lightweight")
        self.assertEqual(props["size_chars"], 1200)
        self.assertEqual(props["llm_call_count"], 3)
        self.assertEqual(props["prompt_tokens"], 50)
        self.assertEqual(props["completion_tokens"], 10)
        self.assertEqual(props["total_tokens"], 60)
        self.assertEqual(props["unit_count"], 3)
        self.assertNotIn("embed", json.dumps(props))

    def test_capture_does_not_retain_without_enable(self):
        from src.lib.tracing.ingest_telemetry import capture_event

        disable_event_capture()
        capture_event(RUNTIME_EVENT, {"duration_ms": 1, "outcome": "succeeded"})
        self.assertEqual(captured_events(), [])


class IngestStatusTelemetryTests(unittest.TestCase):
    def setUp(self):
        _install_test_capture()

    def tearDown(self):
        _uninstall_test_capture()

    def test_completed_emits_runtime_and_ingestion_events(self):
        from src.workers.tasks import ingestion as ingestion_mod

        store, fake_get, fake_set = _cache_store()
        extra = telemetry_extra(
            operation=INGEST_DATA_OPERATION,
            kind="text",
            pipeline_mode="accurate",
            started_at_ms=1,
            size_chars=len(SECRET_TEXT),
            unit_count=1,
        )
        extra["cost"] = {
            "architect_units": 1,
            "stages": {
                "architect": {"input_tokens": 8, "output_tokens": 2, "calls": 1},
            },
        }
        with patch.object(ingestion_mod, "cache_adapter") as cache_adapter:
            cache_adapter.get.side_effect = fake_get
            cache_adapter.set.side_effect = fake_set
            ingestion_mod.set_ingestion_task_status(
                "t-success",
                "brain-a",
                "started",
                stage="source",
                extra=extra,
            )
            ingestion_mod.set_ingestion_task_status(
                "t-success",
                "brain-a",
                "completed",
                stage="completed",
            )

        runtime = _events_named(RUNTIME_EVENT)
        ingested = _events_named(INGESTION_EVENT)
        self.assertEqual(len(runtime), 1)
        self.assertEqual(len(ingested), 1)
        self.assertEqual(runtime[0]["properties"]["operation"], INGEST_DATA_OPERATION)
        _assert_metric_properties(
            runtime[0]["properties"],
            kind="text",
            outcome=RUNTIME_OUTCOMES["completed"],
        )
        _assert_metric_properties(
            ingested[0]["properties"],
            kind="text",
            outcome=INGESTION_OUTCOMES["completed"],
        )
        self.assertEqual(runtime[0]["properties"]["size_chars"], len(SECRET_TEXT))
        self.assertEqual(runtime[0]["properties"]["prompt_tokens"], 8)
        self.assertEqual(runtime[0]["properties"]["llm_call_count"], 1)
        payload = json.loads(store[("task:t-success", "brain-a")])
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["task_id"], "t-success")

    def test_failed_emits_duration_and_keeps_event_names(self):
        from src.workers.tasks import ingestion as ingestion_mod

        extra = telemetry_extra(
            operation=INGEST_DATA_OPERATION,
            kind="file",
            pipeline_mode="lightweight",
            started_at_ms=1,
            size_chars=40,
            size_pages=1,
        )
        extra["cost"] = {
            "stages": {
                "observations": {"input_tokens": 12, "output_tokens": 3, "calls": 1},
            },
        }
        with patch.object(ingestion_mod, "cache_adapter") as cache_adapter:
            store, fake_get, fake_set = _cache_store()
            cache_adapter.get.side_effect = fake_get
            cache_adapter.set.side_effect = fake_set
            ingestion_mod.set_ingestion_task_status(
                "t-fail",
                "brain-a",
                "started",
                extra=extra,
            )
            ingestion_mod.set_ingestion_task_status(
                "t-fail",
                "brain-a",
                "failed",
                error=SECRET_TEXT,
            )

        runtime = _events_named(RUNTIME_EVENT)
        ingested = _events_named(INGESTION_EVENT)
        self.assertEqual(len(runtime), 1)
        self.assertEqual(runtime[0]["event"], RUNTIME_EVENT)
        self.assertEqual(ingested[0]["event"], INGESTION_EVENT)
        _assert_metric_properties(
            runtime[0]["properties"],
            kind="file",
            outcome=RUNTIME_OUTCOMES["failed"],
            pipeline_mode="lightweight",
        )
        _assert_metric_properties(
            ingested[0]["properties"],
            kind="file",
            outcome=INGESTION_OUTCOMES["failed"],
            pipeline_mode="lightweight",
        )
        self.assertEqual(runtime[0]["properties"]["size_pages"], 1)
        self.assertEqual(runtime[0]["properties"]["llm_call_count"], 1)
        self.assertEqual(runtime[0]["properties"]["prompt_tokens"], 12)
        self.assertNotIn(SECRET_TEXT, json.dumps(captured_events()))

    def test_partial_failed_uses_partial_outcome_once(self):
        from src.workers.tasks import ingestion as ingestion_mod

        extra = telemetry_extra(
            operation=INGEST_DATA_OPERATION,
            kind="text",
            pipeline_mode="accurate",
            started_at_ms=1,
        )
        with patch.object(ingestion_mod, "cache_adapter") as cache_adapter:
            _, fake_get, fake_set = _cache_store()
            cache_adapter.get.side_effect = fake_get
            cache_adapter.set.side_effect = fake_set
            ingestion_mod.set_ingestion_task_status(
                "t-partial", "brain-a", "started", extra=extra
            )
            ingestion_mod.set_ingestion_task_status(
                "t-partial", "brain-a", "partial_failed"
            )
            ingestion_mod.set_ingestion_task_status(
                "t-partial", "brain-a", "failed", error="later"
            )

        runtime = _events_named(RUNTIME_EVENT)
        self.assertEqual(len(runtime), 1)
        self.assertEqual(runtime[0]["properties"]["outcome"], "partial")
        self.assertEqual(_events_named(INGESTION_EVENT)[0]["properties"]["outcome"], "partial")


class IngestDataTaskTelemetryTests(unittest.TestCase):
    def setUp(self):
        _install_test_capture()

    def tearDown(self):
        _uninstall_test_capture()

    def _run_ingest(self, args, pipeline_mode="accurate", fail=None):
        from src.constants.embeddings import Vector
        from src.workers.tasks import ingestion as ingestion_mod

        request = MagicMock()
        request.id = "task-123"
        store, fake_get, fake_set = _cache_store()
        with (
            patch.object(ingestion_mod, "data_adapter") as data_adapter,
            patch.object(ingestion_mod, "embeddings_adapter") as embeddings_adapter,
            patch.object(ingestion_mod, "vector_store_adapter") as vector_store_adapter,
            patch.object(ingestion_mod, "cache_adapter") as cache_adapter,
            patch.object(ingestion_mod.config, "pipeline_mode", pipeline_mode),
            patch("src.core.saving.ingest_effects.remember"),
        ):
            cache_adapter.get.side_effect = fake_get
            cache_adapter.set.side_effect = fake_set
            data_adapter.save_text_chunk.side_effect = lambda chunk, brain_id="default": chunk
            embeddings_adapter.embed_text.return_value = Vector(
                id="v1", embeddings=[0.1], metadata={}
            )
            vector_store_adapter.add_vectors.return_value = ["v1"]
            if fail is not None:
                data_adapter.save_text_chunk.side_effect = fail
            result = ingestion_mod.ingest_data.run.__func__(
                type("Bound", (), {"request": request})(),
                args,
            )
        return result, store

    def test_skip_enrichment_success_emits_duration_and_size(self):
        args = {
            "data": {"data_type": "text", "text_data": SECRET_TEXT},
            "brain_id": "brain-a",
            "skip_enrichment": True,
        }
        result, store = self._run_ingest(args, pipeline_mode="lightweight")
        self.assertEqual(result, "task-123")
        runtime = _events_named(RUNTIME_EVENT)
        ingested = _events_named(INGESTION_EVENT)
        self.assertEqual(len(runtime), 1)
        self.assertEqual(len(ingested), 1)
        props = runtime[0]["properties"]
        self.assertEqual(props["operation"], INGEST_DATA_OPERATION)
        _assert_metric_properties(
            props,
            kind="text",
            outcome="succeeded",
            pipeline_mode="lightweight",
        )
        self.assertEqual(props["size_chars"], len(SECRET_TEXT))
        _assert_metric_properties(
            ingested[0]["properties"],
            kind="text",
            outcome="done",
            pipeline_mode="lightweight",
        )
        self.assertEqual(json.loads(store[("task:task-123", "brain-a")])["status"], "completed")

    def test_worker_failure_emits_failed_outcome(self):
        args = {
            "data": {"data_type": "text", "text_data": SECRET_TEXT},
            "brain_id": "brain-a",
            "skip_enrichment": True,
        }
        with self.assertRaises(RuntimeError):
            self._run_ingest(args, fail=RuntimeError("disk full"))
        runtime = _events_named(RUNTIME_EVENT)
        self.assertEqual(len(runtime), 1)
        self.assertEqual(runtime[0]["properties"]["operation"], INGEST_DATA_OPERATION)
        _assert_metric_properties(
            runtime[0]["properties"],
            kind="text",
            outcome="failed",
        )
        self.assertEqual(
            _events_named(INGESTION_EVENT)[0]["properties"]["outcome"],
            "failed",
        )
        self.assertNotIn(SECRET_TEXT, json.dumps(captured_events()))

    def test_cancelled_ignore_is_not_timed(self):
        from src.workers.tasks import ingestion as ingestion_mod

        request = MagicMock()
        request.id = "task-cancel"
        store, fake_get, fake_set = _cache_store()
        args = {
            "data": {"data_type": "text", "text_data": SECRET_TEXT},
            "brain_id": "brain-a",
        }
        with (
            patch.object(ingestion_mod, "cache_adapter") as cache_adapter,
            patch.object(ingestion_mod, "checkpoint", side_effect=Ignore()),
            patch.object(ingestion_mod.config, "pipeline_mode", "accurate"),
        ):
            cache_adapter.get.side_effect = fake_get
            cache_adapter.set.side_effect = fake_set
            with self.assertRaises(Ignore):
                ingestion_mod.ingest_data.run.__func__(
                    type("Bound", (), {"request": request})(),
                    args,
                )
        self.assertEqual(captured_events(), [])
        self.assertEqual(json.loads(store[("task:task-cancel", "brain-a")])["status"], "started")

    def test_file_page_payload_sets_kind_without_body_in_telemetry_schema(self):
        from src.workers.tasks.ingestion import build_file_page_ingest_payload

        payload = build_file_page_ingest_payload("brain-a", SECRET_TEXT)
        self.assertEqual(payload["kind"], "file")
        extra = telemetry_extra(
            operation=INGEST_DATA_OPERATION,
            kind=payload["kind"],
            pipeline_mode="accurate",
            size_chars=len(payload["data"]["text_data"]),
            size_pages=1,
        )
        self.assertNotIn("text_data", extra)
        self.assertEqual(extra["kind"], "file")


if __name__ == "__main__":
    unittest.main()

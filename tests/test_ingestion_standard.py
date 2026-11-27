import ast
import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parent.parent


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


class SourceTextHelperTests(unittest.TestCase):
    def test_source_text_from_text_payload(self):
        from src.constants.tasks.ingestion import IngestionTaskArgs
        from src.workers.tasks.ingestion import source_text_from_payload

        payload = IngestionTaskArgs(
            data={"data_type": "text", "text_data": "hello world"},
            brain_id="brain1",
        )
        self.assertFalse(payload.skip_enrichment)
        skipped = IngestionTaskArgs(
            data={"data_type": "text", "text_data": "catalog row"},
            brain_id="brain1",
            skip_enrichment=True,
        )
        self.assertTrue(skipped.skip_enrichment)
        self.assertEqual(source_text_from_payload(payload), "hello world")

    def test_source_text_from_json_payload(self):
        from src.constants.tasks.ingestion import IngestionTaskArgs
        from src.workers.tasks.ingestion import source_text_from_payload

        payload = IngestionTaskArgs(
            data={"data_type": "json", "json_data": {"a": 1, "b": "x"}},
            brain_id="brain1",
        )
        self.assertEqual(
            source_text_from_payload(payload),
            json.dumps({"a": 1, "b": "x"}),
        )


class PipelineModeValidationTests(unittest.TestCase):
    def test_invalid_pipeline_mode_raises(self):
        from src.config import validate_pipeline_mode

        with self.assertRaises(ValueError):
            validate_pipeline_mode(None)
        with self.assertRaises(ValueError):
            validate_pipeline_mode("turbo")
        self.assertEqual(validate_pipeline_mode("accurate"), "accurate")
        self.assertEqual(validate_pipeline_mode("lightweight"), "lightweight")


class ConsolidateGraphAsyncReturnTests(unittest.TestCase):
    def test_consolidate_graph_async_has_no_undefined_consolidation_response(self):
        source = (ROOT / "src/workers/tasks/ingestion.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "consolidation_response":
                self.fail(
                    "consolidation_response is still referenced in ingestion.py"
                )


class IngestDataObservationAndEnrichmentTests(unittest.TestCase):
    def _make_request(self):
        request = MagicMock()
        request.id = "task-123"
        return request

    def test_json_ingest_passes_serialized_text_to_enrichment_and_observations(self):
        from src.constants.data import TextChunk
        from src.constants.embeddings import Vector
        from src.workers.tasks import ingestion as ingestion_mod

        saved_observations = []
        enrich_calls = []
        observe_calls = []

        def fake_save_text_chunk(chunk, brain_id="default"):
            return chunk

        def fake_save_observations(observations, brain_id="default"):
            saved_observations.append({"observations": observations, "brain_id": brain_id})
            return observations

        def fake_observe(*, text, observate_for, context=None):
            observe_calls.append(
                {"text": text, "observate_for": observate_for, "context": context}
            )
            return ["obs-1"]

        def fake_enrich(text, targeting=None, brain_id="default", **kwargs):
            from src.core.saving.auto_kg import EnrichmentOrchestrationResult

            enrich_calls.append({"text": text, "brain_id": brain_id})
            return EnrichmentOrchestrationResult(
                session_id=None,
                ingestion_session_id="ingest-1",
                brain_id=brain_id,
                enrichment_relationships=[],
                should_consolidate=False,
            )

        task = ingestion_mod.ingest_data
        request = self._make_request()
        args = {
            "data": {"data_type": "json", "json_data": {"name": "Alice"}},
            "brain_id": "tenant-a",
            "observate_for": ["people"],
        }

        with (
            patch.object(ingestion_mod, "data_adapter") as data_adapter,
            patch.object(ingestion_mod, "embeddings_adapter") as embeddings_adapter,
            patch.object(ingestion_mod, "vector_store_adapter") as vector_store_adapter,
            patch.object(ingestion_mod, "cache_adapter"),
            patch.object(ingestion_mod, "observations_agent") as observations_agent,
            patch.object(ingestion_mod, "enrich_kg_from_input", side_effect=fake_enrich),
            patch.object(ingestion_mod.config, "pipeline_mode", "accurate"),
            patch.object(ingestion_mod.config, "run_observations", True),
            patch.object(ingestion_mod, "set_ingestion_task_status"),
            patch("celery.chain") as chain_mock,
        ):
            chain_mock.return_value.apply_async = MagicMock()
            data_adapter.get_text_chunks.return_value = ([], 0)
            data_adapter.save_text_chunk.side_effect = fake_save_text_chunk
            data_adapter.save_observations.side_effect = fake_save_observations
            embeddings_adapter.embed_text.return_value = Vector(
                id="v1", embeddings=[0.1, 0.2], metadata={}
            )
            observations_agent.observe.side_effect = fake_observe
            vector_store_adapter.add_vectors.return_value = ["v1"]

            result = task.run.__func__(
                type("Bound", (), {"request": request})(),
                args,
            )

        expected_text = json.dumps({"name": "Alice"})
        self.assertEqual(result, "task-123")
        self.assertEqual(len(observe_calls), 1)
        self.assertEqual(observe_calls[0]["text"], expected_text)
        self.assertIsNone(observe_calls[0]["context"])
        self.assertEqual(len(saved_observations), 1)
        self.assertEqual(saved_observations[0]["brain_id"], "tenant-a")
        self.assertEqual(len(enrich_calls), 1)
        self.assertEqual(enrich_calls[0]["text"], expected_text)
        self.assertEqual(enrich_calls[0]["brain_id"], "tenant-a")

    def test_skip_enrichment_writes_chunk_without_kg(self):
        from src.constants.embeddings import Vector
        from src.workers.tasks import ingestion as ingestion_mod

        enrich_calls = []
        observe_calls = []

        def fake_save_text_chunk(chunk, brain_id="default"):
            return chunk

        def fake_enrich(*args, **kwargs):
            enrich_calls.append(kwargs)
            raise AssertionError("enrichment should be skipped")

        def fake_observe(**kwargs):
            observe_calls.append(kwargs)
            raise AssertionError("observations should be skipped")

        task = ingestion_mod.ingest_data
        request = self._make_request()
        args = {
            "data": {"data_type": "text", "text_data": "DOCID p1. Title: kettle"},
            "brain_id": "searchbenchesci",
            "skip_enrichment": True,
        }

        with (
            patch.object(ingestion_mod, "data_adapter") as data_adapter,
            patch.object(ingestion_mod, "embeddings_adapter") as embeddings_adapter,
            patch.object(ingestion_mod, "vector_store_adapter") as vector_store_adapter,
            patch.object(ingestion_mod, "cache_adapter"),
            patch.object(ingestion_mod, "observations_agent") as observations_agent,
            patch.object(ingestion_mod, "enrich_kg_from_input", side_effect=fake_enrich),
            patch.object(ingestion_mod.config, "pipeline_mode", "accurate"),
            patch.object(ingestion_mod, "set_ingestion_task_status") as set_status,
        ):
            data_adapter.get_text_chunks.return_value = ([], 0)
            data_adapter.save_text_chunk.side_effect = fake_save_text_chunk
            embeddings_adapter.embed_text.return_value = Vector(
                id="v1", embeddings=[0.1, 0.2], metadata={}
            )
            observations_agent.observe.side_effect = fake_observe
            vector_store_adapter.add_vectors.return_value = ["v1"]
            result = task.run.__func__(
                type("Bound", (), {"request": request})(),
                args,
            )

        self.assertEqual(result, "task-123")
        self.assertEqual(observe_calls, [])
        self.assertEqual(enrich_calls, [])
        self.assertTrue(data_adapter.save_text_chunk.called)
        self.assertTrue(embeddings_adapter.embed_text.called)
        completed = [
            call
            for call in set_status.call_args_list
            if call.args and call.args[2] == "completed"
        ]
        self.assertTrue(completed)

    def test_ingest_commit_hooks_run_after_chain_finalize(self):
        from src.constants.embeddings import Vector
        from src.core.saving.auto_kg import EnrichmentOrchestrationResult
        from src.workers.tasks import ingestion as ingestion_mod

        order = []

        def fake_save_text_chunk(chunk, brain_id="default"):
            order.append("save")
            chunk.id = "new-chunk"
            return chunk

        def fake_enrich(text, targeting=None, brain_id="default", **kwargs):
            order.append("enrich")
            return EnrichmentOrchestrationResult(
                session_id=None,
                ingestion_session_id="ingest-1",
                brain_id=brain_id,
                enrichment_relationships=[{"name": "FOUNDED_IN"}],
                should_consolidate=False,
            )

        def fake_hook(payload):
            order.append("commit")

        task = ingestion_mod.ingest_data
        request = self._make_request()
        args = {
            "data": {"data_type": "text", "text_data": "Halvorsen Dynamics"},
            "brain_id": "tenant-a",
        }

        with (
            patch.object(ingestion_mod, "data_adapter") as data_adapter,
            patch.object(ingestion_mod, "embeddings_adapter") as embeddings_adapter,
            patch.object(ingestion_mod, "vector_store_adapter") as vector_store_adapter,
            patch.object(ingestion_mod, "cache_adapter"),
            patch.object(ingestion_mod, "enrich_kg_from_input", side_effect=fake_enrich),
            patch.object(ingestion_mod, "_run_ingest_commit_hooks", side_effect=fake_hook),
            patch.object(ingestion_mod.config, "pipeline_mode", "accurate"),
            patch.object(ingestion_mod.config, "run_observations", False),
            patch.object(ingestion_mod, "set_ingestion_task_status"),
            patch("celery.chain") as chain_mock,
        ):
            chain_mock.return_value.apply_async = MagicMock()
            data_adapter.get_text_chunks.return_value = ([], 0)
            data_adapter.save_text_chunk.side_effect = fake_save_text_chunk
            embeddings_adapter.embed_text.return_value = Vector(
                id="v1", embeddings=[0.1, 0.2], metadata={}
            )
            vector_store_adapter.add_vectors.return_value = ["v1"]
            result = task.run.__func__(
                type("Bound", (), {"request": request})(),
                args,
            )

        self.assertEqual(result, "task-123")
        self.assertEqual(order, ["save", "enrich"])
        self.assertTrue(chain_mock.called)
        finalize = ingestion_mod.finalize_ingestion_task
        finalize_request = MagicMock()
        finalize_request.id = "finalize-1"
        with (
            patch.object(ingestion_mod, "graph_adapter"),
            patch.object(ingestion_mod, "set_ingestion_task_status"),
            patch.object(ingestion_mod, "_run_ingest_commit_hooks", side_effect=fake_hook),
            patch.object(ingestion_mod, "data_adapter") as data_adapter,
        ):
            finalize.run.__func__(
                type("Bound", (), {"request": finalize_request})(),
                "task-123",
                "tenant-a",
                "completed",
                None,
                None,
                None,
                None,
                {"brain_id": "tenant-a", "meta_keys": {"source_id": "note-1"}},
                ["old-chunk"],
                "new-chunk",
            )
        self.assertEqual(order, ["save", "enrich", "commit"])
        data_adapter.mark_text_chunks_superseded.assert_called_once()
        self.assertEqual(
            data_adapter.mark_text_chunks_superseded.call_args.args[0], ["old-chunk"]
        )

    def test_ingest_commit_hooks_skip_when_extraction_fails(self):
        from src.constants.embeddings import Vector
        from src.workers.tasks import ingestion as ingestion_mod

        hooks = []

        saved = []

        def fake_save_text_chunk(chunk, brain_id="default"):
            chunk.id = "new-chunk"
            saved.append(chunk.id)
            return chunk

        task = ingestion_mod.ingest_data
        request = self._make_request()
        args = {
            "data": {"data_type": "text", "text_data": "Halvorsen Dynamics"},
            "brain_id": "tenant-a",
            "meta_keys": {"source_id": "offer-sheet", "source_key": "offer-sheet"},
        }

        with (
            patch.object(ingestion_mod, "data_adapter") as data_adapter,
            patch.object(ingestion_mod, "embeddings_adapter") as embeddings_adapter,
            patch.object(ingestion_mod, "vector_store_adapter") as vector_store_adapter,
            patch.object(ingestion_mod, "cache_adapter"),
            patch.object(
                ingestion_mod,
                "enrich_kg_from_input",
                side_effect=RuntimeError("extract failed"),
            ),
            patch.object(
                ingestion_mod,
                "_run_ingest_commit_hooks",
                side_effect=lambda payload: hooks.append(payload),
            ),
            patch.object(ingestion_mod.config, "pipeline_mode", "accurate"),
            patch.object(ingestion_mod.config, "run_observations", False),
            patch.object(ingestion_mod, "set_ingestion_task_status"),
        ):
            data_adapter.save_text_chunk.side_effect = fake_save_text_chunk
            data_adapter.get_text_chunks.return_value = ([], 0)
            embeddings_adapter.embed_text.return_value = Vector(
                id="v1", embeddings=[0.1, 0.2], metadata={}
            )
            vector_store_adapter.add_vectors.return_value = ["v1"]
            with self.assertRaises(RuntimeError):
                task.run.__func__(
                    type("Bound", (), {"request": request})(),
                    args,
                )

        self.assertEqual(hooks, [])
        self.assertEqual(saved, ["new-chunk"])
        data_adapter.mark_text_chunks_superseded.assert_called()
        marked_ids = [
            call.args[0]
            for call in data_adapter.mark_text_chunks_superseded.call_args_list
        ]
        self.assertEqual(marked_ids, [["new-chunk"]])

    def test_failed_revision_leaves_previous_passage_current(self):
        from src.constants.data import TextChunk
        from src.constants.embeddings import Vector
        from src.workers.tasks import ingestion as ingestion_mod

        hooks = []
        previous = TextChunk(
            id="old-chunk",
            text="previous offer",
            metadata={
                "source_key": "offer-sheet",
                "source_content_hash": "old-hash",
            },
        )

        def fake_save_text_chunk(chunk, brain_id="default"):
            chunk.id = "new-chunk"
            return chunk

        task = ingestion_mod.ingest_data
        request = self._make_request()
        args = {
            "data": {"data_type": "text", "text_data": "revised offer"},
            "brain_id": "tenant-a",
            "meta_keys": {"source_id": "offer-sheet", "source_key": "offer-sheet"},
        }

        with (
            patch.object(ingestion_mod, "data_adapter") as data_adapter,
            patch.object(ingestion_mod, "embeddings_adapter") as embeddings_adapter,
            patch.object(ingestion_mod, "vector_store_adapter") as vector_store_adapter,
            patch.object(ingestion_mod, "cache_adapter"),
            patch.object(
                ingestion_mod,
                "enrich_kg_from_input",
                side_effect=RuntimeError("extract failed"),
            ),
            patch.object(
                ingestion_mod,
                "_run_ingest_commit_hooks",
                side_effect=lambda payload: hooks.append(payload),
            ),
            patch.object(ingestion_mod.config, "pipeline_mode", "accurate"),
            patch.object(ingestion_mod.config, "run_observations", False),
            patch.object(ingestion_mod, "set_ingestion_task_status"),
        ):
            data_adapter.save_text_chunk.side_effect = fake_save_text_chunk
            data_adapter.get_text_chunks.return_value = ([previous], 1)
            embeddings_adapter.embed_text.return_value = Vector(
                id="v1", embeddings=[0.1, 0.2], metadata={}
            )
            vector_store_adapter.add_vectors.return_value = ["v1"]
            with self.assertRaises(RuntimeError):
                task.run.__func__(
                    type("Bound", (), {"request": request})(),
                    args,
                )

        self.assertEqual(hooks, [])
        self.assertNotIn("superseded_at", previous.metadata)
        marked_ids = [
            call.args[0]
            for call in data_adapter.mark_text_chunks_superseded.call_args_list
        ]
        self.assertEqual(marked_ids, [["new-chunk"]])

    def test_ingest_commit_hooks_skip_when_chain_fails(self):
        from src.workers.tasks import ingestion as ingestion_mod

        hooks = []
        failed = ingestion_mod.mark_ingestion_chain_failed
        with (
            patch.object(ingestion_mod, "data_adapter") as data_adapter,
            patch.object(ingestion_mod, "set_ingestion_task_status") as set_status,
            patch.object(
                ingestion_mod,
                "_run_ingest_commit_hooks",
                side_effect=lambda payload: hooks.append(payload),
            ),
        ):
            failed(
                {"id": "rel-1"},
                RuntimeError("graph write failed"),
                None,
                "task-123",
                "tenant-a",
                "new-chunk",
            )

        self.assertEqual(hooks, [])
        data_adapter.mark_text_chunks_superseded.assert_called_once()
        self.assertEqual(
            data_adapter.mark_text_chunks_superseded.call_args.args[0], ["new-chunk"]
        )
        set_status.assert_called_once()
        self.assertEqual(set_status.call_args.args[2], "failed")

    def test_finalize_failed_status_does_not_commit(self):
        from src.workers.tasks import ingestion as ingestion_mod

        hooks = []
        finalize = ingestion_mod.finalize_ingestion_task
        finalize_request = MagicMock()
        finalize_request.id = "finalize-fail"
        with (
            patch.object(ingestion_mod, "graph_adapter"),
            patch.object(ingestion_mod, "set_ingestion_task_status"),
            patch.object(
                ingestion_mod,
                "_run_ingest_commit_hooks",
                side_effect=lambda payload: hooks.append(payload),
            ),
            patch.object(ingestion_mod, "data_adapter") as data_adapter,
        ):
            finalize.run.__func__(
                type("Bound", (), {"request": finalize_request})(),
                "task-123",
                "tenant-a",
                "failed",
                "graph write failed",
                None,
                None,
                None,
                {"brain_id": "tenant-a", "meta_keys": {"source_id": "note-1"}},
                ["old-chunk"],
                "new-chunk",
            )
        self.assertEqual(hooks, [])
        data_adapter.mark_text_chunks_superseded.assert_not_called()

    def test_invalid_pipeline_mode_fails_ingest(self):
        from src.workers.tasks import ingestion as ingestion_mod

        task = ingestion_mod.ingest_data
        request = self._make_request()
        args = {
            "data": {"data_type": "text", "text_data": "hello"},
            "brain_id": "tenant-a",
        }

        with (
            patch.object(ingestion_mod, "cache_adapter"),
            patch.object(ingestion_mod.config, "pipeline_mode", None),
        ):
            with self.assertRaises(ValueError):
                task.run.__func__(
                    type("Bound", (), {"request": request})(),
                    args,
                )


if __name__ == "__main__":
    unittest.main()

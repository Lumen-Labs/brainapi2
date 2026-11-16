from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from evals.adapters.brainapi import (
    BrainAPIAdapter,
    default_as_of_plugin,
    temporal_conflicts_plugin,
)
from evals.adapters.hybrid import HybridAdapter
from evals.config import Settings
from evals.datasets.loader import docs_for_run, load_corpus, load_questions


def _settings() -> Settings:
    return Settings.load()


class AdapterTests(unittest.TestCase):
    def test_hybrid_as_of_smoke_question(self) -> None:
        docs = docs_for_run(load_corpus("versioned_rules"), smoke=True)
        question = load_questions("versioned_rules", split="smoke")[0]
        adapter = HybridAdapter(_settings(), answerer="extractive", smoke=True)
        adapter.ingest(docs)
        result = adapter.query(question)
        self.assertTrue(result.answer)
        self.assertGreater(result.e2e_ms, 0)
        blob = result.answer.lower() + " " + " ".join(p.get("text", "") for p in result.passages)
        self.assertIn("30", blob)

    def test_as_of_plugin_unsupported_without_schema(self) -> None:
        body, mode = default_as_of_plugin(
            {},
            {"as_of": "2021-06-15"},
            field=None,
            schema_fields=set(),
        )
        self.assertEqual(mode, "unsupported")
        self.assertEqual(body, {})
        body, mode = default_as_of_plugin(
            {},
            {"as_of": "2021-06-15"},
            field="as_of",
            schema_fields={"as_of", "text"},
        )
        self.assertEqual(mode, "api_field")
        self.assertEqual(body["as_of"], "2021-06-15")

    def test_temporal_conflicts_plugin(self) -> None:
        payload = {
            "temporal_conflicts": [
                {"hubs": ["h1", "h2"], "entity_uuid": "e1"},
            ]
        }
        pairs = temporal_conflicts_plugin(payload, {})
        self.assertTrue(pairs)

    def test_brainapi_probe_marks_pending(self) -> None:
        settings = _settings()
        settings.brainpat_token = "test-token"
        adapter = BrainAPIAdapter(settings, brain_id="evalucversionedrules")
        client = MagicMock()
        client.openapi.return_value = MagicMock(
            data={
                "components": {
                    "schemas": {
                        "GetContextRequestBody": {"properties": {"text": {}, "brain_id": {}}},
                        "GetContextResponse": {
                            "properties": {
                                "text_context": {},
                                "temporal_conflicts": {},
                            }
                        },
                    }
                },
                "paths": {"/retrieve/context": {}},
            },
            status_code=200,
        )
        adapter.client = client
        caps = adapter.probe()
        self.assertFalse(caps["as_of"])
        self.assertEqual(caps["as_of_mode"], "unsupported_on_this_deployment")
        self.assertTrue(caps["pluggable"])
        self.assertTrue(caps["temporal_conflicts_sidecar"])
        self.assertFalse(caps["conflict_pairs"])

    def test_ingest_rejects_failed_task(self) -> None:
        settings = _settings()
        settings.brainpat_token = "test-token"
        adapter = BrainAPIAdapter(settings, brain_id="evalucversionedrules")
        client = MagicMock()
        client.ingest_text.return_value = MagicMock(
            data={"task_id": "task-1"},
            latency_ms=1.0,
        )
        client.wait_for_task.return_value = MagicMock(
            data={"status": "failed"},
            latency_ms=2.0,
        )
        adapter.client = client
        adapter._probed = True
        with self.assertRaises(RuntimeError) as ctx:
            adapter.ingest([{"id": "doc-1", "text": "hello"}])
        self.assertIn("failed", str(ctx.exception))
        self.assertEqual(client.wait_for_task.call_count, 1)

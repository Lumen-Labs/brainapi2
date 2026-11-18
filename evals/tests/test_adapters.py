from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from evals.adapters.brainapi import (
    BrainAPIAdapter,
    default_as_of_plugin,
    ingest_meta_keys,
    remap_conflict_pairs,
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

    def test_conflict_detection_is_cached_across_questions(self) -> None:
        docs = docs_for_run(load_corpus("policy_conflicts"), smoke=True)
        question = load_questions("policy_conflicts", split="smoke")[0]
        adapter = HybridAdapter(_settings(), answerer="extractive", smoke=True)
        adapter.ingest(docs)
        with patch("evals.adapters.hybrid.detect_conflicts", return_value=[]) as mocked:
            adapter.query(question)
            adapter.query(question)
        self.assertEqual(mocked.call_count, 1)

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

    def test_as_of_plugin_uses_include_history_when_no_as_of_field(self) -> None:
        body, mode = default_as_of_plugin(
            {},
            {"as_of": "2021-06-15", "task": "as_of"},
            field=None,
            schema_fields={"text", "brain_id", "include_history"},
        )
        self.assertEqual(mode, "include_history")
        self.assertTrue(body["include_history"])
        body, mode = default_as_of_plugin(
            {},
            {"as_of": "2021-06-15", "task": "as_of"},
            field="as_of",
            schema_fields={"as_of", "include_history", "text"},
        )
        self.assertEqual(mode, "api_field")
        self.assertEqual(body["as_of"], "2021-06-15")
        self.assertNotIn("include_history", body)

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
        self.assertFalse(caps.get("include_history"))

    def test_probe_include_history_is_interim_as_of(self) -> None:
        settings = _settings()
        settings.brainpat_token = "test-token"
        adapter = BrainAPIAdapter(settings, brain_id="evalucversionedrules")
        client = MagicMock()
        client.openapi.return_value = MagicMock(
            data={
                "components": {
                    "schemas": {
                        "GetContextRequestBody": {
                            "properties": {
                                "text": {},
                                "brain_id": {},
                                "include_history": {},
                            }
                        },
                        "GetContextResponse": {
                            "properties": {"text_context": {}, "source_passages": {}}
                        },
                    }
                },
                "paths": {"/retrieve/context": {}},
            },
            status_code=200,
        )
        adapter.client = client
        caps = adapter.probe()
        self.assertTrue(caps["as_of"])
        self.assertEqual(caps["as_of_mode"], "include_history")
        self.assertTrue(caps["include_history"])

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

    def test_ingest_stores_statement_metadata(self) -> None:
        settings = _settings()
        settings.brainpat_token = "test-token"
        adapter = BrainAPIAdapter(settings, brain_id="evalucpolicyconflicts")
        client = MagicMock()
        client.ingest_text.return_value = MagicMock(
            data={"task_id": "task-1"},
            latency_ms=1.0,
        )
        client.wait_for_task.return_value = MagicMock(
            data={"status": "completed", "chunk_id": "chunk-abc"},
            latency_ms=2.0,
        )
        adapter.client = client
        adapter._probed = True
        adapter.ingest(
            [
                {
                    "id": "hc-refunds-billing",
                    "statement_id": "stmt-refund-14",
                    "text": "refund in 14 days",
                }
            ]
        )
        kwargs = client.ingest_text.call_args.kwargs
        self.assertEqual(kwargs["meta_keys"]["statement_id"], "stmt-refund-14")
        self.assertEqual(kwargs["meta_keys"]["eval_doc_id"], "hc-refunds-billing")
        self.assertIn("Statement id: stmt-refund-14", client.ingest_text.call_args.args[0])
        self.assertEqual(adapter._canonical("chunk-abc"), "stmt-refund-14")
        self.assertEqual(ingest_meta_keys({"id": "d1", "clause_id": "FAR 13"})["source_key"], "FAR 13")

    def test_query_sends_include_history_for_as_of_questions(self) -> None:
        adapter = _adapter_with_client()
        adapter._schema_fields = {"text", "brain_id", "include_history"}
        adapter.client.retrieve_context.return_value = MagicMock(
            data={"source_passages": ["Net 30 days"], "text_context": "Net 30 days"},
            latency_ms=3.0,
            status_code=200,
        )
        result = adapter.query(
            {
                "id": "vr-smoke-asof-pay",
                "task": "as_of",
                "as_of": "2021-06-15",
                "question": "What did FAMR 12.4 say as of 2021-06-15?",
            }
        )
        extra = adapter.client.retrieve_context.call_args.kwargs["extra_body"]
        self.assertTrue(extra["include_history"])
        self.assertEqual(result.raw["as_of_mode"], "include_history")
        self.assertIn("as_of_mode=include_history", result.notes)

    def test_query_prefers_as_of_api_field_over_include_history(self) -> None:
        adapter = _adapter_with_client()
        adapter._schema_fields = {"text", "as_of", "include_history"}
        adapter.settings.as_of_field = "as_of"
        adapter.client.retrieve_context.return_value = MagicMock(
            data={"source_passages": ["ok"]},
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query(
            {"task": "as_of", "as_of": "2019-07-05", "question": "as of?"}
        )
        extra = adapter.client.retrieve_context.call_args.kwargs["extra_body"]
        self.assertEqual(extra["as_of"], "2019-07-05")
        self.assertNotIn("include_history", extra)
        self.assertEqual(result.raw["as_of_mode"], "api_field")

    def test_conflict_pairs_map_graph_ids_to_statement_ids(self) -> None:
        adapter = _adapter_with_client()
        adapter._remember("stmt-refund-14", "stmt-refund-14")
        adapter._remember("stmt-refund-30", "stmt-refund-30")
        adapter._remember("hc-refunds-billing", "stmt-refund-14")
        adapter.client.retrieve_context.return_value = MagicMock(
            data={
                "temporal_conflicts": [
                    {"hubs": ["hub-a", "hub-b"], "entity_uuid": "ent-1"},
                ],
                "triples": [
                    {
                        "identified_entity": "stmt-refund-14",
                        "source_chunk_ids": ["chunk-a"],
                        "triple": [
                            {"uuid": "ent-1", "name": "Refund"},
                            {"uuid": "rel-1", "name": "HAS"},
                            {"uuid": "hub-a", "name": "Event A"},
                            {"uuid": "rel-2", "name": "FOR"},
                            {"uuid": "obj-1", "name": "14 days"},
                        ],
                    },
                    {
                        "identified_entity": "stmt-refund-30",
                        "source_chunk_ids": ["chunk-b"],
                        "triple": [
                            {"uuid": "ent-1", "name": "Refund"},
                            {"uuid": "rel-3", "name": "HAS"},
                            {"uuid": "hub-b", "name": "Event B"},
                            {"uuid": "rel-4", "name": "FOR"},
                            {"uuid": "obj-2", "name": "30 days"},
                        ],
                    },
                ],
                "source_passages": [
                    "Document id: hc-refunds-billing\nStatement id: stmt-refund-14\n\n14 days",
                    "Document id: hc-refunds-legal\nStatement id: stmt-refund-30\n\n30 days",
                ],
            },
            latency_ms=4.0,
            status_code=200,
        )
        result = adapter.query(
            {"task": "conflict_audit", "question": "Which refund windows conflict?"}
        )
        self.assertIn(("stmt-refund-14", "stmt-refund-30"), result.conflict_pairs)
        self.assertTrue(adapter.capabilities["conflict_pairs"])
        self.assertEqual(adapter.capabilities["conflict_mode"], "temporal_conflicts_mapped")
        self.assertIn("conflict_ids_mapped", result.notes)
        self.assertFalse(result.raw["conflict_mapping"]["supported"] is False)

    def test_unmapped_conflict_uuids_are_unsupported(self) -> None:
        adapter = _adapter_with_client()
        adapter._remember("stmt-refund-14", "stmt-refund-14")
        adapter.client.retrieve_context.return_value = MagicMock(
            data={
                "temporal_conflicts": [
                    {"hubs": ["aaaaaaaa-1111", "bbbbbbbb-2222"], "entity_uuid": "cccccccc-3333"},
                ],
                "source_passages": ["no statement headers here"],
            },
            latency_ms=2.0,
            status_code=200,
        )
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertEqual(result.conflict_pairs, [])
        self.assertIn("conflict_mapping_unsupported", result.notes)
        self.assertFalse(adapter.capabilities["conflict_pairs"])
        self.assertEqual(adapter.capabilities["conflict_mode"], "id_mapping_unsupported")
        self.assertFalse(result.raw["conflict_mapping"]["supported"])

    def test_passages_use_real_source_doc_ids(self) -> None:
        adapter = _adapter_with_client()
        adapter._remember("stmt-pw-8", "stmt-pw-8")
        adapter._remember("hc-password-help", "stmt-pw-8")
        adapter._remember("chunk-meta", "stmt-pw-12")
        adapter.client.retrieve_context.return_value = MagicMock(
            data={
                "source_passages": [
                    "Document id: hc-password-help\nStatement id: stmt-pw-8\n\nPasswords must be 8 characters.",
                    {
                        "id": "chunk-meta",
                        "text": "Passwords must be 12 characters.",
                        "metadata": {"eval_doc_id": "stmt-pw-12", "statement_id": "stmt-pw-12"},
                    },
                ],
                "historical_context": [
                    "Document id: hc-password-help\nStatement id: stmt-pw-8\n\n[superseded] older copy",
                ],
            },
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query({"question": "password length?"})
        doc_ids = [p["doc_id"] for p in result.passages]
        self.assertIn("stmt-pw-8", doc_ids)
        self.assertIn("stmt-pw-12", doc_ids)
        self.assertFalse(any(doc_id.startswith("passage-") for doc_id in doc_ids))
        hist = [p for p in result.passages if p["title"] == "historical_context"]
        self.assertTrue(hist)
        self.assertEqual(hist[0]["doc_id"], "stmt-pw-8")

    def test_remap_conflict_pairs_helper(self) -> None:
        mapped = remap_conflict_pairs(
            [("hub-a", "hub-b"), ("hub-a", "unknown")],
            {"hub-a": "stmt-1", "hub-b": "stmt-2"},
        )
        self.assertEqual(mapped["pairs"], [("stmt-1", "stmt-2")])
        self.assertFalse(mapped["supported"] is False)
        empty = remap_conflict_pairs([("u1", "u2")], {})
        self.assertEqual(empty["pairs"], [])
        self.assertFalse(empty["supported"])


def _adapter_with_client() -> BrainAPIAdapter:
    settings = _settings()
    settings.brainpat_token = "test-token"
    adapter = BrainAPIAdapter(settings, brain_id="evaluctest")
    adapter.client = MagicMock()
    adapter._probed = True
    adapter._schema_fields = {"text", "brain_id"}
    return adapter

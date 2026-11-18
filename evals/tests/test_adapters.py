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
    settings = Settings.load()
    settings.conflict_path = None
    return settings


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
            [("hub-a", "hub-b")],
            {"hub-a": "stmt-1", "hub-b": "stmt-2"},
        )
        self.assertEqual(mapped["pairs"], [("stmt-1", "stmt-2")])
        self.assertTrue(mapped["supported"])
        mixed = remap_conflict_pairs(
            [("hub-a", "hub-b"), ("hub-a", "unknown")],
            {"hub-a": "stmt-1", "hub-b": "stmt-2"},
        )
        self.assertEqual(mixed["pairs"], [("stmt-1", "stmt-2")])
        self.assertFalse(mixed["supported"])
        self.assertEqual(mixed["n_unmapped"], 1)
        empty = remap_conflict_pairs([("u1", "u2")], {})
        self.assertEqual(empty["pairs"], [])
        self.assertFalse(empty["supported"])

    def test_clause_source_key_is_not_a_gold_alias(self) -> None:
        adapter = _adapter_with_client()
        adapter._remember("famr-12-4-2019", "famr-12-4-2019", lock=True)
        adapter._remember("famr-12-4-2024", "famr-12-4-2024", lock=True)
        adapter._remember("FAR 12.4", "famr-12-4-2019")
        adapter._remember("FAR 12.4", "famr-12-4-2024")
        self.assertIsNone(adapter._canonical("FAR 12.4"))
        self.assertEqual(adapter._canonical("famr-12-4-2019"), "famr-12-4-2019")
        self.assertEqual(adapter._canonical("famr-12-4-2024"), "famr-12-4-2024")

    def test_co_retrieved_passages_are_not_conflict_pairs(self) -> None:
        adapter = _adapter_with_client()
        adapter._remember("stmt-a", "stmt-a")
        adapter._remember("stmt-b", "stmt-b")
        adapter.client.retrieve_context.return_value = MagicMock(
            data={
                "temporal_conflicts": [
                    {"hubs": ["hub-unknown-a", "hub-unknown-b"]},
                ],
                "source_passages": [
                    "Document id: doc-a\nStatement id: stmt-a\n\n14 days",
                    "Document id: doc-b\nStatement id: stmt-b\n\n30 days",
                ],
            },
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertEqual(result.conflict_pairs, [])
        self.assertIn("conflict_mapping_unsupported", result.notes)
        self.assertFalse(adapter.capabilities["conflict_pairs"])

    def test_conflict_capability_is_order_independent(self) -> None:
        adapter = _adapter_with_client()
        adapter._remember("stmt-a", "stmt-a")
        adapter._remember("stmt-b", "stmt-b")
        adapter._remember("hub-a", "stmt-a")
        adapter._remember("hub-b", "stmt-b")
        mapped_payload = MagicMock(
            data={"temporal_conflicts": [{"hubs": ["hub-a", "hub-b"]}]},
            latency_ms=1.0,
            status_code=200,
        )
        unmapped_payload = MagicMock(
            data={"temporal_conflicts": [{"hubs": ["u1", "u2"]}]},
            latency_ms=1.0,
            status_code=200,
        )
        adapter.client.retrieve_context.side_effect = [unmapped_payload, mapped_payload]
        adapter.query({"task": "conflict_audit", "question": "one"})
        adapter.query({"task": "conflict_audit", "question": "two"})
        self.assertFalse(adapter.capabilities["conflict_pairs"])
        self.assertEqual(adapter.capabilities["conflict_mode"], "id_mapping_unsupported")

        other = _adapter_with_client()
        other._remember("stmt-a", "stmt-a")
        other._remember("stmt-b", "stmt-b")
        other._remember("hub-a", "stmt-a")
        other._remember("hub-b", "stmt-b")
        other.client.retrieve_context.side_effect = [mapped_payload, unmapped_payload]
        other.query({"task": "conflict_audit", "question": "one"})
        other.query({"task": "conflict_audit", "question": "two"})
        self.assertFalse(other.capabilities["conflict_pairs"])
        self.assertEqual(other.capabilities["conflict_mode"], "id_mapping_unsupported")
        other.finalize_conflict_mappings()
        self.assertFalse(other.capabilities["conflict_pairs"])
        first = other._pending_conflict_jobs[0]["result"]
        self.assertIn(("stmt-a", "stmt-b"), first.conflict_pairs)
        second = other._pending_conflict_jobs[1]["result"]
        self.assertIn("conflict_mapping_unsupported", second.notes)

    def test_shared_chunk_mapping_is_order_independent(self) -> None:
        def run(order: list[int]) -> tuple[list[list[tuple[str, str]]], bool]:
            adapter = _adapter_with_client()
            adapter._remember("stmt-a", "stmt-a")
            adapter._remember("stmt-b", "stmt-b")
            payloads = [
                MagicMock(
                    data={
                        "temporal_conflicts": [{"hubs": ["shared-chunk", "hub-a"]}],
                        "triples": [
                            {
                                "identified_entity": "stmt-a",
                                "source_chunk_ids": ["shared-chunk"],
                                "triple": [{"uuid": "hub-a", "name": "A"}],
                            }
                        ],
                    },
                    latency_ms=1.0,
                    status_code=200,
                ),
                MagicMock(
                    data={
                        "temporal_conflicts": [{"hubs": ["shared-chunk", "hub-b"]}],
                        "triples": [
                            {
                                "identified_entity": "stmt-b",
                                "source_chunk_ids": ["shared-chunk"],
                                "triple": [{"uuid": "hub-b", "name": "B"}],
                            }
                        ],
                    },
                    latency_ms=1.0,
                    status_code=200,
                ),
            ]
            adapter.client.retrieve_context.side_effect = [payloads[i] for i in order]
            adapter.query({"task": "conflict_audit", "question": "one"})
            adapter.query({"task": "conflict_audit", "question": "two"})
            adapter.finalize_conflict_mappings()
            pairs = [list(job["result"].conflict_pairs) for job in adapter._pending_conflict_jobs]
            return pairs, bool(adapter.capabilities["conflict_pairs"])

        pairs_a, cap_a = run([0, 1])
        pairs_b, cap_b = run([1, 0])
        self.assertEqual(cap_a, cap_b)
        self.assertFalse(cap_a)
        self.assertEqual(pairs_a, pairs_b)
        self.assertEqual(pairs_a, [[], []])

    def test_retrieval_text_does_not_register_new_gold_ids(self) -> None:
        adapter = _adapter_with_client()
        adapter._remember("stmt-known", "stmt-known")
        adapter.client.retrieve_context.return_value = MagicMock(
            data={
                "source_passages": [
                    "Document id: invented-doc\nStatement id: invented-stmt\n\nhello",
                ]
            },
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query({"question": "hello?"})
        self.assertNotIn("invented-stmt", adapter._gold_ids)
        self.assertNotIn("invented-doc", adapter._gold_ids)
        self.assertIsNone(adapter._canonical("invented-stmt"))
        self.assertTrue(result.passages[0]["doc_id"].startswith("passage-"))

    def test_chunk_id_resolved_independently_of_passage_id(self) -> None:
        adapter = _adapter_with_client()
        adapter._remember("stmt-pw-12", "stmt-pw-12")
        adapter._remember("chunk-meta", "stmt-pw-12")
        adapter.client.retrieve_context.return_value = MagicMock(
            data={
                "source_passages": [
                    {
                        "id": "passage-uuid-1",
                        "chunk_id": "chunk-meta",
                        "text": "Passwords must be 12 characters.",
                    }
                ]
            },
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query({"question": "password length?"})
        self.assertEqual(result.passages[0]["id"], "passage-uuid-1")
        self.assertEqual(result.passages[0]["doc_id"], "stmt-pw-12")

    def test_as_of_questions_reserve_historical_context(self) -> None:
        adapter = _adapter_with_client()
        adapter._schema_fields = {"text", "brain_id", "include_history"}
        current = [f"current {i}" for i in range(8)]
        history = [
            f"Document id: old-{i}\nStatement id: stmt-old-{i}\n\nhist {i}" for i in range(8)
        ]
        for i in range(8):
            adapter._remember(f"stmt-old-{i}", f"stmt-old-{i}")
        adapter.client.retrieve_context.return_value = MagicMock(
            data={"source_passages": current, "historical_context": history},
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query(
            {
                "task": "as_of",
                "as_of": "2021-06-15",
                "question": "What did it say as of 2021-06-15?",
            }
        )
        self.assertEqual(len(result.passages), 8)
        self.assertTrue(any(p.get("title") == "historical_context" for p in result.passages))
        extra = adapter.client.retrieve_context.call_args.kwargs["extra_body"]
        self.assertTrue(extra["include_history"])
        self.assertNotIn("as_of", extra)
        self.assertEqual(result.raw["as_of_mode"], "include_history")

    def test_as_of_plugin_keeps_history_for_date_range(self) -> None:
        body, mode = default_as_of_plugin(
            {},
            {"task": "changelog", "date_from": "2019-01-01", "date_to": "2024-01-01"},
            field="as_of",
            schema_fields={"as_of", "include_history", "text"},
        )
        self.assertEqual(mode, "include_history")
        self.assertNotIn("as_of", body)
        self.assertTrue(body["include_history"])
        body, mode = default_as_of_plugin(
            {},
            {"task": "changelog", "date_from": "2019-01-01", "date_to": "2024-01-01"},
            field="as_of",
            schema_fields={"as_of", "date_to", "text"},
        )
        self.assertEqual(mode, "api_schema")
        self.assertEqual(body["date_to"], "2024-01-01")
        self.assertNotIn("as_of", body)
        self.assertNotIn("include_history", body)

    def test_empty_conflict_producer_is_scoreable(self) -> None:
        adapter = _adapter_with_client()
        adapter.capabilities["temporal_conflicts_sidecar"] = True
        adapter.settings.conflict_path = "/retrieve/conflicts"
        adapter.client.request_path.return_value = MagicMock(
            data={"pairs": []},
            status_code=200,
        )
        adapter.client.retrieve_context.return_value = MagicMock(
            data={"temporal_conflicts": []},
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertEqual(result.conflict_pairs, [])
        self.assertIn("conflict_producer_empty", result.notes)
        self.assertNotIn("conflict_mapping_unsupported", result.notes)
        self.assertNotIn("conflict_capability_pending", result.notes)
        self.assertTrue(adapter.capabilities["conflict_pairs"])

    def test_sidecar_empty_response_is_scoreable(self) -> None:
        adapter = _adapter_with_client()
        adapter.client.retrieve_context.return_value = MagicMock(
            data={"temporal_conflicts": []},
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertEqual(result.conflict_pairs, [])
        self.assertIn("conflict_producer_empty", result.notes)
        self.assertNotIn("conflict_capability_pending", result.notes)
        self.assertTrue(adapter.capabilities["conflict_pairs"])

    def test_failed_conflict_endpoint_is_unavailable(self) -> None:
        adapter = _adapter_with_client()
        adapter.settings.conflict_path = "/retrieve/conflicts"
        adapter.client.request_path.return_value = MagicMock(
            data={"detail": "missing"},
            status_code=404,
        )
        adapter.client.retrieve_context.return_value = MagicMock(
            data={"source_passages": ["nothing here"]},
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertEqual(result.conflict_pairs, [])
        self.assertIn("conflict_capability_pending", result.notes)
        self.assertNotIn("conflict_producer_empty", result.notes)
        self.assertFalse(adapter.capabilities["conflict_pairs"])

    def test_partial_conflict_mapping_is_unsupported(self) -> None:
        adapter = _adapter_with_client()
        adapter._remember("stmt-a", "stmt-a")
        adapter._remember("hub-a", "stmt-a")
        adapter.client.retrieve_context.return_value = MagicMock(
            data={"temporal_conflicts": [{"hubs": ["hub-a", "hub-unknown"]}]},
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertEqual(result.conflict_pairs, [])
        self.assertIn("conflict_mapping_unsupported", result.notes)
        self.assertFalse(result.raw["conflict_mapping"]["supported"])
        self.assertEqual(result.raw["conflict_mapping"]["n_unmapped"], 1)

    def test_metadata_doc_id_indexes_passage_alias(self) -> None:
        adapter = _adapter_with_client()
        adapter._remember("stmt-a", "stmt-a")
        adapter._remember("stmt-b", "stmt-b")
        adapter._remember("doc-a", "stmt-a")
        adapter._remember("doc-b", "stmt-b")
        adapter.client.retrieve_context.return_value = MagicMock(
            data={
                "temporal_conflicts": [{"hubs": ["hub-a", "hub-b"]}],
                "source_passages": [
                    {"id": "hub-a", "text": "14 days", "metadata": {"doc_id": "doc-a"}},
                    {"id": "hub-b", "text": "30 days", "metadata": {"doc_id": "doc-b"}},
                ],
            },
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertIn(("stmt-a", "stmt-b"), result.conflict_pairs)
        self.assertEqual(result.passages[0]["doc_id"], "stmt-a")

    def test_passage_properties_statement_id_is_used(self) -> None:
        adapter = _adapter_with_client()
        adapter._remember("stmt-a", "stmt-a")
        adapter.client.retrieve_context.return_value = MagicMock(
            data={
                "source_passages": [
                    {
                        "text": "Passwords must be 8 characters.",
                        "properties": {"statement_id": "stmt-a"},
                    }
                ]
            },
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query({"question": "password length?"})
        self.assertEqual(result.passages[0]["doc_id"], "stmt-a")

    def test_shared_clause_key_does_not_label_other_passages(self) -> None:
        adapter = _adapter_with_client()
        adapter._remember("stmt-a", "stmt-a", lock=True)
        adapter._remember("stmt-b", "stmt-b", lock=True)
        adapter.client.retrieve_context.return_value = MagicMock(
            data={
                "source_passages": [
                    {
                        "id": "chunk-a",
                        "text": "Document id: famr-2019\nStatement id: stmt-a\n\nold",
                        "metadata": {"source_key": "FAR 12.4", "statement_id": "stmt-a"},
                    },
                    {
                        "id": "FAR 12.4",
                        "text": "unrelated later passage",
                    },
                ]
            },
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query({"question": "what did it say?"})
        self.assertEqual(result.passages[0]["doc_id"], "stmt-a")
        self.assertNotEqual(result.passages[1]["doc_id"], "stmt-a")

    def test_unavailable_conflict_producer_is_unsupported(self) -> None:
        adapter = _adapter_with_client()
        adapter.client.retrieve_context.return_value = MagicMock(
            data={"source_passages": ["nothing here"]},
            latency_ms=1.0,
            status_code=200,
        )
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertEqual(result.conflict_pairs, [])
        self.assertIn("conflict_capability_pending", result.notes)
        self.assertFalse(adapter.capabilities["conflict_pairs"])
        self.assertFalse(result.raw["conflict_mapping"]["supported"])

    def test_probe_schema_as_of_field_without_env(self) -> None:
        settings = _settings()
        settings.brainpat_token = "test-token"
        settings.as_of_field = ""
        adapter = BrainAPIAdapter(settings, brain_id="evalucversionedrules")
        client = MagicMock()
        client.openapi.return_value = MagicMock(
            data={
                "components": {
                    "schemas": {
                        "GetContextRequestBody": {
                            "properties": {"text": {}, "brain_id": {}, "valid_at": {}}
                        },
                        "GetContextResponse": {"properties": {"text_context": {}}},
                    }
                },
                "paths": {"/retrieve/context": {}},
            },
            status_code=200,
        )
        adapter.client = client
        caps = adapter.probe()
        self.assertTrue(caps["as_of"])
        self.assertEqual(caps["as_of_mode"], "api_schema")


def _adapter_with_client() -> BrainAPIAdapter:
    settings = _settings()
    settings.brainpat_token = "test-token"
    adapter = BrainAPIAdapter(settings, brain_id="evaluctest")
    adapter.client = MagicMock()
    adapter._probed = True
    adapter._schema_fields = {"text", "brain_id"}
    return adapter

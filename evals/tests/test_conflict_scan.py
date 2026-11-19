from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from evals.evaluate import _conflict_headline, _score_one, compare_systems
from evals.tests.test_adapters import _adapter_with_client, _settings
from evals.adapters.brainapi import BrainAPIAdapter


def _conflict(left: str, right: str, *, left_key: str | None = None, right_key: str | None = None) -> dict:
    return {
        "id": f"{left}::{right}",
        "sides": [
            {
                "fact_id": "fact-" + left,
                "source": {
                    "source_key": left_key or left,
                    "chunk_ids": [left + "-chunk"],
                    "text": f"Statement id: {left_key or left}",
                },
            },
            {
                "fact_id": "fact-" + right,
                "source": {
                    "source_key": right_key or right,
                    "chunk_ids": [right + "-chunk"],
                    "text": f"Statement id: {right_key or right}",
                },
            },
        ],
    }


def _scan_adapter(conflicts: list[dict] | None = None) -> BrainAPIAdapter:
    adapter = _adapter_with_client()
    adapter.settings.conflict_path = "/retrieve/conflicts"
    adapter.settings.conflict_scan_path = "/retrieve/conflicts/scan"
    rows = conflicts or []
    adapter.client.start_conflict_scan.return_value = MagicMock(
        data={"task_id": "scan-1", "message": "Conflict scan accepted"},
        status_code=202,
        latency_ms=1.0,
    )
    adapter.client.wait_for_task.return_value = MagicMock(
        data={"status": "completed", "stage": "conflict_scan"},
        status_code=200,
        latency_ms=4.0,
    )
    adapter.client.list_conflicts.return_value = MagicMock(
        data={"conflicts": rows, "count": len(rows), "total": len(rows)},
        status_code=200,
        latency_ms=2.0,
    )
    adapter.client.retrieve_context.return_value = MagicMock(
        data={"source_passages": ["refund windows"]},
        latency_ms=1.0,
        status_code=200,
    )
    return adapter


class ConflictScanTests(unittest.TestCase):
    def test_probe_marks_scan_then_list(self) -> None:
        settings = _settings()
        settings.brainpat_token = "test-token"
        settings.conflict_path = "/retrieve/conflicts"
        settings.conflict_scan_path = "/retrieve/conflicts/scan"
        adapter = BrainAPIAdapter(settings, brain_id="evalucpolicyconflicts")
        client = MagicMock()
        client.openapi.return_value = MagicMock(
            data={
                "components": {
                    "schemas": {
                        "GetContextRequestBody": {"properties": {"text": {}, "brain_id": {}}},
                        "GetContextResponse": {"properties": {"text_context": {}}},
                    }
                },
                "paths": {
                    "/retrieve/context": {},
                    "/retrieve/conflicts": {"get": {}},
                    "/retrieve/conflicts/scan": {"post": {}},
                },
            },
            status_code=200,
        )
        adapter.client = client
        caps = adapter.probe()
        self.assertTrue(caps["conflict_pairs"])
        self.assertEqual(caps["conflict_mode"], "scan_then_list")

    def test_scan_then_list_maps_source_keys(self) -> None:
        adapter = _scan_adapter(
            [_conflict("stmt-refund-14", "stmt-refund-30")]
        )
        adapter._remember("stmt-refund-14", "stmt-refund-14")
        adapter._remember("stmt-refund-30", "stmt-refund-30")
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertIn(("stmt-refund-14", "stmt-refund-30"), result.conflict_pairs)
        self.assertIn("conflict_scan_then_list", result.notes)
        self.assertIn("conflict_ids_mapped", result.notes)
        self.assertEqual(adapter.capabilities["conflict_mode"], "scan_then_list_mapped")
        adapter.client.start_conflict_scan.assert_called_once()
        kwargs = adapter.client.start_conflict_scan.call_args
        self.assertEqual(kwargs.kwargs.get("use_llm"), False)
        self.assertEqual(kwargs.kwargs.get("path"), "/retrieve/conflicts/scan")
        adapter.client.list_conflicts.assert_called_once()
        list_kwargs = adapter.client.list_conflicts.call_args.kwargs
        self.assertEqual(list_kwargs.get("path"), "/retrieve/conflicts")
        self.assertEqual(list_kwargs.get("status"), "open")
        adapter.client.start_conflict_scan.reset_mock()
        adapter.query({"task": "conflict_audit", "question": "again?"})
        adapter.client.start_conflict_scan.assert_not_called()

    def test_scan_then_list_maps_chunk_ids(self) -> None:
        adapter = _scan_adapter(
            [
                _conflict(
                    "hub-a",
                    "hub-b",
                    left_key="unknown-left",
                    right_key="unknown-right",
                )
            ]
        )
        adapter._remember("stmt-a", "stmt-a")
        adapter._remember("stmt-b", "stmt-b")
        adapter._remember("hub-a-chunk", "stmt-a")
        adapter._remember("hub-b-chunk", "stmt-b")
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertIn(("stmt-a", "stmt-b"), result.conflict_pairs)

    def test_scan_does_not_post_to_list_path(self) -> None:
        adapter = _scan_adapter([])
        adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        adapter.client.request_path.assert_not_called()
        scan_call = adapter.client.start_conflict_scan.call_args
        self.assertNotEqual(scan_call.kwargs.get("path"), "/retrieve/conflicts")

    def test_empty_scan_is_scoreable(self) -> None:
        adapter = _scan_adapter([])
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertEqual(result.conflict_pairs, [])
        self.assertIn("conflict_producer_empty", result.notes)
        self.assertIn("conflict_scan_then_list", result.notes)
        self.assertTrue(adapter.capabilities["conflict_pairs"])
        scored = _score_one(
            {"task": "conflict_audit"},
            result,
            gold_pairs=[("stmt-a", "stmt-b")],
        )
        self.assertFalse(scored.get("unsupported"))
        self.assertEqual(scored["scoring"], "conflict_pairs")
        self.assertEqual(scored["f1"], 0.0)

    def test_scan_404_is_skipped(self) -> None:
        adapter = _scan_adapter()
        adapter.client.start_conflict_scan.return_value = MagicMock(
            data={"detail": "not found"},
            status_code=404,
        )
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertIn("conflict_scan_unavailable", result.notes)
        self.assertEqual(result.conflict_pairs, [])
        scored = _score_one(
            {"task": "conflict_audit"},
            result,
            gold_pairs=[("stmt-a", "stmt-b")],
        )
        self.assertTrue(scored["unsupported"])
        self.assertIn("not available", scored["reason"])

    def test_failed_scan_task_is_error(self) -> None:
        adapter = _scan_adapter()
        adapter.client.wait_for_task.return_value = MagicMock(
            data={"status": "failed"},
            status_code=200,
            latency_ms=3.0,
        )
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertIn("conflict_scan_task_failed", result.notes)
        scored = _score_one(
            {"task": "conflict_audit"},
            result,
            gold_pairs=[("stmt-a", "stmt-b")],
        )
        self.assertFalse(scored.get("unsupported"))
        self.assertEqual(scored["scoring"], "conflict_scan_error")
        self.assertEqual(scored["correct"], False)
        self.assertTrue(scored.get("error"))
        headline = _conflict_headline(
            [{"notes": result.notes, "error": scored["error"], "scoring": scored["scoring"]}],
            [("stmt-a", "stmt-b")],
        )
        self.assertEqual(headline["scoring"], "conflict_scan_error")
        self.assertIsNone(headline["f1"])

    def test_paginated_list_is_consumed(self) -> None:
        adapter = _scan_adapter()
        first = MagicMock(
            data={
                "conflicts": [_conflict("stmt-a", "stmt-b")],
                "count": 1,
                "total": 2,
            },
            status_code=200,
        )
        second = MagicMock(
            data={
                "conflicts": [_conflict("stmt-c", "stmt-d")],
                "count": 1,
                "total": 2,
            },
            status_code=200,
        )
        adapter._remember("stmt-a", "stmt-a")
        adapter._remember("stmt-b", "stmt-b")
        adapter._remember("stmt-c", "stmt-c")
        adapter._remember("stmt-d", "stmt-d")

        def _list(brain_id, **kwargs):
            skip = int(kwargs.get("skip") or 0)
            return first if skip == 0 else second

        adapter.client.list_conflicts.side_effect = _list
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertIn(("stmt-a", "stmt-b"), result.conflict_pairs)
        self.assertIn(("stmt-c", "stmt-d"), result.conflict_pairs)
        self.assertGreaterEqual(adapter.client.list_conflicts.call_count, 2)

    def test_truncated_pagination_is_an_error(self) -> None:
        adapter = _scan_adapter()

        def _list(brain_id, **kwargs):
            skip = int(kwargs.get("skip") or 0)
            return MagicMock(
                data={
                    "conflicts": [_conflict(f"stmt-a{skip}", f"stmt-b{skip}")],
                    "count": 1,
                    "total": 10101,
                },
                status_code=200,
            )

        adapter.client.list_conflicts.side_effect = _list
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertIn("conflict_list_error", result.notes)
        scored = _score_one(
            {"task": "conflict_audit"},
            result,
            gold_pairs=[("stmt-a", "stmt-b")],
        )
        self.assertEqual(scored["scoring"], "conflict_scan_error")
        self.assertTrue(scored.get("error"))

    def test_ambiguous_alias_is_unmapped_on_finalize(self) -> None:
        adapter = _scan_adapter(
            [
                _conflict(
                    "hub-a",
                    "hub-b",
                    left_key="unknown-left",
                    right_key="unknown-right",
                )
            ]
        )
        adapter._remember("stmt-a", "stmt-a")
        adapter._remember("stmt-b", "stmt-b")
        adapter._remember("hub-a-chunk", "stmt-a")
        adapter._remember("hub-b-chunk", "stmt-b")
        result = adapter.query({"task": "conflict_audit", "question": "conflicts?"})
        self.assertIn(("stmt-a", "stmt-b"), result.conflict_pairs)
        adapter._remember("hub-a-chunk", "stmt-other")
        adapter.finalize_conflict_mappings()
        self.assertEqual(result.conflict_pairs, [])
        self.assertIn("conflict_mapping_unsupported", result.notes)

    def test_compare_systems_does_not_hide_scan_error(self) -> None:
        result = compare_systems(
            {
                "hybrid_baseline": {
                    "metrics": {
                        "n": 2,
                        "n_unsupported": 0,
                        "accuracy": 1.0,
                        "wilson_lo": 0.3,
                        "wilson_hi": 1.0,
                    }
                },
                "brainapi": {
                    "capabilities": {
                        "conflict_pairs": False,
                        "conflict_error": True,
                        "conflict_mode": "scan_then_list_error",
                    },
                    "metrics": {
                        "n": 2,
                        "n_unsupported": 0,
                        "n_errors": 2,
                        "accuracy": 0.0,
                        "wilson_lo": 0.0,
                        "wilson_hi": 0.6,
                    },
                },
            },
            use_case="policy_conflicts",
        )
        self.assertNotEqual(result["verdict"], "capability_pending")

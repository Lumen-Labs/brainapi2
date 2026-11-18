import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.saving.ingest_effects import (
    ChunkSnapshot,
    DigestMismatch,
    EffectTouch,
    GraphSnapshot,
    IngestEffect,
    PurgePartial,
    PurgeUnsafe,
    apply_actions,
    chunks_linked_to_task,
    confirm_digest,
    handle_purge,
    plan_purge,
    reconstruct_effect,
    reconstruct_live,
    remember,
)


def _screenshot_ops():
    path = Path(__file__).resolve().parents[1] / "scripts" / "select_screenshot_imports.py"
    spec = importlib.util.spec_from_file_location("select_screenshot_imports", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Sink:
    def __init__(self):
        self.deleted_chunks: list[str] = []
        self.hubs: list[str] = []
        self.nodes: list[str] = []
        self.edges: list[str] = []
        self.vectors: list[tuple[str, list[str]]] = []
        self.unlinked: list[tuple[str, str]] = []
        self.structured: list[str] = []
        self.observations: list[str] = []

    def delete_chunks(self, ids):
        self.deleted_chunks.extend(ids)

    def delete_passage_vectors(self, chunk_ids, vector_ids):
        self.vectors.append(("data", list(chunk_ids)))

    def unlink_chunk(self, chunk_id, task_ids):
        self.unlinked.append(("chunk", chunk_id))

    def delete_observations(self, ids):
        self.observations.extend(ids)

    def delete_structured(self, structured_id):
        self.structured.append(structured_id)

    def unlink_graph(self, kind, uuid, detail):
        self.unlinked.append((kind, uuid))

    def delete_nodes(self, ids):
        self.nodes.extend(ids)

    def delete_edges(self, ids):
        self.edges.extend(ids)

    def delete_graph_vectors(self, store, ids):
        self.vectors.append((store, list(ids)))

    def delete_hubs(self, ids):
        self.hubs.extend(ids)


def _effect() -> IngestEffect:
    return IngestEffect(
        task_id="task-1",
        created_chunk_ids=["c1"],
        reused_chunk_ids=["c-reused"],
        chunk_vector_ids={"c1": "v-c1"},
        observations=[{"id": "o1", "resource_id": "c1"}],
        nodes=[
            EffectTouch(uuid="exclusive", created=True, v_id="v-node", name="Harbor"),
            EffectTouch(uuid="shared", created=False, name="Slack"),
        ],
        edges=[
            EffectTouch(uuid="structured-edge", created=True, v_id="v-edge", name="HAS"),
        ],
        structured_data_id="structured-1",
    )


def _snapshots():
    chunks = {
        "c1": ChunkSnapshot(ingest_task_ids=["task-1"]),
        "c-reused": ChunkSnapshot(ingest_task_ids=["task-1", "task-older"]),
    }
    nodes = {
        "exclusive": GraphSnapshot(source_chunk_ids=["c1"], ingest_task_ids=["task-1"], v_id="v-node", name="Harbor"),
        "shared": GraphSnapshot(source_chunk_ids=["c1", "c-other"], ingest_task_ids=["task-1", "task-older"], name="Slack"),
    }
    edges = {
        "structured-edge": GraphSnapshot(source_chunk_ids=[], ingest_task_ids=["task-1"], v_id="v-edge", name="HAS"),
    }
    return chunks, nodes, edges


class PurgePlanTests(unittest.TestCase):
    def test_exclusive_node_is_deleted_and_shared_node_is_unlinked(self):
        chunks, nodes, edges = _snapshots()
        plan = plan_purge(_effect(), chunks=chunks, nodes=nodes, edges=edges, claimed_chunks=set())
        kinds = {(action.kind, action.id) for action in plan.actions}
        self.assertIn(("delete_node", "exclusive"), kinds)
        self.assertIn(("delete_hub", "exclusive"), kinds)
        self.assertIn(("unlink_node", "shared"), kinds)
        self.assertNotIn(("delete_node", "shared"), kinds)
        shared = next(action for action in plan.actions if action.id == "shared")
        self.assertEqual(shared.detail["source_chunk_ids"], ["c-other"])
        self.assertEqual(shared.detail["ingest_task_ids"], ["task-older"])

    def test_reused_passage_is_not_deleted(self):
        chunks, nodes, edges = _snapshots()
        plan = plan_purge(_effect(), chunks=chunks, nodes=nodes, edges=edges, claimed_chunks={"c1"})
        kinds = {(action.kind, action.id) for action in plan.actions}
        self.assertNotIn(("delete_chunk", "c-reused"), kinds)
        self.assertNotIn(("delete_chunk", "c1"), kinds)
        self.assertIn(("unlink_chunk", "c1"), kinds)
        self.assertIn(("unlink_chunk", "c-reused"), kinds)
        self.assertEqual(plan.reused, 1)

    def test_structured_edge_without_chunk_ids_is_deleted(self):
        chunks, nodes, edges = _snapshots()
        plan = plan_purge(_effect(), chunks=chunks, nodes=nodes, edges=edges, claimed_chunks=set())
        kinds = {(action.kind, action.id) for action in plan.actions}
        self.assertIn(("delete_edge", "structured-edge"), kinds)
        self.assertIn(("delete_structured", "structured-1"), kinds)
        self.assertIn(("delete_chunk", "c1"), kinds)
        self.assertIn(("delete_observation", "o1"), kinds)

    def test_apply_deletes_exclusive_artifacts_and_hubs(self):
        chunks, nodes, edges = _snapshots()
        plan = plan_purge(_effect(), chunks=chunks, nodes=nodes, edges=edges, claimed_chunks=set())
        sink = _Sink()
        apply_actions(plan, sink)
        self.assertEqual(sink.deleted_chunks, ["c1"])
        self.assertEqual(sink.nodes, ["exclusive"])
        self.assertEqual(sink.edges, ["structured-edge"])
        self.assertEqual(sink.hubs, ["exclusive"])
        self.assertIn(("node", "shared"), sink.unlinked)
        self.assertEqual(sink.structured, ["structured-1"])
        self.assertEqual(sink.observations, ["o1"])

    def test_confirm_refuses_a_changed_digest(self):
        chunks, nodes, edges = _snapshots()
        effect = _effect()
        first = handle_purge(effect, chunks=chunks, nodes=nodes, edges=edges)
        self.assertFalse(first["executed"])
        sink = _Sink()
        with self.assertRaises(DigestMismatch):
            handle_purge(
                effect,
                chunks=chunks,
                nodes=nodes,
                edges=edges,
                digest="not-the-plan",
                sink=sink,
            )
        self.assertEqual(sink.deleted_chunks, [])
        second = handle_purge(
            effect,
            chunks=chunks,
            nodes=nodes,
            edges=edges,
            digest=first["digest"],
            sink=sink,
        )
        self.assertTrue(second["executed"])
        self.assertEqual(sink.deleted_chunks, ["c1"])


class ReconstructTests(unittest.TestCase):
    def test_dry_run_marks_shared_and_exclusive_without_deleting(self):
        effect = reconstruct_effect(
            "old-task",
            chunks=[{"id": "c1"}],
            nodes=[
                {"uuid": "exclusive", "name": "Probe", "source_chunk_ids": ["c1"], "v_id": "vn"},
                {"uuid": "shared", "name": "Tools", "source_chunk_ids": ["c1", "c9"]},
                {"uuid": "unrelated", "name": "Other", "source_chunk_ids": ["c9"]},
            ],
            edges=[
                {"uuid": "only-ours", "name": "HAS", "source_chunk_ids": ["c1"]},
                {"uuid": "structured-edge", "name": "HAS", "source_chunk_ids": []},
            ],
            structured_data_id="structured-1",
            relationship_ids=["structured-edge"],
        )
        self.assertEqual(effect.created_chunk_ids, ["c1"])
        self.assertNotIn("unrelated", [node.uuid for node in effect.nodes])
        plan = plan_purge(
            effect,
            chunks={"c1": ChunkSnapshot(ingest_task_ids=[])},
            nodes={
                "exclusive": GraphSnapshot(source_chunk_ids=["c1"], name="Probe"),
                "shared": GraphSnapshot(source_chunk_ids=["c1", "c9"], name="Tools"),
            },
            edges={
                "only-ours": GraphSnapshot(source_chunk_ids=["c1"], name="HAS"),
                "structured-edge": GraphSnapshot(source_chunk_ids=[], name="HAS"),
            },
            claimed_chunks=set(),
        )
        kinds = {(action.kind, action.id) for action in plan.actions}
        self.assertIn(("delete_node", "exclusive"), kinds)
        self.assertIn(("unlink_node", "shared"), kinds)
        self.assertIn(("delete_edge", "structured-edge"), kinds)
        sink = _Sink()
        self.assertEqual(sink.deleted_chunks, [])
        self.assertTrue(plan.digest)


class ScreenshotSelectionTests(unittest.TestCase):
    def test_only_the_named_imports_are_selected(self):
        ops = _screenshot_ops()
        jobs = [
            {"brain_id": ops.TARGET_BRAIN, "label": "Slack message", "chars": 26, "id": "a"},
            {"brain_id": ops.TARGET_BRAIN, "label": "The harbor bell code is now 6612-WALRUS.", "chars": 40, "id": "b"},
            {"brain_id": ops.TARGET_BRAIN, "label": "<@U0CHWHK374K> has joined the channel.", "chars": 37, "id": "c"},
            {"brain_id": ops.TARGET_BRAIN, "label": "The harbor bell code is 2204-MINK.", "chars": 34, "id": "d"},
            {"brain_id": ops.TARGET_BRAIN, "label": "Connect your tools", "chars": 270, "id": "e"},
            {"brain_id": ops.TARGET_BRAIN, "label": "Connect your tools", "chars": 1273, "id": "f"},
            {"brain_id": ops.TARGET_BRAIN, "label": "Sync probe cedar", "chars": 66, "id": "g"},
            {"brain_id": ops.TARGET_BRAIN, "label": "Real customer note", "chars": 66, "id": "nope"},
            {"brain_id": "another-brain", "label": "Sync probe cedar", "chars": 66, "id": "other"},
        ]
        selected = [job["id"] for job in ops.select_screenshot_jobs(jobs)]
        self.assertEqual(selected, ["a", "b", "c", "d", "e", "f", "g"])
        self.assertFalse(ops.matches_screenshot_import("Real customer note", 66))


class SharedSourceKeyTests(unittest.TestCase):
    def test_unlinked_shared_source_key_chunks_cannot_be_purged(self):
        class _Chunk:
            def __init__(self, chunk_id, task_ids=None):
                self.id = chunk_id
                self.metadata = {"source_key": "src-shared", "ingest_task_ids": task_ids or []}

        older = _Chunk("c-older")
        other = _Chunk("c-other", ["task-other"])
        with self.assertRaises(PurgeUnsafe) as raised:
            chunks_linked_to_task([older, other], "task-old", source_key="src-shared")
        self.assertEqual(raised.exception.preview["reason"], "unlinked_source_key")
        self.assertEqual(raised.exception.preview["candidate_passages"], 2)
        self.assertEqual(raised.exception.preview["matched_passages"], 0)

    def test_task_id_match_keeps_only_this_import(self):
        class _Chunk:
            def __init__(self, chunk_id, task_ids):
                self.id = chunk_id
                self.metadata = {"source_key": "src-shared", "ingest_task_ids": task_ids}

        ours = _Chunk("c-ours", ["task-old"])
        theirs = _Chunk("c-theirs", ["task-other"])
        matched = chunks_linked_to_task([ours, theirs], "task-old", source_key="src-shared")
        self.assertEqual([chunk.id for chunk in matched], ["c-ours"])

    @patch("src.core.saving.ingest_effects._structured_payload", return_value=None)
    @patch("src.core.saving.ingest_effects._chunks_for_source")
    def test_reconstruct_live_refuses_shared_source_key_without_task_ids(
        self, chunks_for_source, _payload
    ):
        class _Chunk:
            def __init__(self, chunk_id):
                self.id = chunk_id
                self.metadata = {"source_key": "src-shared"}

        chunks_for_source.return_value = [_Chunk("c-old"), _Chunk("c-other")]
        with self.assertRaises(PurgeUnsafe) as raised:
            reconstruct_live("brain", "task-old", "src-shared")
        self.assertEqual(raised.exception.preview["reason"], "unlinked_source_key")
        self.assertIn("cannot be purged safely", str(raised.exception))


class LedgerAndPartialPurgeTests(unittest.TestCase):
    def test_remember_flags_incomplete_ledger_when_writes_fail(self):
        store = MagicMock()
        store.find_one.return_value = None
        store.insert_one.side_effect = RuntimeError("mongo down")
        store.update_one.return_value = MagicMock(modified_count=1)
        with patch("src.core.saving.ingest_effects._effects", return_value=store):
            remember("brain", "task-1", {"created_chunk_ids": ["c1"]})
        flagged = False
        for call in store.update_one.call_args_list:
            payload = call.args[1] if len(call.args) > 1 else {}
            if payload.get("$set", {}).get("ledger_incomplete") is True:
                flagged = True
                self.assertIn("mongo down", payload["$set"].get("ledger_error", ""))
        self.assertTrue(flagged)

    def test_preview_surfaces_incomplete_ledger(self):
        chunks, nodes, edges = _snapshots()
        effect = _effect()
        effect.ledger_incomplete = True
        effect.ledger_error = "mongo down"
        preview = handle_purge(effect, chunks=chunks, nodes=nodes, edges=edges)
        self.assertTrue(preview["ledger_incomplete"])
        self.assertEqual(preview["ledger_error"], "mongo down")
        self.assertFalse(preview["executed"])

    def test_partial_purge_is_reported_and_not_marked_executed(self):
        chunks, nodes, edges = _snapshots()
        sink = _Sink()
        sink.delete_nodes = MagicMock(side_effect=RuntimeError("graph down"))
        preview = handle_purge(_effect(), chunks=chunks, nodes=nodes, edges=edges)
        result = handle_purge(
            _effect(),
            chunks=chunks,
            nodes=nodes,
            edges=edges,
            digest=preview["digest"],
            sink=sink,
        )
        self.assertFalse(result["executed"])
        self.assertTrue(result["partial"])
        self.assertTrue(result["completed"])
        self.assertIn("graph down", result["error"])
        self.assertEqual(sink.deleted_chunks, ["c1"])
        self.assertEqual(sink.nodes, [])

    def test_apply_actions_wraps_mid_run_failure(self):
        chunks, nodes, edges = _snapshots()
        plan = plan_purge(_effect(), chunks=chunks, nodes=nodes, edges=edges, claimed_chunks=set())
        sink = _Sink()
        sink.delete_nodes = MagicMock(side_effect=RuntimeError("graph down"))
        with self.assertRaises(PurgePartial) as raised:
            apply_actions(plan, sink)
        self.assertTrue(raised.exception.completed)
        self.assertEqual(sink.deleted_chunks, ["c1"])


if __name__ == "__main__":
    unittest.main()

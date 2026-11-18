import unittest

from src.core.saving.ingest_effects import (
    ChunkSnapshot,
    DigestMismatch,
    EffectTouch,
    GraphSnapshot,
    IngestEffect,
    apply_actions,
    confirm_digest,
    handle_purge,
    matches_screenshot_import,
    plan_purge,
    reconstruct_effect,
    select_screenshot_jobs,
)


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
        jobs = [
            {"brain_id": "6aeae5d478a5d35c3d03a00e", "label": "Slack message", "chars": 26, "id": "a"},
            {"brain_id": "6aeae5d478a5d35c3d03a00e", "label": "The harbor bell code is now 6612-WALRUS.", "chars": 40, "id": "b"},
            {"brain_id": "6aeae5d478a5d35c3d03a00e", "label": "<@U0CHWHK374K> has joined the channel.", "chars": 37, "id": "c"},
            {"brain_id": "6aeae5d478a5d35c3d03a00e", "label": "The harbor bell code is 2204-MINK.", "chars": 34, "id": "d"},
            {"brain_id": "6aeae5d478a5d35c3d03a00e", "label": "Connect your tools", "chars": 270, "id": "e"},
            {"brain_id": "6aeae5d478a5d35c3d03a00e", "label": "Connect your tools", "chars": 1273, "id": "f"},
            {"brain_id": "6aeae5d478a5d35c3d03a00e", "label": "Sync probe cedar", "chars": 66, "id": "g"},
            {"brain_id": "6aeae5d478a5d35c3d03a00e", "label": "Real customer note", "chars": 66, "id": "nope"},
            {"brain_id": "another-brain", "label": "Sync probe cedar", "chars": 66, "id": "other"},
        ]
        selected = [job["id"] for job in select_screenshot_jobs(jobs)]
        self.assertEqual(selected, ["a", "b", "c", "d", "e", "f", "g"])
        self.assertFalse(matches_screenshot_import("Real customer note", 66))


if __name__ == "__main__":
    unittest.main()

"""Exercise staged publication with real PostgreSQL data and pgvector storage."""

import os
from uuid import uuid4

import pytest


@pytest.mark.parametrize("mode", ["keep_saved", "rollback"])
def test_staged_cancellation_with_postgresql_and_pgvector(mode, monkeypatch, tmp_path):
    if os.getenv("INGEST_TEST_POSTGRES") != "1":
        pytest.skip("Run with INGEST_TEST_POSTGRES=1 against an isolated PostgreSQL instance")

    from src.adapters.data import DataAdapter
    from src.adapters.embeddings import VectorStoreAdapter
    from src.constants.data import TextChunk
    from src.constants.embeddings import EMBEDDING_STORES_SIZES, Vector
    from src.core.saving import (
        ingest_access_gate,
        ingest_cancellation,
        ingest_runtime,
        ingest_staged_input,
        ingest_staged_output,
        ingest_staging,
        ingest_worker,
    )
    from src.core.saving.pipeline_store import PostgreSQLCollection
    from src.lib.postgresql.data import PostgreSQLDataClient
    from src.lib.postgresql.vectors import PostgreSQLVectorStoreClient, string_to_int64

    brain = f"staging{uuid4().hex}"
    data = DataAdapter()
    vectors = VectorStoreAdapter()
    data.add_client(PostgreSQLDataClient())
    vectors.add_client(PostgreSQLVectorStoreClient())
    monkeypatch.setenv("INGEST_PUBLICATION_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(ingest_cancellation, "_runs", lambda key: PostgreSQLCollection(data.data, key, "ingest_runs"))
    monkeypatch.setattr(ingest_access_gate, "_gates", lambda key: PostgreSQLCollection(data.data, key, "ingest_access_gates"))
    monkeypatch.setattr(ingest_staged_output, "collection", lambda key, name: PostgreSQLCollection(data.data, key, name))
    monkeypatch.setattr(ingest_staged_input, "collection", lambda key, name: PostgreSQLCollection(data.data, key, name))

    def stage():
        data.save_text_chunk(TextChunk(id="staged", text="new data"), brain)
        vectors.add_vectors([Vector(
            id="staged-vector",
            embeddings=[0.1] * EMBEDDING_STORES_SIZES["data"],
            metadata={"resource_id": "staged"},
        )], "data", brain)

    def stored_chunks():
        return data.get_text_chunks_by_ids(["existing", "staged"], False, brain)[0]

    def stored_vectors():
        return vectors.get_by_ids([string_to_int64("staged-vector")], "data", brain)

    data.save_text_chunk(TextChunk(id="existing", text="keep me"), brain)
    ingest_staging.create_batch_run(brain, "root", [{"payload": {}}, {"payload": {}}])
    monkeypatch.setattr(ingest_staged_input, "PART_SIZE", 4)
    part_count = ingest_staged_input.save_file(brain, "root", "c3RhZ2Vk")
    assert part_count == 2
    assert ingest_staged_input.load_file(brain, "root", part_count) == "c3RhZ2Vk"
    [(first, _), (second, _)] = ingest_staging.pending_dispatches(brain, "root")
    assert ingest_worker.execute_staged_unit(brain, "root", first, stage)
    assert [chunk.id for chunk in stored_chunks()] == ["existing"]
    assert not stored_vectors()

    ingest_cancellation.request_cancellation(brain, "root", mode)
    assert ingest_staging.claim_unit(brain, "root", second) is None
    ingest_runtime.recover_brain(brain, data.data, vectors.vector_store, None)

    assert ingest_cancellation._runs(brain).find_one({"_id": "root"})["status"] == "cancelled"
    assert PostgreSQLCollection(data.data, brain, "ingest_staged_inputs").count_documents({"import_id": "root"}) == 0
    expected = {"existing", "staged"} if mode == "keep_saved" else {"existing"}
    assert {chunk.id for chunk in stored_chunks()} == expected
    assert bool(stored_vectors()) == (mode == "keep_saved")


def test_small_imports_publish_concurrently_under_pool_pressure(monkeypatch, tmp_path):
    if os.getenv("INGEST_TEST_POSTGRES") != "1":
        pytest.skip("Run with INGEST_TEST_POSTGRES=1 against an isolated PostgreSQL instance")

    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from src.adapters.data import DataAdapter
    from src.constants.data import TextChunk
    from src.core.saving import (
        ingest_access_gate,
        ingest_cancellation,
        ingest_runtime,
        ingest_staged_input,
        ingest_staged_output,
        ingest_staging,
        ingest_worker,
    )
    from src.core.saving.pipeline_store import PostgreSQLCollection
    from src.lib.postgresql import _provisioning
    from src.lib.postgresql.data import PostgreSQLDataClient

    data = DataAdapter()
    data.add_client(PostgreSQLDataClient())

    def collection(brain, name):
        return PostgreSQLCollection(data.data, brain, name)

    monkeypatch.setenv("INGEST_PUBLICATION_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(ingest_cancellation, "_runs", lambda brain: collection(brain, "ingest_runs"))
    monkeypatch.setattr(ingest_access_gate, "_gates", lambda brain: collection(brain, "ingest_access_gates"))
    monkeypatch.setattr(ingest_staged_output, "collection", collection)
    monkeypatch.setattr(ingest_staged_input, "collection", collection)

    brains = [f"pooltest{uuid4().hex}" for _ in range(2)]
    for brain in brains:
        data.data._ensure_brain_schema(brain)
    monkeypatch.setattr(_provisioning, "_MAX_OPEN_POOLS", 1)
    monkeypatch.setattr(_provisioning, "_registry", _provisioning._PoolRegistry())
    concurrent_writes = Barrier(2, timeout=10)

    def ingest(brain):
        ingest_staging.create_batch_run(brain, "root", [{"payload": {}}])
        [(unit, _)] = ingest_staging.pending_dispatches(brain, "root")

        def stage():
            with _provisioning.borrow_brain(brain) as connection:
                concurrent_writes.wait()
                with connection.cursor() as cursor:
                    cursor.execute("SELECT current_database()")
                    assert cursor.fetchone()[0] == f"brain_{brain}"
                data.save_text_chunk(TextChunk(id="same-id", text=f"Small source for {brain}"), brain)

        assert ingest_worker.execute_staged_unit(brain, "root", unit, stage)
        assert not data.get_text_chunks_by_ids(["same-id"], False, brain)[0]
        ingest_runtime.recover_brain(brain, data.data, None, None)
        assert ingest_cancellation._runs(brain).find_one({"_id": "root"})["status"] == "completed"
        [chunk] = data.get_text_chunks_by_ids(["same-id"], False, brain)[0]
        return chunk.text

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(ingest, brains)) == [f"Small source for {brain}" for brain in brains]


def test_explore_reads_committed_graph_during_atomic_publication(monkeypatch, tmp_path):
    if os.getenv("INGEST_TEST_POSTGRES") != "1":
        pytest.skip("Run with INGEST_TEST_POSTGRES=1 against an isolated PostgreSQL instance")

    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from starlette.concurrency import run_in_threadpool
    from starlette.responses import JSONResponse

    from src.adapters.data import DataAdapter
    from src.adapters.embeddings import VectorStoreAdapter
    from src.constants.embeddings import EMBEDDING_STORES_SIZES, Vector
    from src.config import config
    from src.constants.data import TextChunk
    from src.constants.kg import Node
    from src.core.saving import (
        ingest_access_gate, ingest_cancellation, ingest_publication, ingest_runtime,
        ingest_staged_input, ingest_staged_output, ingest_staging, ingest_worker,
    )
    from src.core.saving.ingest_write_buffer import capture_operation
    from src.core.saving.pipeline_store import PostgreSQLCollection
    from src.lib.postgresql.data import PostgreSQLDataClient
    from src.lib.postgresql.networkx_client import NetworkXGraphClient
    from src.lib.postgresql.vectors import PostgreSQLVectorStoreClient, string_to_int64
    from src.services.api.middlewares.ingest_publication import IngestPublicationMiddleware
    from src.services.explore.store import atlas_store

    brain = f"atomic{uuid4().hex}"
    data = DataAdapter()
    data.add_client(PostgreSQLDataClient())
    writer = NetworkXGraphClient()
    reader = NetworkXGraphClient()
    vectors = VectorStoreAdapter()
    vectors.add_client(PostgreSQLVectorStoreClient())
    staged_vector = Vector(id="new-vector", embeddings=[0.1] * EMBEDDING_STORES_SIZES["data"], metadata={})

    def collection(brain_id, name):
        return PostgreSQLCollection(data.data, brain_id, name)

    monkeypatch.setenv("INGEST_PUBLICATION_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("STAGED_INGESTION_ENABLED", "1")
    monkeypatch.setenv("INGEST_POSTGRES_ATOMIC_PUBLICATION", "1")
    monkeypatch.setattr(config, "data_db", "postgresql")
    monkeypatch.setattr(config, "vector_db", "postgresql")
    monkeypatch.setattr(config, "graph_db", "networkx")
    monkeypatch.setattr(ingest_cancellation, "_runs", lambda key: collection(key, "ingest_runs"))
    monkeypatch.setattr(ingest_access_gate, "_gates", lambda key: collection(key, "ingest_access_gates"))
    monkeypatch.setattr(ingest_staged_input, "collection", collection)
    monkeypatch.setattr(ingest_staged_output, "collection", collection)

    writer.add_nodes([Node(uuid="entity", name="Original", labels=["PERSON"])], brain)
    assert reader._store.get_brain(brain).node_data("entity")["name"] == "Original"
    data.save_text_chunk(TextChunk(id="original", text="Committed"), brain)
    ingest_staging.create_batch_run(brain, "root", [{"payload": {}}, {"payload": {}}])
    [(first, _), (second, _)] = ingest_staging.pending_dispatches(brain, "root")

    assert ingest_access_gate.reserve_publication(brain, "legacy")
    with ingest_access_gate.publication_gate(brain, "legacy", atomic=True) as atomic:
        assert atomic is False
        with pytest.raises(ingest_access_gate.BrainPublicationBusy):
            with ingest_access_gate.brain_access(brain):
                pytest.fail("Legacy publication must keep its read lock")

    def stage_graph():
        capture_operation(brain, {
            "adapter": "graph",
            "method": "add_nodes",
            "args": {
                "nodes": [Node(uuid="entity", name="Updated", labels=["PERSON"]).model_dump(mode="json")],
                "identification_params": None,
                "metadata": None,
            },
        })
        data.save_text_chunk(TextChunk(id="new", text="New data"), brain)
        vectors.add_vectors([staged_vector], "data", brain)

    assert ingest_worker.execute_staged_unit(brain, "root", first, stage_graph)
    assert ingest_worker.execute_staged_unit(
        brain, "root", second, lambda: data.save_text_chunk(TextChunk(id="final", text="Final"), brain)
    )

    async def app(scope, receive, send):
        if scope["path"].endswith("overview"):
            result = {"atlas": (await run_in_threadpool(atlas_store().overview, brain))[0]}
        else:
            chunks = data.get_text_chunks_by_ids(["original", "new", "final"], False, brain)[0]
            result = {
                "name": reader._store.get_brain(brain).node_data("entity")["name"],
                "chunks": sorted(chunk.id for chunk in chunks),
                "vectors": len(vectors.get_by_ids([string_to_int64(staged_vector.id)], "data", brain)),
            }
        await JSONResponse(result)(scope, receive, send)

    async def request(method="GET", path="/retrieve/entities"):
        messages = []

        async def send(message):
            messages.append(message)

        await IngestPublicationMiddleware(app)(
            {"type": "http", "method": method, "path": path, "state": {"brain_id": brain}},
            None, send,
        )
        import json
        return messages[0]["status"], json.loads(messages[1]["body"])

    replay = ingest_publication.publish_next_unit
    applied = Event()
    resume = Event()

    def fail_after_first(*args):
        result = replay(*args)
        if result:
            applied.set()
            assert resume.wait(15)
            raise RuntimeError("injected after writing first unit")
        return result

    monkeypatch.setattr(ingest_publication, "publish_next_unit", fail_after_first)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(ingest_runtime.recover_brain, brain, data.data, vectors.vector_store, writer)
        try:
            assert applied.wait(15)
            assert asyncio.run(request()) == (200, {"name": "Original", "chunks": ["original"], "vectors": 0})
            assert asyncio.run(request(path="/retrieve/explore/v1/overview")) == (200, {"atlas": None})
            assert asyncio.run(request("POST"))[0] == 503
        finally:
            resume.set()
        running.result(timeout=15)

    run = ingest_cancellation._runs(brain).find_one({"_id": "root"})
    assert run["status"] == "publishing" and run["publication_cursor"] == 0
    assert asyncio.run(request()) == (200, {"name": "Original", "chunks": ["original"], "vectors": 0})

    monkeypatch.setattr(ingest_publication, "publish_next_unit", replay)
    ingest_runtime.recover_brain(brain, data.data, vectors.vector_store, writer)
    assert asyncio.run(request()) == (200, {"name": "Updated", "chunks": ["final", "new", "original"], "vectors": 1})
    assert ingest_cancellation._runs(brain).find_one({"_id": "root"})["status"] == "completed"


def test_shared_admission_limits_each_brain_and_recovers_queued_imports(monkeypatch, tmp_path):
    if os.getenv("INGEST_TEST_POSTGRES") != "1":
        pytest.skip("Run with INGEST_TEST_POSTGRES=1 against an isolated PostgreSQL instance")

    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from unittest.mock import Mock

    from src.core.saving import (
        ingest_access_gate, ingest_cancellation, ingest_staged_output, ingest_staging, ingest_worker,
    )
    from src.core.saving.pipeline_store import PostgreSQLCollection
    from src.lib.postgresql.data import PostgreSQLDataClient
    from src.workers.tasks.ingestion import ingest_data

    client = PostgreSQLDataClient()
    brain = f"admission{uuid4().hex}"
    other = f"admission{uuid4().hex}"
    monkeypatch.setenv("BRAINAPI_DEPLOYMENT_TIER", "shared")
    monkeypatch.setenv("INGEST_PUBLICATION_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(
        ingest_cancellation, "_runs", lambda key: PostgreSQLCollection(client, key, "ingest_runs"),
    )
    monkeypatch.setattr(
        ingest_staged_output, "collection", lambda key, name: PostgreSQLCollection(client, key, name),
    )
    sends = Mock()
    monkeypatch.setattr(ingest_data, "apply_async", sends)
    roots = {f"root-{n}": ingest_staging.create_run(brain, f"root-{n}", message={"payload": {}}) for n in range(3)}
    barrier = Barrier(3, timeout=10)

    def dispatch(root):
        barrier.wait()
        ingest_worker.dispatch_staged_source(brain, root, roots[root])

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(dispatch, roots))
    sent = {call.kwargs["headers"]["staged_ingestion_root"] for call in sends.call_args_list}
    assert len(sent) == sends.call_count == 2
    queued = (roots.keys() - sent).pop()
    assert ingest_staging.pending_dispatches(brain, queued)
    process = Mock()
    assert not ingest_worker.execute_staged_unit(brain, queued, roots[queued], process)
    process.assert_not_called()

    other_unit = ingest_staging.create_run(other, "other", message={"payload": {}})
    ingest_worker.dispatch_staged_source(other, "other", other_unit)
    assert sends.call_count == 3

    settled = sent.pop()
    ingest_cancellation.request_cancellation(brain, settled, "rollback")
    ingest_worker.dispatch_staged_source(brain, queued, roots[queued])
    assert sends.call_count == 3
    ingest_staging.finish_discard(brain, settled)
    ingest_worker.dispatch_staged_source(brain, queued, roots[queued])
    assert sends.call_count == 4
    assert sends.call_args.kwargs["task_id"] == roots[queued]

    with ingest_access_gate.ingestion_worker_slot(brain, 2) as first:
        with ingest_access_gate.ingestion_worker_slot(brain, 2) as second:
            assert first and second
            assert not ingest_worker.execute_staged_unit(brain, queued, roots[queued], process)
            process.assert_not_called()
            with ingest_access_gate.ingestion_worker_slot(other, 2) as independent:
                assert independent
    with ingest_access_gate.ingestion_worker_slot(brain, 2) as available:
        assert available

    work = [(brain, root, roots[root]) for root in [*sent, queued]]
    work.append((other, "other", other_unit))
    processing = Barrier(3, timeout=10)

    def execute(unit):
        return ingest_worker.execute_staged_unit(*unit, processing.wait)

    with ThreadPoolExecutor(max_workers=3) as pool:
        assert all(pool.map(execute, work))

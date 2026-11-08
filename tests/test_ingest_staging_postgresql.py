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

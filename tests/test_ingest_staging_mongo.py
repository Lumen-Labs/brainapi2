"""Staged publication integration using an isolated Mongo database."""

import os
from uuid import uuid4

import pytest
from pymongo import MongoClient

from src.adapters.data import DataAdapter
from src.constants.data import TextChunk
from src.core.saving import (
    ingest_access_gate,
    ingest_cancellation,
    ingest_runtime,
    ingest_staged_output,
    ingest_staging,
    ingest_worker,
)


@pytest.mark.parametrize("mode", ["keep_saved", "rollback"])
def test_cancel_staged_run_with_durable_mongo_storage(mode, monkeypatch, tmp_path):
    uri = os.getenv("INGEST_TEST_MONGO_URI")
    if not uri:
        pytest.skip("Set INGEST_TEST_MONGO_URI to run the Mongo staging integration")
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    brain = f"ingest_test_{uuid4().hex}"
    database = client[brain]
    monkeypatch.setenv("INGEST_PUBLICATION_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(ingest_cancellation, "_runs", lambda _: database.ingest_runs)
    monkeypatch.setattr(ingest_access_gate, "_gates", lambda _: database.ingest_access_gates)
    monkeypatch.setattr(ingest_staged_output, "collection", lambda *_: database.ingest_staged_outputs)

    class DataClient:
        def get_collection(self, name, database):
            return client[database][name]

    adapter = DataAdapter()
    adapter.add_client(DataClient())
    try:
        database.text_chunks.insert_one({"_id": "shared", "text": "already present"})
        ingest_staging.create_batch_run(brain, "root", [{"payload": {}}, {"payload": {}}])
        [(first, _), (second, _)] = ingest_staging.pending_dispatches(brain, "root")
        assert ingest_worker.execute_staged_unit(
            brain, "root", first,
            lambda: adapter.save_text_chunk(TextChunk(id="staged", text="new data"), brain),
        )
        assert database.text_chunks.count_documents({}) == 1
        accepted = ingest_cancellation.request_cancellation(brain, "root", mode)
        assert ingest_cancellation.cancellation_status(accepted) == "cancelling"
        assert ingest_staging.claim_unit(brain, "root", second) is None

        ingest_runtime.recover_brain(brain, adapter.data, None, None)
        assert database.ingest_runs.find_one({"_id": "root"})["status"] == "cancelled"
        assert database.text_chunks.find_one({"_id": "shared"})["text"] == "already present"
        assert bool(database.text_chunks.find_one({"_id": "staged"})) == (mode == "keep_saved")
        with ingest_access_gate.brain_access(brain):
            pass
    finally:
        client.drop_database(brain)
        client.close()

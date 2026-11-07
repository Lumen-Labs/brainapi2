"""Staged publication integration using an isolated Mongo database."""

import os
import subprocess
import sys
from uuid import uuid4

import pytest
from pymongo import MongoClient

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
    monkeypatch.setattr(ingest_staged_input, "collection", lambda *_: database.ingest_staged_inputs)

    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from src.services.api.dependencies import get_brain_id
    from src.services.api.routes import tasks

    monkeypatch.setenv("STAGED_INGESTION_ENABLED", "1")
    monkeypatch.setattr(tasks.cache_adapter, "get_task", lambda *args, **kwargs: None)

    def brain_from_request(request: Request):
        return request.headers["X-Brain-ID"]

    app = FastAPI()
    app.include_router(tasks.tasks_router)
    app.dependency_overrides[get_brain_id] = brain_from_request
    api = TestClient(app, headers={"X-Brain-ID": brain})
    other = f"{brain}_other"
    monkeypatch.setattr(ingest_cancellation, "_runs", lambda key: client[key].ingest_runs)

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
        assert api.get("/tasks/root").json()["cancellation_modes"] == ["keep_saved", "rollback"]
        unauthorized = api.post("/tasks/root/cancel", json={"mode": mode}, headers={"X-Brain-ID": other})
        assert unauthorized.status_code == 404
        accepted = api.post("/tasks/root/cancel", json={"mode": mode})
        assert accepted.status_code == 202
        assert accepted.json()["status"] == "cancelling"
        assert ingest_staging.claim_unit(brain, "root", second) is None

        # Simulate a publisher dying after persisting the reservation.
        assert ingest_access_gate.reserve_publication(brain, "root")
        code = (
            "import fcntl, os, sys; "
            "lock = open(sys.argv[1], 'a'); "
            "fcntl.flock(lock, fcntl.LOCK_EX); "
            "os._exit(7)"
        )
        import hashlib

        lock_path = tmp_path / hashlib.sha256(brain.encode()).hexdigest()
        interrupted = subprocess.run([sys.executable, "-c", code, str(lock_path)], check=False)
        assert interrupted.returncode == 7
        with pytest.raises(ingest_access_gate.BrainPublicationBusy), ingest_access_gate.brain_access(brain):
            pytest.fail("The durable reservation must survive publisher exit")

        ingest_runtime.recover_brain(brain, adapter.data, None, None)
        assert api.get("/tasks/root").json()["status"] == "cancelled"
        assert database.ingest_staged_outputs.count_documents({"import_id": "root"}) == 0
        root = database.ingest_runs.find_one({"_id": "root"})
        assert all("message" not in unit for unit in root["units"].values())
        assert database.text_chunks.find_one({"_id": "shared"})["text"] == "already present"
        assert bool(database.text_chunks.find_one({"_id": "staged"})) == (mode == "keep_saved")
        with ingest_access_gate.brain_access(brain):
            pass
    finally:
        client.drop_database(brain)
        client.close()


def test_large_docling_file_uses_durable_input_and_discards_it_on_cancel(monkeypatch, tmp_path):
    import base64
    from unittest.mock import Mock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.services.api.dependencies import get_brain_id
    from src.services.api.routes import ingest

    uri = os.getenv("INGEST_TEST_MONGO_URI")
    if not uri:
        pytest.skip("Set INGEST_TEST_MONGO_URI to run the Mongo staging integration")
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    brain = f"ingest_test_{uuid4().hex}"
    database = client[brain]
    monkeypatch.setenv("INGEST_PUBLICATION_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("STAGED_INGESTION_ENABLED", "1")
    monkeypatch.setenv("STAGED_ENRICHMENT_ENABLED", "1")
    monkeypatch.setattr(ingest.config, "ocr_mode", "docling")
    monkeypatch.setattr(ingest, "set_ingestion_task_status", Mock())
    monkeypatch.setattr(ingest_cancellation, "_runs", lambda _: database.ingest_runs)
    monkeypatch.setattr(ingest_access_gate, "_gates", lambda _: database.ingest_access_gates)
    monkeypatch.setattr(ingest_staged_output, "collection", lambda *_: database.ingest_staged_outputs)
    monkeypatch.setattr(ingest_staged_input, "collection", lambda *_: database.ingest_staged_inputs)
    dispatch = Mock()
    monkeypatch.setattr(ingest_worker, "dispatch_staged_source", dispatch)

    app = FastAPI()
    app.include_router(ingest.ingest_router)
    app.dependency_overrides[get_brain_id] = lambda: brain
    file = b"x" * (13 * 1024 * 1024)
    try:
        response = TestClient(app).post("/ingest/file", files={"file": ("large.txt", file)})
        assert response.status_code == 202
        root = response.json()["task_id"]
        [(unit_id, message)] = ingest_staging.pending_dispatches(brain, root)
        dispatch.assert_called_once_with(brain, root, unit_id)
        payload = message["payload"]
        assert not payload["content_b64"]
        assert payload["content_parts"] > 1
        assert base64.b64decode(ingest_staged_input.load_file(brain, root, payload["content_parts"])) == file

        ingest_cancellation.request_cancellation(brain, root, "rollback")
        ingest_runtime.recover_brain(brain, None, None, None)
        assert database.ingest_runs.find_one({"_id": root})["status"] == "cancelled"
        assert database.ingest_staged_inputs.count_documents({"import_id": root}) == 0

        ingest_staging.create_run(brain, "interrupted", upload_deadline=1)
        ingest_staged_input.save_file(brain, "interrupted", "cGFydGlhbA==")
        ingest_runtime.recover_brain(brain, None, None, None)
        assert database.ingest_runs.find_one({"_id": "interrupted"})["status"] == "failed"
        assert database.ingest_staged_inputs.count_documents({"import_id": "interrupted"}) == 0
    finally:
        client.drop_database(brain)
        client.close()

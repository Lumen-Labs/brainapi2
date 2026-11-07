from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from celery.exceptions import Ignore

from src.core.saving import ingest_cancellation as cancellation


@pytest.fixture
def run(monkeypatch):
    documents = {}

    def find_one(query):
        row = documents.get(query["_id"])
        return row.copy() if row else None

    def update_one(query, update, upsert=False):
        task_id = query["_id"]
        if upsert:
            documents.setdefault(task_id, {"_id": task_id, **update["$setOnInsert"]})
        row = documents[task_id]
        if "revision" in query and row["revision"] != query["revision"]:
            return SimpleNamespace(modified_count=0)
        row.update(update.get("$set", {}))
        return SimpleNamespace(modified_count=1)

    monkeypatch.setenv("INGESTION_CANCELLATION_ENABLED", "1")
    monkeypatch.setattr(
        cancellation, "_runs",
        lambda brain_id: SimpleNamespace(find_one=find_one, update_one=update_one)
        if brain_id == "brain" else Mock(find_one=Mock(return_value=None)),
    )
    cancellation.register("brain", "root")
    return documents["root"]


def test_queued_cancellation_blocks_worker_and_is_idempotent(run):
    assert cancellation.request_cancellation("other", "root", "keep_saved") is None
    accepted = cancellation.request_cancellation("brain", "root", "keep_saved")
    assert accepted["status"] == "cancelled"
    assert cancellation.request_cancellation("brain", "root", "keep_saved") == accepted
    callback = Mock()
    worker = cancellation.tracked_worker(lambda: "brain")(callback)
    with pytest.raises(Ignore):
        worker(SimpleNamespace(request=SimpleNamespace(headers={"ingestion_root": "root"})))
    callback.assert_not_called()


def test_running_worker_settles_before_cancellation_finishes(run):
    @cancellation.tracked_worker(lambda: "brain")
    def worker(task):
        accepted = cancellation.request_cancellation("brain", "root", "keep_saved")
        assert accepted["status"] == "cancelling"
        cancellation.record_terminal("brain", "root", "completed")
        cancellation.checkpoint()

    with pytest.raises(Ignore):
        worker(SimpleNamespace(request=SimpleNamespace(headers={"ingestion_root": "root"})))
    assert run["status"] == "cancelled"
    assert not run["workers"]


def test_completion_wins_and_rollback_is_unavailable(run):
    with pytest.raises(cancellation.CancellationConflict):
        cancellation.request_cancellation("brain", "root", "rollback")
    assert run["status"] == "active"
    cancellation.record_terminal("brain", "root", "completed")
    with pytest.raises(cancellation.CancellationConflict):
        cancellation.request_cancellation("brain", "root", "keep_saved")

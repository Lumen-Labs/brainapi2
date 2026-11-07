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


def test_expired_staging_claim_can_be_redispatched(run, monkeypatch):
    from src.core.saving import ingest_staging as staging

    now = 1000
    monkeypatch.setattr(staging, "time", lambda: now)
    staging.create_run("brain", "staged")
    unit = staging.register_unit("brain", "staged", "page:1", {"task": "page"})
    assert staging.register_unit("brain", "staged", "page:1", {}) == unit
    first = staging.claim_unit("brain", "staged", unit)
    assert staging.claim_unit("brain", "staged", unit) is None
    assert staging.pending_dispatches("brain", "staged") == []

    now += staging.CLAIM_SECONDS
    assert staging.pending_dispatches("brain", "staged") == [(unit, {"task": "page"})]
    second = staging.claim_unit("brain", "staged", unit)
    assert second and first != second
    assert not staging.renew_claim("brain", "staged", unit, first)
    assert not staging.complete_unit("brain", "staged", unit, first, "stale")
    assert staging.complete_unit("brain", "staged", unit, second, "current")


@pytest.mark.parametrize(("mode", "status"), [
    ("keep_saved", "publishing"), ("rollback", "discarding"),
])
def test_staged_cancellation_fences_late_completion(run, mode, status):
    from src.core.saving import ingest_staging as staging

    staging.create_run("brain", "staged")
    unit = staging.register_unit("brain", "staged", "page:1", {})
    claim = staging.claim_unit("brain", "staged", unit)
    assert staging.close_staging("brain", "staged")["status"] == "staging"
    assert staging.close_staging("brain", "staged", mode)["status"] == status
    assert not staging.complete_unit("brain", "staged", unit, claim, "late")
    assert staging.register_unit("brain", "staged", "page:2", {}) is None
    assert staging.pending_dispatches("brain", "staged") == []

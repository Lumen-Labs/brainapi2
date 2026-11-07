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


def test_publication_waits_for_discovery_and_resumes_from_cursor(run):
    from src.core.saving import ingest_staging as staging

    staging.create_run("brain", "staged")
    for key in ("first", "second"):
        unit = staging.register_unit("brain", "staged", key, {})
        claim = staging.claim_unit("brain", "staged", unit)
        assert staging.complete_unit("brain", "staged", unit, claim, key)

    assert staging.close_staging("brain", "staged")["status"] == "staging"
    staging.seal_units("brain", "staged")
    assert staging.register_unit("brain", "staged", "late", {}) is None
    assert staging.close_staging("brain", "staged")["status"] == "publishing"
    assert staging.close_staging("brain", "staged", "rollback")["mode"] is None

    first = staging.next_publication_unit("brain", "staged")
    assert first["output_ref"] == "first"
    assert staging.next_publication_unit("brain", "staged") == first
    staging.advance_publication("brain", "staged", first["cursor"])
    staging.advance_publication("brain", "staged", first["cursor"])
    second = staging.next_publication_unit("brain", "staged")
    assert second["output_ref"] == "second"
    assert second["operation_id"] != first["operation_id"]
    staging.advance_publication("brain", "staged", second["cursor"])
    assert staging.next_publication_unit("brain", "staged") is None


@pytest.mark.parametrize("mode", ["keep_saved", "rollback"])
def test_staged_cancellation_reports_pending_until_finalized(run, mode):
    from src.core.saving import ingest_staging as staging

    staging.create_run("brain", "staged")
    accepted = cancellation.request_cancellation("brain", "staged", mode)
    assert cancellation.cancellation_status(accepted) == "cancelling"
    assert cancellation.request_cancellation("brain", "staged", mode) == accepted
    other_mode = "keep_saved" if mode == "rollback" else "rollback"
    with pytest.raises(cancellation.CancellationConflict):
        cancellation.request_cancellation("brain", "staged", other_mode)
    finish = staging.finish_discard if mode == "rollback" else staging.finish_publication
    finish("brain", "staged")
    assert cancellation.public_state("brain", "staged", {})["status"] == "cancelled"


def test_reconciler_retries_dispatch_and_discard(run, monkeypatch, tmp_path):
    from src.core.saving import ingest_access_gate as gate, ingest_recovery, ingest_staging as staging

    monkeypatch.setenv("INGEST_PUBLICATION_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(gate, "_gates", cancellation._runs)

    staging.create_run("brain", "staged")
    unit = staging.register_unit("brain", "staged", "page:1", {"task": "page"})
    dispatch = Mock()
    publish = Mock()
    ingest_recovery.reconcile_run("brain", "staged", dispatch, publish)
    dispatch.assert_called_once_with("brain", "staged", unit, {"task": "page"})

    discard = Mock(side_effect=[RuntimeError("interrupted"), None])
    monkeypatch.setattr(ingest_recovery, "discard_outputs", discard)
    with gate.brain_access("brain"):
        cancellation.request_cancellation("brain", "staged", "rollback")
        with pytest.raises(gate.BrainPublicationBusy):
            ingest_recovery.reconcile_run("brain", "staged", dispatch, publish)
        discard.assert_not_called()

    with pytest.raises(RuntimeError, match="interrupted"):
        ingest_recovery.reconcile_run("brain", "staged", dispatch, publish)
    assert cancellation.public_state("brain", "staged", {})["status"] == "cancelling"

    ingest_recovery.reconcile_run("brain", "staged", dispatch, publish)
    assert cancellation.public_state("brain", "staged", {})["status"] == "cancelled"
    assert discard.call_count == 2
    assert dispatch.call_count == 1
    publish.assert_not_called()


def test_rollback_failure_is_terminal_and_reopens_access(run, monkeypatch, tmp_path):
    from src.core.saving import ingest_access_gate as gate, ingest_recovery, ingest_staging as staging

    monkeypatch.setenv("INGEST_PUBLICATION_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(gate, "_gates", cancellation._runs)
    monkeypatch.setattr(ingest_recovery, "discard_outputs", Mock(side_effect=RuntimeError("cleanup failed")))

    staging.create_run("brain", "staged", message={"payload": {"text": "private"}})
    cancellation.request_cancellation("brain", "staged", "rollback")
    for _ in range(staging.MAX_ROLLBACK_FAILURES - 1):
        with pytest.raises(RuntimeError, match="cleanup failed"):
            ingest_recovery.reconcile_run("brain", "staged", Mock(), Mock())
        assert cancellation.public_state("brain", "staged", {})["status"] == "cancelling"

    ingest_recovery.reconcile_run("brain", "staged", Mock(), Mock())
    assert cancellation.public_state("brain", "staged", {})["status"] == "rollback_failed"
    assert cancellation.request_cancellation("brain", "staged", "rollback")["status"] == "rollback_failed"
    with gate.brain_access("brain"):
        pass


def test_publication_drains_existing_access_and_rejects_new_access(run, monkeypatch, tmp_path):
    from src.core.saving import ingest_access_gate as gate

    monkeypatch.setattr(gate, "_gates", cancellation._runs)
    monkeypatch.setenv("INGEST_PUBLICATION_LOCK_DIR", str(tmp_path))
    with gate.brain_access("brain"):
        with pytest.raises(gate.BrainPublicationBusy), gate.publication_gate("brain", "root"):
            pytest.fail("Publication must wait for existing access")
        with pytest.raises(gate.BrainPublicationBusy), gate.brain_access("brain"):
            pytest.fail("Access must be blocked during publication")
    assert gate.reserve_publication("brain", "root")
    assert not gate.reserve_publication("brain", "other")
    gate.release_publication("brain", "other")
    with pytest.raises(gate.BrainPublicationBusy), gate.brain_access("brain"):
        pytest.fail("Another run cannot release the gate")
    with gate.publication_gate("brain", "root"):
        pass
    with gate.brain_access("brain"):
        pass


def test_background_writer_retries_during_publication(run, monkeypatch, tmp_path):
    from src.core.saving import ingest_access_gate as gate

    monkeypatch.setenv("STAGED_INGESTION_ENABLED", "1")
    monkeypatch.setenv("INGEST_PUBLICATION_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(gate, "_gates", cancellation._runs)
    process = Mock(return_value="completed")
    worker = gate.guard_worker(lambda brain: brain)(process)
    task = Mock()
    task.retry.side_effect = RuntimeError("retry scheduled")

    with gate.publication_gate("brain", "root"):
        with pytest.raises(RuntimeError, match="retry scheduled"):
            worker(task, "brain")
        process.assert_not_called()

    assert worker(task, "brain") == "completed"
    task.retry.assert_called_once()


def test_discovery_registers_pages_atomically_and_seals_run(run):
    from src.core.saving import ingest_staging as staging

    staging.create_run("brain", "staged")
    discovery = staging.register_unit("brain", "staged", "discovery", {"kind": "url_discovery"})
    token = staging.claim_unit("brain", "staged", discovery)
    messages = [{"payload": {"text": "page one"}}, {"payload": {"text": "page two"}}]
    result = staging.complete_discovery("brain", "staged", discovery, token, messages)
    assert result["sealed"]
    pending = staging.pending_dispatches("brain", "staged")
    assert [message for _, message in pending] == messages
    assert discovery not in {unit for unit, _ in pending}
    cancellation.request_cancellation("brain", "staged", "rollback")
    assert staging.pending_dispatches("brain", "staged") == []


def test_dispatch_reservation_retries_after_window_without_blocking_worker(run, monkeypatch):
    from src.core.saving import ingest_staging as staging

    now = 1000
    monkeypatch.setattr(staging, "time", lambda: now)
    staging.create_run("brain", "staged")
    message = {"payload": {"text": "one"}}
    unit = staging.register_unit("brain", "staged", "source", message)
    assert staging.reserve_dispatch("brain", "staged", unit) == message
    assert staging.reserve_dispatch("brain", "staged", unit) is None
    assert staging.pending_dispatches("brain", "staged") == []
    now += staging.DISPATCH_RETRY_SECONDS
    assert staging.pending_dispatches("brain", "staged") == [(unit, message)]
    assert staging.reserve_dispatch("brain", "staged", unit) == message
    assert staging.claim_unit("brain", "staged", unit)
    now += staging.DISPATCH_RETRY_SECONDS
    assert staging.reserve_dispatch("brain", "staged", unit) is None


@pytest.mark.parametrize("discovery", [False, True])
def test_staged_submission_survives_initial_dispatch_failure(run, monkeypatch, discovery):
    from celery.exceptions import OperationalError

    from src.core.saving import ingest_staging, ingest_worker
    from src.services.api.routes.ingest import _submit_staged

    now = 1000
    monkeypatch.setattr(ingest_staging, "time", lambda: now)

    def unavailable(brain_id, root_id, unit_id):
        assert ingest_staging.reserve_dispatch(brain_id, root_id, unit_id)["payload"] == {"text": "one"}
        raise OperationalError("broker unavailable")

    monkeypatch.setattr(ingest_worker, "dispatch_staged_source", unavailable)
    _submit_staged("brain", "staged", {"text": "one"}, None, discovery=discovery)

    stored = cancellation._runs("brain").find_one({"_id": "staged"})
    assert stored["sealed"] is not discovery
    now += ingest_staging.DISPATCH_RETRY_SECONDS
    [(unit_id, message)] = ingest_staging.pending_dispatches("brain", "staged")
    assert ingest_staging.reserve_dispatch("brain", "staged", unit_id) == message


def test_staged_unit_stops_retrying_after_failure_budget(run, monkeypatch):
    from src.core.saving import ingest_staging as staging

    now = 1000
    monkeypatch.setattr(staging, "time", lambda: now)
    staging.create_run("brain", "staged")
    unit = staging.register_unit("brain", "staged", "source", {"payload": {}})
    for _ in range(staging.MAX_UNIT_FAILURES):
        token = staging.claim_unit("brain", "staged", unit)
        assert token
        staging.record_unit_failure("brain", "staged", unit, token)
        assert staging.pending_dispatches("brain", "staged") == []
        now += staging.DISPATCH_RETRY_SECONDS

    assert staging.claim_unit("brain", "staged", unit) is None
    assert staging.finish_discard("brain", "staged")["status"] == "failed"
    assert cancellation.public_state("brain", "staged", {})["status"] == "failed"


def test_remote_ocr_callback_cannot_revive_cancelled_run(run):
    from src.core.saving import ingest_staging as staging

    staging.create_run("brain", "ocr", callback_deadline=1000)
    assert staging.is_remote_ocr_run("brain", "ocr")
    cancellation.request_cancellation("brain", "ocr", "rollback")
    assert staging.register_unit("brain", "ocr", "source", {"payload": {}}, seal=True) is None
    assert staging.pending_dispatches("brain", "ocr") == []


def test_batch_has_one_cancellation_cutoff_for_every_unit(run):
    from src.core.saving import ingest_staging as staging

    staging.create_batch_run("brain", "batch", [{"payload": {}}, {"payload": {}}])
    [(first, _), (second, _)] = staging.pending_dispatches("brain", "batch")
    claim = staging.claim_unit("brain", "batch", first)
    assert staging.complete_unit("brain", "batch", first, claim, "first-output")
    assert staging.close_staging("brain", "batch")["status"] == "staging"
    cancellation.request_cancellation("brain", "batch", "keep_saved")
    assert staging.claim_unit("brain", "batch", second) is None
    assert staging.next_publication_unit("brain", "batch")["output_ref"] == "first-output"


def test_ocr_callback_registers_work_and_rejects_late_payload(run, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.core.saving import ingest_staging, ingest_worker
    from src.services.api.dependencies import get_brain_id
    from src.services.api.routes import ingest

    monkeypatch.setenv("STAGED_INGESTION_ENABLED", "1")
    ingest_staging.create_run("brain", "ocr", callback_deadline=1000)
    dispatch = Mock()
    set_status = Mock()
    monkeypatch.setattr(ingest_worker, "dispatch_staged_source", dispatch)
    monkeypatch.setattr(ingest, "set_ingestion_task_status", set_status)

    app = FastAPI()
    app.include_router(ingest.ingest_router)
    app.dependency_overrides[get_brain_id] = lambda: "brain"
    api = TestClient(app)
    payload = {"data": {"data_type": "text", "text_data": "Extracted document"}}
    headers = {"Task-Identifier": "ocr"}

    accepted = api.post("/ingest/", json=payload, headers=headers)
    assert accepted.status_code == 202
    assert accepted.json()["task_id"] == "ocr"
    [(unit, message)] = ingest_staging.pending_dispatches("brain", "ocr")
    assert message["payload"]["data"]["text_data"] == "Extracted document"
    dispatch.assert_called_once_with("brain", "ocr", unit)
    assert cancellation._runs("brain").find_one({"_id": "ocr"})["sealed"]

    cancellation.request_cancellation("brain", "ocr", "rollback")
    late = api.post("/ingest/", json=payload, headers=headers)
    assert late.status_code == 409
    assert dispatch.call_count == 1
    set_status.assert_not_called()

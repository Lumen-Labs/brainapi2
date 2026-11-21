import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.saving.ingest_plain_text import (
    decode_plain_text,
    spreadsheet_experiment_owns_upload,
    uses_plain_text_ingest,
)
from src.core.saving.ingest_staging import (
    REMOTE_OCR_DEADLINE_SECONDS,
    REMOTE_OCR_TIMEOUT_ERROR,
    SMALL_REMOTE_OCR_DEADLINE_SECONDS,
    SMALL_REMOTE_OCR_MAX_BYTES,
    fail_expired_remote_ocr,
    remote_ocr_callback_deadline,
)


def _ingest_app(brain_id="brain"):
    from src.services.api.dependencies import get_brain_id
    from src.services.api.routes import ingest

    app = FastAPI()
    app.include_router(ingest.ingest_router)
    app.dependency_overrides[get_brain_id] = lambda: brain_id
    return app, ingest


def _tasks_app(brain_id="brain"):
    from src.services.api.dependencies import get_brain_id
    from src.services.api.routes import tasks

    app = FastAPI()
    app.include_router(tasks.tasks_router)
    app.dependency_overrides[get_brain_id] = lambda: brain_id
    return app, tasks


def test_plain_text_classification_and_utf8_decode():
    assert uses_plain_text_ingest("b", "notes.txt", "application/octet-stream")
    assert uses_plain_text_ingest("b", "notes.MD", "text/plain")
    assert uses_plain_text_ingest("b", "rows.csv", "text/csv")
    assert uses_plain_text_ingest("b", "blob", "application/json")
    assert uses_plain_text_ingest("b", "notes", "text/plain; charset=utf-8")
    assert not uses_plain_text_ingest("b", "report.pdf", "application/pdf")
    assert not uses_plain_text_ingest("b", "page.html", "text/html")
    assert not uses_plain_text_ingest("b", "notes.txt", "text/plain", b"\x00binary")
    assert decode_plain_text(b"\xef\xbb\xbfhello") == "hello"
    assert "\ufffd" in decode_plain_text(b"hi \xff there")


def test_spreadsheet_experiment_claims_csv_ahead_of_text_path(monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "sheet-brain")
    assert spreadsheet_experiment_owns_upload("sheet-brain", "rows.csv", "text/csv")
    assert not uses_plain_text_ingest("sheet-brain", "rows.csv", "text/csv")
    assert uses_plain_text_ingest("other-brain", "rows.csv", "text/csv")


def test_small_files_use_shorter_remote_ocr_deadline(monkeypatch):
    monkeypatch.setattr("src.core.saving.ingest_staging.time", lambda: 1_000)
    assert remote_ocr_callback_deadline(77) == 1_000 + SMALL_REMOTE_OCR_DEADLINE_SECONDS
    assert remote_ocr_callback_deadline(SMALL_REMOTE_OCR_MAX_BYTES) == (
        1_000 + SMALL_REMOTE_OCR_DEADLINE_SECONDS
    )
    assert remote_ocr_callback_deadline(SMALL_REMOTE_OCR_MAX_BYTES + 1) == (
        1_000 + REMOTE_OCR_DEADLINE_SECONDS
    )
    assert remote_ocr_callback_deadline(None) == 1_000 + REMOTE_OCR_DEADLINE_SECONDS


def test_txt_upload_uses_text_ingest_and_skips_docparser(monkeypatch):
    app, ingest = _ingest_app()
    queued = Mock()
    parser = Mock(side_effect=AssertionError("docparser must not be called"))
    monkeypatch.setattr(ingest.config, "ocr_mode", "docparser")
    monkeypatch.setattr(ingest.config, "docparser_endpoint", "http://docparser.example")
    monkeypatch.setattr(ingest.config, "docparser_token", "parser-token")
    monkeypatch.setattr(ingest, "set_ingestion_task_status", Mock())
    monkeypatch.setattr(ingest.ingest_data_task, "apply_async", queued)
    monkeypatch.setattr(ingest.ingest_file_task, "apply_async", Mock())
    monkeypatch.setattr(ingest.requests, "post", parser)

    response = TestClient(app).post(
        "/ingest/file",
        files={"file": ("notes.txt", b"\xef\xbb\xbfplain text body", "text/plain")},
    )
    assert response.status_code == 202
    assert response.json()["message"] == "File ingestion accepted"
    queued.assert_called_once()
    body = queued.call_args.kwargs["args"][0]
    assert body["data"]["data_type"] == "text"
    assert body["data"]["text_data"] == "plain text body"
    parser.assert_not_called()
    ingest.ingest_file_task.apply_async.assert_not_called()


@pytest.mark.parametrize(
    "filename,content_type,body",
    [
        ("readme.md", "text/markdown", b"# title"),
        ("rows.csv", "text/csv", b"a,b\n1,2"),
        ("payload.json", "application/json", b'{"ok": true}'),
        ("notes", "text/plain", b"hello"),
    ],
)
def test_plain_text_variants_skip_docparser(monkeypatch, filename, content_type, body):
    app, ingest = _ingest_app()
    queued = Mock()
    parser = Mock(side_effect=AssertionError("docparser must not be called"))
    monkeypatch.setattr(ingest.config, "ocr_mode", "docparser")
    monkeypatch.setattr(ingest, "set_ingestion_task_status", Mock())
    monkeypatch.setattr(ingest.ingest_data_task, "apply_async", queued)
    monkeypatch.setattr(ingest.requests, "post", parser)
    response = TestClient(app).post(
        "/ingest/file",
        files={"file": (filename, body, content_type)},
    )
    assert response.status_code == 202
    queued.assert_called_once()
    parser.assert_not_called()


def test_pdf_upload_still_goes_to_docparser(monkeypatch):
    app, ingest = _ingest_app()
    parser = Mock(return_value=SimpleNamespace(status_code=200))
    queued_text = Mock()
    monkeypatch.setattr(ingest.config, "ocr_mode", "docparser")
    monkeypatch.setattr(ingest.config, "docparser_endpoint", "http://docparser.example")
    monkeypatch.setattr(ingest.config, "docparser_token", "parser-token")
    monkeypatch.setattr(ingest, "set_ingestion_task_status", Mock())
    monkeypatch.setattr(ingest.ingest_data_task, "apply_async", queued_text)
    monkeypatch.setattr(ingest.requests, "post", parser)

    response = TestClient(app).post(
        "/ingest/file",
        files={"file": ("report.pdf", b"%PDF-1.4 mock", "application/pdf")},
    )
    assert response.status_code == 202
    parser.assert_called_once()
    queued_text.assert_not_called()
    sent = parser.call_args.kwargs["files"]["file"]
    assert sent[0] == "report.pdf"


def test_csv_falls_through_when_spreadsheet_experiment_owns_brain(monkeypatch):
    app, ingest = _ingest_app("sheet-brain")
    parser = Mock(return_value=SimpleNamespace(status_code=200))
    queued_text = Mock()
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "sheet-brain")
    monkeypatch.setattr(ingest.config, "ocr_mode", "docparser")
    monkeypatch.setattr(ingest.config, "docparser_endpoint", "http://docparser.example")
    monkeypatch.setattr(ingest.config, "docparser_token", "parser-token")
    monkeypatch.setattr(ingest, "set_ingestion_task_status", Mock())
    monkeypatch.setattr(ingest.ingest_data_task, "apply_async", queued_text)
    monkeypatch.setattr(ingest.requests, "post", parser)

    response = TestClient(app).post(
        "/ingest/file",
        files={"file": ("rows.csv", b"a,b\n1,2", "text/csv")},
    )
    assert response.status_code == 202
    parser.assert_called_once()
    queued_text.assert_not_called()


def test_fail_expired_remote_ocr_updates_cached_task_status(monkeypatch):
    monkeypatch.setattr(
        "src.core.saving.ingest_staging.ingest_cancellation._runs",
        lambda brain_id: SimpleNamespace(find_one=lambda query: None),
    )
    recorded = {}

    def fake_mark(brain_id, task_id, error):
        recorded["payload"] = {
            "status": "failed",
            "stage": "failed",
            "error": error,
            "task_id": task_id,
        }
        return recorded["payload"]

    monkeypatch.setattr("src.core.saving.ingest_staging._mark_remote_ocr_failed", fake_mark)
    monkeypatch.setattr("src.core.saving.ingest_staging.time", lambda: 5_000)
    result = fail_expired_remote_ocr(
        "brain",
        "ocr-task",
        {"status": "queued", "remote_ocr_deadline": 10, "task_id": "ocr-task"},
    )
    assert result["status"] == "failed"
    assert result["error"] == REMOTE_OCR_TIMEOUT_ERROR
    assert recorded["payload"]["status"] == "failed"


def test_fail_expired_remote_ocr_keeps_accepted_pages(monkeypatch):
    monkeypatch.setattr("src.core.saving.ingest_staging.time", lambda: 5_000)
    result = fail_expired_remote_ocr(
        "brain",
        "ocr-task",
        {
            "status": "queued",
            "remote_ocr_deadline": 10,
            "ocr_pages_accepted": True,
            "task_id": "ocr-task",
        },
    )
    assert result["status"] == "queued"


def test_set_ingestion_task_status_does_not_flip_completed_to_failed():
    from src.workers.tasks.ingestion import set_ingestion_task_status

    stored = {}

    class FakeCache:
        def get(self, key, brain_id="default"):
            payload = stored.get((key, brain_id))
            return json.dumps(payload) if payload else None

        def get_task(self, task_id, brain_id="default"):
            return self.get(f"task:{task_id}", brain_id=brain_id)

        def set(self, key, value, brain_id="default", expires_in=None):
            stored[(key, brain_id)] = json.loads(value)

    import src.workers.tasks.ingestion as ingestion

    original = ingestion.cache_adapter
    ingestion.cache_adapter = FakeCache()
    try:
        set_ingestion_task_status("t1", "brain", "completed", stage="completed")
        set_ingestion_task_status(
            "t1", "brain", "failed", stage="failed", error="Success callback failed"
        )
        assert stored[("task:t1", "brain")]["status"] == "completed"
        set_ingestion_task_status(
            "t2", "brain", "queued", extra={"ocr_pages_accepted": True}
        )
        set_ingestion_task_status(
            "t2", "brain", "failed", stage="failed", error="later failure"
        )
        assert stored[("task:t2", "brain")]["status"] == "queued"
        assert stored[("task:t2", "brain")]["ocr_pages_accepted"] is True
    finally:
        ingestion.cache_adapter = original


def test_get_task_reports_failed_after_remote_ocr_deadline(monkeypatch):
    app, tasks = _tasks_app()
    cached = {
        "status": "queued",
        "stage": "queued",
        "task_id": "ocr-task",
        "remote_ocr_deadline": 10,
    }
    monkeypatch.setattr(
        tasks.cache_adapter,
        "get_task",
        lambda task_id, brain_id="default": json.dumps(cached),
    )
    monkeypatch.setattr(
        "src.core.saving.ingest_staging.ingest_cancellation._runs",
        lambda brain_id: SimpleNamespace(find_one=lambda query: None),
    )
    monkeypatch.setattr(
        "src.core.saving.ingest_cancellation._runs",
        lambda brain_id: SimpleNamespace(find_one=lambda query: None),
    )
    monkeypatch.setattr("src.core.saving.ingest_staging.time", lambda: 5_000)

    def fake_mark(brain_id, task_id, error):
        return {
            "status": "failed",
            "stage": "failed",
            "error": error,
            "task_id": task_id,
        }

    monkeypatch.setattr("src.core.saving.ingest_staging._mark_remote_ocr_failed", fake_mark)
    monkeypatch.delenv("INGESTION_CANCELLATION_ENABLED", raising=False)
    monkeypatch.delenv("STAGED_INGESTION_ENABLED", raising=False)

    response = TestClient(app).get("/tasks/ocr-task")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"] == REMOTE_OCR_TIMEOUT_ERROR
    assert body["task_id"] == "ocr-task"

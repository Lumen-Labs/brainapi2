import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.saving.ingest_plain_text import (
    decode_plain_text,
    is_blank_plain_text,
    is_spreadsheet_upload,
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
    assert decode_plain_text(b"hi \xff there") == "hi ÿ there"
    assert decode_plain_text(b"cost \x80 5") == "cost € 5"
    assert is_blank_plain_text("notes.txt", "  \n\t")
    assert is_blank_plain_text("README.md", "")
    assert is_blank_plain_text("rows.csv", "   ")
    assert not is_blank_plain_text("payload.json", "")
    assert not is_blank_plain_text("notes.txt", "hello")


def test_spreadsheet_uploads_keep_text_fallback_after_importer(monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "sheet-brain")
    assert is_spreadsheet_upload("rows.csv", "text/csv")
    assert is_spreadsheet_upload("book.xlsx", "application/octet-stream")
    assert not is_spreadsheet_upload("rows.tsv", "text/tab-separated-values")
    assert not is_spreadsheet_upload("notes.txt", "text/csv")
    assert uses_plain_text_ingest("sheet-brain", "rows.csv", "text/csv")
    assert uses_plain_text_ingest("sheet-brain", "rows.tsv", "text/tab-separated-values")
    assert uses_plain_text_ingest("sheet-brain", "notes.txt", "text/csv")
    assert uses_plain_text_ingest("other-brain", "rows.csv", "text/csv")


def test_small_files_keep_long_ocr_backstop_until_first_page(monkeypatch):
    monkeypatch.setattr("src.core.saving.ingest_staging.time", lambda: 1_000)
    assert remote_ocr_callback_deadline(77) == 1_000 + REMOTE_OCR_DEADLINE_SECONDS
    assert remote_ocr_callback_deadline(SMALL_REMOTE_OCR_MAX_BYTES) == (
        1_000 + REMOTE_OCR_DEADLINE_SECONDS
    )
    assert remote_ocr_callback_deadline(SMALL_REMOTE_OCR_MAX_BYTES + 1) == (
        1_000 + REMOTE_OCR_DEADLINE_SECONDS
    )
    assert remote_ocr_callback_deadline(None) == 1_000 + REMOTE_OCR_DEADLINE_SECONDS
    assert SMALL_REMOTE_OCR_DEADLINE_SECONDS < REMOTE_OCR_DEADLINE_SECONDS


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
    parser = Mock(side_effect=AssertionError("docparser must not get declined spreadsheets"))
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
    queued_text.assert_called_once()
    body = queued_text.call_args.kwargs["args"][0]
    assert body["data"]["text_data"] == "a,b\n1,2"
    parser.assert_not_called()


def test_fail_expired_remote_ocr_updates_cached_task_status(monkeypatch):
    cache, _store, _hashes = _writable_cache({})
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
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
    cache, _store, _hashes = _writable_cache({
        ("task:ocr-task", "brain"): cached,
    })
    monkeypatch.setattr(tasks, "cache_adapter", cache)
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
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


def test_fail_expired_remote_ocr_does_not_timeout_started_tasks(monkeypatch):
    monkeypatch.setattr(
        "src.core.saving.ingest_staging.ingest_cancellation._runs",
        lambda brain_id: SimpleNamespace(find_one=lambda query: None),
    )
    monkeypatch.setattr("src.core.saving.ingest_staging.time", lambda: 5_000)
    result = fail_expired_remote_ocr(
        "brain",
        "ocr-task",
        {
            "status": "started",
            "stage": "processing",
            "remote_ocr_deadline": 10,
            "task_id": "ocr-task",
        },
    )
    assert result["status"] == "started"
    assert result.get("error") is None


def test_completed_status_clears_previous_timeout_error():
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
        set_ingestion_task_status(
            "t3",
            "brain",
            "failed",
            stage="failed",
            error=REMOTE_OCR_TIMEOUT_ERROR,
        )
        set_ingestion_task_status("t3", "brain", "completed", stage="completed")
        payload = stored[("task:t3", "brain")]
        assert payload["status"] == "completed"
        assert "error" not in payload
    finally:
        ingestion.cache_adapter = original


def test_get_task_keeps_started_after_remote_ocr_deadline(monkeypatch):
    app, tasks = _tasks_app()
    cached = {
        "status": "started",
        "stage": "processing",
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
    monkeypatch.delenv("INGESTION_CANCELLATION_ENABLED", raising=False)
    monkeypatch.delenv("STAGED_INGESTION_ENABLED", raising=False)

    response = TestClient(app).get("/tasks/ocr-task")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "started"
    assert body.get("error") in (None, "")


def test_unstaged_failed_callback_accepted_for_known_task(monkeypatch):
    app, ingest = _ingest_app()
    monkeypatch.delenv("STAGED_INGESTION_ENABLED", raising=False)
    monkeypatch.setattr(ingest, "set_ingestion_task_status", Mock())

    class FakeCache:
        def get(self, key, brain_id="default"):
            if key == "task:known":
                return json.dumps({"status": "started", "task_id": "known"})
            return None

        def get_task(self, task_id, brain_id="default"):
            return self.get(f"task:{task_id}", brain_id=brain_id)

    monkeypatch.setattr(ingest, "cache_adapter", FakeCache())
    api = TestClient(app)
    known = api.post(
        "/ingest/",
        json={"status": "failed", "error": "Success callback failed", "brain_id": "brain"},
        headers={"Task-Identifier": "known"},
    )
    assert known.status_code == 202
    unknown = api.post(
        "/ingest/",
        json={"status": "failed", "error": "nope", "brain_id": "brain"},
        headers={"Task-Identifier": "missing"},
    )
    assert unknown.status_code == 422
    missing_header = api.post(
        "/ingest/",
        json={"status": "failed", "error": "nope", "brain_id": "brain"},
    )
    assert missing_header.status_code == 422
    ingest.set_ingestion_task_status.assert_called()
    failed_call = ingest.set_ingestion_task_status.call_args_list[-1]
    assert failed_call.args[2] == "failed"
    assert "Success callback failed" in failed_call.kwargs["error"]


def _writable_cache(initial):
    store = dict(initial)
    hashes = {}

    class FakeCache:
        def get(self, key, brain_id="default"):
            payload = store.get((key, brain_id))
            return json.dumps(payload) if payload else None

        def get_task(self, task_id, brain_id="default"):
            return self.get(f"task:{task_id}", brain_id=brain_id)

        def set(self, key, value, brain_id="default", expires_in=None):
            store[(key, brain_id)] = json.loads(value) if isinstance(value, str) else value

        def _bucket(self, key, brain_id):
            return hashes.setdefault((key, brain_id), {})

        def hset(self, key, field, value, brain_id="default"):
            bucket = self._bucket(key, brain_id)
            bucket[str(field)] = value
            return 1

        def hset_unless_locked(self, key, lock_field, field, value, brain_id="default"):
            bucket = self._bucket(key, brain_id)
            if bucket.get(str(lock_field)):
                return False
            bucket[str(field)] = value
            return True

        def hsetnx(self, key, field, value, brain_id="default"):
            bucket = self._bucket(key, brain_id)
            field = str(field)
            if field in bucket:
                return False
            bucket[field] = value
            return True

        def hgetall(self, key, brain_id="default"):
            return dict(hashes.get((key, brain_id), {}))

        def hdel(self, key, field, brain_id="default"):
            bucket = hashes.get((key, brain_id), {})
            return 1 if bucket.pop(str(field), None) is not None else 0

        def hincrby(self, key, field, amount=1, brain_id="default"):
            bucket = self._bucket(key, brain_id)
            field = str(field)
            bucket[field] = int(bucket.get(field) or 0) + int(amount)
            return bucket[field]

        def hmax(self, key, field, value, brain_id="default"):
            bucket = self._bucket(key, brain_id)
            field = str(field)
            current = int(bucket.get(field) or 0)
            incoming = int(value)
            if incoming > current:
                bucket[field] = incoming
                return incoming
            return current

        def expire(self, key, seconds, brain_id="default"):
            return True

    return FakeCache(), store, hashes


def test_unstaged_failed_callback_caps_error_at_2kb(monkeypatch):
    app, ingest = _ingest_app()
    monkeypatch.delenv("STAGED_INGESTION_ENABLED", raising=False)
    cache, store, _hashes = _writable_cache({
        ("task:known", "brain"): {"status": "queued", "task_id": "known"},
    })
    monkeypatch.setattr(ingest, "cache_adapter", cache)
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    huge = "x" * 5000
    response = TestClient(app).post(
        "/ingest/",
        json={"status": "failed", "error": huge, "brain_id": "brain"},
        headers={"Task-Identifier": "known"},
    )
    assert response.status_code == 202
    payload = store[("task:known", "brain")]
    assert payload["status"] == "failed"
    assert len(payload["error"]) == 2000


def _silence_remote_ocr_runs(monkeypatch):
    empty = SimpleNamespace(find_one=lambda query: None, update_one=lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "src.core.saving.ingest_staging.ingest_cancellation._runs",
        lambda brain_id: empty,
    )
    monkeypatch.setattr(
        "src.core.saving.ingest_cancellation._runs",
        lambda brain_id: empty,
    )
    monkeypatch.delenv("STAGED_INGESTION_ENABLED", raising=False)


def test_unstaged_multipage_callbacks_group_into_one_ingest(monkeypatch):
    app, ingest = _ingest_app()
    _silence_remote_ocr_runs(monkeypatch)
    cache, store, hashes = _writable_cache({
        ("task:ocr", "brain"): {"status": "queued", "task_id": "ocr"},
    })
    monkeypatch.setattr(ingest, "cache_adapter", cache)
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    queued = Mock()
    monkeypatch.setattr(ingest.ingest_data_task, "apply_async", queued)
    api = TestClient(app)
    headers = {"Task-Identifier": "ocr"}
    first = api.post(
        "/ingest/",
        json={
            "data": {"data_type": "text", "text_data": "page one"},
            "page_index": 0,
            "total_pages": 2,
        },
        headers=headers,
    )
    assert first.status_code == 202
    queued.assert_not_called()
    payload = store[("task:ocr", "brain")]
    assert payload["status"] == "started"
    assert "ocr_page_texts" not in payload
    assert hashes[("ocr_pages:ocr", "brain")]["p:0"] == "page one"
    second = api.post(
        "/ingest/",
        json={
            "data": {"data_type": "text", "text_data": "page two"},
            "page_index": 1,
            "total_pages": 2,
        },
        headers=headers,
    )
    assert second.status_code == 202
    queued.assert_called_once()
    body = queued.call_args.kwargs["args"][0]
    assert body["data"]["text_data"] == "page one\n\npage two"
    assert "p:0" not in hashes.get(("ocr_pages:ocr", "brain"), {})
    assert "p:1" not in hashes.get(("ocr_pages:ocr", "brain"), {})
    assert store[("task:ocr", "brain")].get("ocr_dispatch_started") is True


def test_empty_text_uploads_are_rejected(monkeypatch):
    app, ingest = _ingest_app()
    monkeypatch.setattr(ingest, "set_ingestion_task_status", Mock())
    monkeypatch.setattr(ingest.ingest_data_task, "apply_async", Mock())
    api = TestClient(app)
    for filename, content_type, body in (
        ("notes.txt", "text/plain", b"   \n"),
        ("README.md", "text/markdown", b""),
        ("rows.csv", "text/csv", b"\t  "),
    ):
        response = api.post(
            "/ingest/file",
            files={"file": (filename, body, content_type)},
        )
        assert response.status_code == 422, filename
    ingest.ingest_data_task.apply_async.assert_not_called()


def test_cp1252_text_upload_is_decoded(monkeypatch):
    app, ingest = _ingest_app()
    queued = Mock()
    monkeypatch.setattr(ingest, "set_ingestion_task_status", Mock())
    monkeypatch.setattr(ingest.ingest_data_task, "apply_async", queued)
    response = TestClient(app).post(
        "/ingest/file",
        files={"file": ("notes.txt", b"caf\xe9", "text/plain")},
    )
    assert response.status_code == 202
    body = queued.call_args.kwargs["args"][0]
    assert body["data"]["text_data"] == "café"


def test_redis_outage_returns_503_on_ingest_callback_and_tasks(monkeypatch):
    from redis.exceptions import ConnectionError as RedisConnectionError

    app, ingest = _ingest_app()
    monkeypatch.delenv("STAGED_INGESTION_ENABLED", raising=False)

    def boom(*_args, **_kwargs):
        raise RedisConnectionError("redis down")

    monkeypatch.setattr(ingest.cache_adapter, "get", boom)
    monkeypatch.setattr(ingest.cache_adapter, "get_task", boom)
    ingest_api = TestClient(app, raise_server_exceptions=False)
    callback = ingest_api.post(
        "/ingest/",
        json={"status": "failed", "error": "nope", "brain_id": "brain"},
        headers={"Task-Identifier": "known"},
    )
    assert callback.status_code == 503
    assert callback.headers.get("retry-after") == "5"

    file_app, file_ingest = _ingest_app()
    monkeypatch.setattr(file_ingest, "set_ingestion_task_status", boom)
    file_response = TestClient(file_app, raise_server_exceptions=False).post(
        "/ingest/file",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert file_response.status_code == 503
    assert file_response.headers.get("retry-after") == "5"

    tasks_app, tasks = _tasks_app()
    monkeypatch.setattr(tasks.cache_adapter, "get_task", boom)
    task_response = TestClient(tasks_app, raise_server_exceptions=False).get("/tasks/ocr-task")
    assert task_response.status_code == 503
    assert task_response.headers.get("retry-after") == "5"


def test_completed_status_keeps_partial_failed_ocr(monkeypatch):
    from src.workers.tasks.ingestion import set_ingestion_task_status

    cache, store, _hashes = _writable_cache({})
    import src.workers.tasks.ingestion as ingestion

    original = ingestion.cache_adapter
    ingestion.cache_adapter = cache
    try:
        set_ingestion_task_status(
            "t-partial",
            "brain",
            "partial_failed",
            stage="partial_failed",
            error="OCR finished with 1 of 3 pages accepted; 0 failed, 2 missing",
            extra={"ocr_partial": True, "ocr_pages_accepted": 1, "ocr_total_pages": 3},
        )
        set_ingestion_task_status("t-partial", "brain", "completed", stage="completed")
        payload = store[("task:t-partial", "brain")]
        assert payload["status"] == "partial_failed"
        assert payload.get("ocr_partial") is True
    finally:
        ingestion.cache_adapter = original


def test_customer_task_strips_internal_ocr_fields():
    from src.services.api.routes.tasks import customer_task

    result = customer_task(
        {
            "status": "started",
            "ocr_page_texts": {"0": "full document text"},
            "ocr_failed_pages": ["1"],
            "ocr_dispatch_started": True,
            "ocr_pages_accepted": 1,
            "ocr_pages_failed": 1,
            "ocr_partial": True,
        }
    )
    assert "ocr_page_texts" not in result
    assert "ocr_failed_pages" not in result
    assert "ocr_dispatch_started" not in result
    assert result["ocr_pages_accepted"] == 1
    assert result["ocr_pages_failed"] == 1
    assert result["ocr_partial"] is True


def test_get_task_strips_ocr_page_texts(monkeypatch):
    app, tasks = _tasks_app()
    _silence_remote_ocr_runs(monkeypatch)
    cached = {
        "status": "started",
        "stage": "processing",
        "task_id": "ocr-task",
        "remote_ocr_deadline": 10**12,
        "ocr_page_texts": {"0": "full document text"},
        "ocr_failed_pages": ["1"],
        "ocr_pages_accepted": 1,
        "ocr_dispatch_started": False,
    }
    monkeypatch.setattr(
        tasks.cache_adapter,
        "get_task",
        lambda task_id, brain_id="default": json.dumps(cached),
    )
    response = TestClient(app).get("/tasks/ocr-task")
    assert response.status_code == 200
    body = response.json()
    assert "ocr_page_texts" not in body
    assert "ocr_failed_pages" not in body
    assert body["ocr_pages_accepted"] == 1
    assert body["status"] == "started"


def test_expired_unstaged_pages_are_ingested_once(monkeypatch):
    cache, store, hashes = _writable_cache({
        ("task:ocr", "brain"): {
            "status": "started",
            "task_id": "ocr",
            "remote_ocr_deadline": 10,
            "ocr_total_pages": 3,
        },
    })
    hashes[("ocr_pages:ocr", "brain")] = {"p:0": "kept page", "total": "3"}
    _silence_remote_ocr_runs(monkeypatch)
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    queued = Mock()
    monkeypatch.setattr("src.workers.tasks.ingestion.ingest_data.apply_async", queued)
    monkeypatch.setattr("src.core.saving.unstaged_ocr.time", lambda: 5_000)
    monkeypatch.setattr("src.core.saving.ingest_staging.time", lambda: 5_000)
    first = fail_expired_remote_ocr(
        "brain", "ocr", dict(store[("task:ocr", "brain")]),
    )
    second = fail_expired_remote_ocr(
        "brain", "ocr", dict(store[("task:ocr", "brain")]),
    )
    queued.assert_called_once()
    assert queued.call_args.kwargs["args"][0]["data"]["text_data"] == "kept page"
    assert first["status"] == "started"
    assert first["ocr_partial"] is True
    assert "p:0" not in hashes.get(("ocr_pages:ocr", "brain"), {})
    assert store[("task:ocr", "brain")].get("ocr_dispatch_started") is True
    assert second["status"] == "started"
    assert second["ocr_partial"] is True


def test_expired_unstaged_without_total_pages_is_ingested(monkeypatch):
    cache, store, hashes = _writable_cache({
        ("task:ocr", "brain"): {
            "status": "started",
            "task_id": "ocr",
            "remote_ocr_deadline": 10,
        },
    })
    hashes[("ocr_pages:ocr", "brain")] = {"p:0": "page one", "p:1": "page two"}
    _silence_remote_ocr_runs(monkeypatch)
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    queued = Mock()
    monkeypatch.setattr("src.workers.tasks.ingestion.ingest_data.apply_async", queued)
    monkeypatch.setattr("src.core.saving.unstaged_ocr.time", lambda: 5_000)
    result = fail_expired_remote_ocr(
        "brain", "ocr", dict(store[("task:ocr", "brain")]),
    )
    queued.assert_called_once()
    assert queued.call_args.kwargs["args"][0]["data"]["text_data"] == "page one\n\npage two"
    assert result["status"] == "started"
    assert result["ocr_partial"] is True


def test_expired_unstaged_with_no_accepted_pages_fails(monkeypatch):
    cache, store, hashes = _writable_cache({
        ("task:ocr", "brain"): {
            "status": "started",
            "task_id": "ocr",
            "remote_ocr_deadline": 10,
        },
    })
    hashes[("ocr_pages:ocr", "brain")] = {"f:0": "1", "f:1": "1", "total": "2"}
    _silence_remote_ocr_runs(monkeypatch)
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    queued = Mock()
    monkeypatch.setattr("src.workers.tasks.ingestion.ingest_data.apply_async", queued)
    monkeypatch.setattr("src.core.saving.unstaged_ocr.time", lambda: 5_000)
    result = fail_expired_remote_ocr(
        "brain", "ocr", dict(store[("task:ocr", "brain")]),
    )
    queued.assert_not_called()
    assert result["status"] == "failed"
    assert store[("task:ocr", "brain")]["status"] == "failed"


def test_get_task_dispatches_expired_unstaged_pages(monkeypatch):
    app, tasks = _tasks_app()
    cache, store, hashes = _writable_cache({
        ("task:ocr-task", "brain"): {
            "status": "started",
            "stage": "processing",
            "task_id": "ocr-task",
            "remote_ocr_deadline": 10,
            "ocr_total_pages": 2,
        },
    })
    hashes[("ocr_pages:ocr-task", "brain")] = {"p:0": "kept page", "total": "2"}
    _silence_remote_ocr_runs(monkeypatch)
    monkeypatch.setattr(tasks, "cache_adapter", cache)
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    queued = Mock()
    monkeypatch.setattr("src.workers.tasks.ingestion.ingest_data.apply_async", queued)
    monkeypatch.setattr("src.core.saving.unstaged_ocr.time", lambda: 5_000)
    monkeypatch.setattr("src.core.saving.ingest_staging.time", lambda: 5_000)
    response = TestClient(app).get("/tasks/ocr-task")
    assert response.status_code == 200
    body = response.json()
    queued.assert_called_once()
    assert body["status"] == "started"
    assert body.get("ocr_partial") is True
    assert "ocr_page_texts" not in body
    assert "p:0" not in hashes.get(("ocr_pages:ocr-task", "brain"), {})


def test_unstaged_page_hash_keeps_concurrent_pages(monkeypatch):
    cache, _store, _hashes = _writable_cache({
        ("task:ocr", "brain"): {"status": "queued", "task_id": "ocr"},
    })
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    from src.core.saving import unstaged_ocr

    unstaged_ocr.record_page("brain", "ocr", "0", "one")
    unstaged_ocr.record_page("brain", "ocr", "1", "two")
    snapshot = unstaged_ocr.load_pages("brain", "ocr")
    assert snapshot.pages == {"0": "one", "1": "two"}
    assert unstaged_ocr.claim_dispatch("brain", "ocr") is True
    assert unstaged_ocr.claim_dispatch("brain", "ocr") is False


def test_unstaged_mixed_pages_dispatch_partial(monkeypatch):
    app, ingest = _ingest_app()
    _silence_remote_ocr_runs(monkeypatch)
    cache, store, hashes = _writable_cache({
        ("task:ocr", "brain"): {"status": "queued", "task_id": "ocr"},
    })
    monkeypatch.setattr(ingest, "cache_adapter", cache)
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    queued = Mock()
    monkeypatch.setattr(ingest.ingest_data_task, "apply_async", queued)
    api = TestClient(app)
    headers = {"Task-Identifier": "ocr"}
    ok = api.post(
        "/ingest/",
        json={
            "data": {"data_type": "text", "text_data": "kept"},
            "page_index": 0,
            "total_pages": 2,
        },
        headers=headers,
    )
    failed = api.post(
        "/ingest/",
        json={"status": "failed", "error": "blank page", "page_index": 1, "total_pages": 2},
        headers=headers,
    )
    assert ok.status_code == 202
    assert failed.status_code == 202
    queued.assert_called_once()
    assert queued.call_args.kwargs["args"][0]["data"]["text_data"] == "kept"
    payload = store[("task:ocr", "brain")]
    assert payload["ocr_partial"] is True
    assert payload["status"] == "started"
    assert "p:0" not in hashes.get(("ocr_pages:ocr", "brain"), {})


def test_replayed_complete_callback_does_not_ingest_twice(monkeypatch):
    app, ingest = _ingest_app()
    _silence_remote_ocr_runs(monkeypatch)
    cache, _store, _hashes = _writable_cache({
        ("task:ocr", "brain"): {"status": "queued", "task_id": "ocr"},
    })
    monkeypatch.setattr(ingest, "cache_adapter", cache)
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    queued = Mock()
    monkeypatch.setattr(ingest.ingest_data_task, "apply_async", queued)
    api = TestClient(app)
    payload = {
        "data": {"data_type": "text", "text_data": "only page"},
        "page_index": 0,
        "total_pages": 1,
    }
    headers = {"Task-Identifier": "ocr"}
    first = api.post("/ingest/", json=payload, headers=headers)
    second = api.post("/ingest/", json=payload, headers=headers)
    assert first.status_code == 202
    assert second.status_code == 202
    queued.assert_called_once()


def test_expired_ocr_does_not_finalize_when_page_hash_unreadable(monkeypatch):
    cache, store, hashes = _writable_cache({
        ("task:ocr", "brain"): {
            "status": "started",
            "task_id": "ocr",
            "remote_ocr_deadline": 10,
            "ocr_pages_accepted": 1,
            "ocr_total_pages": 2,
        },
    })
    hashes[("ocr_pages:ocr", "brain")] = {"p:0": "kept page", "total": "2"}

    def boom(*_args, **_kwargs):
        raise RuntimeError("redis down")

    cache.hgetall = boom
    _silence_remote_ocr_runs(monkeypatch)
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    queued = Mock()
    monkeypatch.setattr("src.workers.tasks.ingestion.ingest_data.apply_async", queued)
    monkeypatch.setattr("src.core.saving.unstaged_ocr.time", lambda: 5_000)
    monkeypatch.setattr("src.core.saving.ingest_staging.time", lambda: 5_000)
    result = fail_expired_remote_ocr(
        "brain", "ocr", dict(store[("task:ocr", "brain")]),
    )
    queued.assert_not_called()
    assert result["status"] == "started"
    assert store[("task:ocr", "brain")]["status"] == "started"


def test_dispatch_releases_claim_when_setup_fails(monkeypatch):
    cache, store, hashes = _writable_cache({
        ("task:ocr", "brain"): {"status": "started", "task_id": "ocr"},
    })
    hashes[("ocr_pages:ocr", "brain")] = {"p:0": "kept"}
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)

    def boom(*_args, **_kwargs):
        raise RuntimeError("ttl failed")

    cache.expire = boom
    from src.core.saving import unstaged_ocr

    snapshot = unstaged_ocr.Snapshot({"0": "kept"}, [], 1)
    with pytest.raises(RuntimeError):
        unstaged_ocr.dispatch_snapshot(
            "brain", "ocr", store[("task:ocr", "brain")], snapshot,
        )
    assert "lock" not in hashes[("ocr_pages:ocr", "brain")]
    assert unstaged_ocr.claim_dispatch("brain", "ocr") is True


def test_anon_pages_join_in_numeric_order():
    from src.core.saving import unstaged_ocr

    pages = {f"anon:{index}": f"p{index}" for index in range(12)}
    assert unstaged_ocr.joined_text(pages) == "\n\n".join(
        f"p{index}" for index in range(12)
    )


def test_pages_after_dispatch_claim_are_not_stored(monkeypatch):
    cache, _store, hashes = _writable_cache({
        ("task:ocr", "brain"): {"status": "queued", "task_id": "ocr"},
    })
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    from src.core.saving import unstaged_ocr

    unstaged_ocr.record_page("brain", "ocr", "0", "one")
    snapshot = unstaged_ocr.load_pages("brain", "ocr")
    queued = Mock()
    monkeypatch.setattr("src.workers.tasks.ingestion.ingest_data.apply_async", queued)
    assert unstaged_ocr.dispatch_snapshot("brain", "ocr", {"status": "started"}, snapshot)
    assert unstaged_ocr.record_page("brain", "ocr", "1", "late") is False
    assert "p:1" not in hashes.get(("ocr_pages:ocr", "brain"), {})
    queued.assert_called_once()
    assert queued.call_args.kwargs["args"][0]["data"]["text_data"] == "one"


def test_dispatch_keeps_live_snapshot_page_counts(monkeypatch):
    cache, _store, hashes = _writable_cache({
        ("task:ocr", "brain"): {"status": "started", "task_id": "ocr"},
    })
    hashes[("ocr_pages:ocr", "brain")] = {"p:0": "one", "p:1": "two", "total": "2"}
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    queued = Mock()
    monkeypatch.setattr("src.workers.tasks.ingestion.ingest_data.apply_async", queued)
    from src.core.saving import unstaged_ocr

    stale = unstaged_ocr.Snapshot({"0": "one"}, [], 2)
    extra = unstaged_ocr.task_extra({}, stale, closing=True)
    extra["ocr_partial"] = True
    result = unstaged_ocr.dispatch_snapshot(
        "brain", "ocr", {"status": "started"}, stale, extra=extra,
    )
    queued.assert_called_once()
    assert queued.call_args.kwargs["args"][0]["data"]["text_data"] == "one\n\ntwo"
    assert result["ocr_pages_accepted"] == 2
    assert result["ocr_partial"] is True


def test_clear_page_text_keeps_pages_not_in_snapshot(monkeypatch):
    cache, _store, hashes = _writable_cache({})
    hashes[("ocr_pages:ocr", "brain")] = {"p:0": "queued", "p:1": "late", "lock": "1"}
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    from src.core.saving import unstaged_ocr

    unstaged_ocr.clear_page_text(
        "brain", "ocr", unstaged_ocr.Snapshot({"0": "queued"}, [], 2, True),
    )
    leftover = hashes[("ocr_pages:ocr", "brain")]
    assert "p:0" not in leftover
    assert leftover["p:1"] == "late"
    assert leftover["lock"] == "1"


def test_started_ocr_partial_completes_as_partial_failed(monkeypatch):
    from src.workers.tasks.ingestion import set_ingestion_task_status

    cache, store, _hashes = _writable_cache({})
    import src.workers.tasks.ingestion as ingestion

    original = ingestion.cache_adapter
    ingestion.cache_adapter = cache
    try:
        set_ingestion_task_status(
            "t-partial",
            "brain",
            "started",
            stage="processing",
            extra={"ocr_partial": True, "ocr_pages_accepted": 1, "ocr_total_pages": 2},
        )
        set_ingestion_task_status("t-partial", "brain", "completed", stage="completed")
        payload = store[("task:t-partial", "brain")]
        assert payload["status"] == "partial_failed"
        assert payload.get("ocr_partial") is True
    finally:
        ingestion.cache_adapter = original


def test_unstaged_partial_dispatch_does_not_record_terminal(monkeypatch):
    cache, store, hashes = _writable_cache({
        ("task:ocr", "brain"): {
            "status": "started",
            "task_id": "ocr",
            "remote_ocr_deadline": 10,
            "ocr_total_pages": 2,
        },
    })
    hashes[("ocr_pages:ocr", "brain")] = {"p:0": "kept page", "total": "2"}
    _silence_remote_ocr_runs(monkeypatch)
    monkeypatch.setenv("INGESTION_CANCELLATION_ENABLED", "1")
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    recorded = []
    monkeypatch.setattr(
        "src.workers.tasks.ingestion.record_terminal",
        lambda *args, **_kwargs: recorded.append(args),
    )
    queued = Mock()
    monkeypatch.setattr("src.workers.tasks.ingestion.ingest_data.apply_async", queued)
    monkeypatch.setattr("src.core.saving.unstaged_ocr.time", lambda: 5_000)
    monkeypatch.setattr("src.core.saving.ingest_staging.time", lambda: 5_000)
    result = fail_expired_remote_ocr(
        "brain", "ocr", dict(store[("task:ocr", "brain")]),
    )
    queued.assert_called_once()
    assert result["status"] == "started"
    assert result["ocr_partial"] is True
    assert recorded == []


def test_unstaged_callback_returns_503_when_page_hash_unreadable(monkeypatch):
    app, ingest = _ingest_app()
    _silence_remote_ocr_runs(monkeypatch)
    cache, _store, _hashes = _writable_cache({
        ("task:ocr", "brain"): {"status": "queued", "task_id": "ocr"},
    })

    def boom(*_args, **_kwargs):
        raise RuntimeError("redis down")

    cache.hgetall = boom
    monkeypatch.setattr(ingest, "cache_adapter", cache)
    monkeypatch.setattr("src.workers.tasks.ingestion.cache_adapter", cache)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/ingest/",
        json={
            "data": {"data_type": "text", "text_data": "page"},
            "page_index": 0,
            "total_pages": 1,
        },
        headers={"Task-Identifier": "ocr"},
    )
    assert response.status_code == 503
    assert response.headers.get("retry-after") == "5"

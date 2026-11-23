from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import contextmanager
from unittest.mock import Mock, patch

import psycopg2
import psycopg2.pool
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from psycopg2.extensions import TRANSACTION_STATUS_IDLE

from src.lib.postgresql import _provisioning as provisioning
from src.lib.postgresql._provisioning import (
    PoolTimeout,
    _PoolRegistry,
    borrow,
    reset_pools_after_fork,
)
from src.services.api.dependencies import get_brain_id
from src.services.api.errors import install_error_handlers
from src.services.api.middlewares import ingest_publication as middleware
from src.services.api.routes.spreadsheets import router as spreadsheet_router
from src.services.explore.store import AtlasStore


class FakeConnection:
    def __init__(self):
        self.closed = 0
        self.status = TRANSACTION_STATUS_IDLE
        self.rollback_calls = 0

    def rollback(self):
        self.rollback_calls += 1
        if self.closed:
            raise psycopg2.InterfaceError("connection already closed")

    def close(self):
        self.closed = 1


class FakePool:
    def __init__(self, maxconn=1, dbname="brain_test"):
        self.maxconn = maxconn
        self.dbname = dbname
        self._kwargs = {"dbname": dbname}
        self.closed = False
        self.closeall_calls = 0
        self._idle = []
        self._used = []
        self._lock = threading.Lock()
        self._wait_limiter = threading.BoundedSemaphore(maxconn)

    def getconn(self):
        with self._lock:
            if self.closed:
                raise psycopg2.pool.PoolError("connection pool is closed")
            if self._idle:
                conn = self._idle.pop()
                self._used.append(conn)
                return conn
            if len(self._used) >= self.maxconn:
                raise psycopg2.pool.PoolError("connection pool exhausted")
            conn = FakeConnection()
            self._used.append(conn)
            return conn

    def putconn(self, conn, close=False):
        with self._lock:
            if self.closed:
                raise psycopg2.pool.PoolError("connection pool is closed")
            if conn in self._used:
                self._used.remove(conn)
            if close:
                conn.close()
                return
            self._idle.append(conn)

    def closeall(self):
        with self._lock:
            self.closeall_calls += 1
            self.closed = True
            for conn in self._idle + self._used:
                conn.closed = 1


@pytest.fixture
def registry(monkeypatch):
    instance = _PoolRegistry()
    created = []

    def create(dbname, minconn=None, maxconn=None):
        pool = FakePool(maxconn=maxconn or 1, dbname=dbname)
        created.append(pool)
        return pool

    instance._create_pool = create
    monkeypatch.setattr(provisioning, "_registry", instance)
    return instance, created


def test_env_pool_settings_reject_invalid_values(monkeypatch):
    monkeypatch.setenv("POSTGRES_POOL_MAX_OPEN", "nope")
    monkeypatch.setenv("POSTGRES_POOL_MAX_CONN", "")
    monkeypatch.setenv("POSTGRES_POOL_TIMEOUT", "abc")
    monkeypatch.setenv("POSTGRES_POOL_MIN_CONN", "-2")
    assert provisioning._max_open_pools() == 16
    assert provisioning._max_conn() >= 16
    assert provisioning.pool_timeout_seconds() == 5.0
    assert provisioning._min_conn() == 0


def test_pool_timeout_prefers_main_env_and_aliases_getconn(monkeypatch):
    monkeypatch.delenv("POSTGRES_POOL_TIMEOUT", raising=False)
    monkeypatch.setenv("POSTGRES_POOL_GETCONN_TIMEOUT", "3.5")
    assert provisioning.pool_timeout_seconds() == 3.5
    monkeypatch.setenv("POSTGRES_POOL_TIMEOUT", "8")
    monkeypatch.setenv("POSTGRES_POOL_GETCONN_TIMEOUT", "3.5")
    assert provisioning.pool_timeout_seconds() == 8.0


def test_fork_reset_drops_inherited_pools_without_closeall(registry):
    instance, created = registry
    pool = instance.get("brain_a")
    reset_pools_after_fork()
    assert pool.closeall_calls == 0
    assert "brain_a" not in instance._pools
    replacement = instance.get("brain_a")
    assert replacement is not pool
    assert replacement in created


def test_pid_mismatch_recreates_pool(registry):
    instance, _ = registry
    original = instance.get("brain_a")
    instance._pid = os.getpid() + 1
    replacement = instance.get("brain_a")
    assert original.closeall_calls == 0
    assert replacement is not original
    assert not replacement.closed


def test_closed_pool_is_recreated_on_get(registry):
    instance, created = registry
    original = instance.get("brain_a")
    original.closeall()
    replacement = instance.get("brain_a")
    assert replacement is not original
    assert not replacement.closed
    assert len(created) == 2


def test_eviction_skips_in_use_pools(monkeypatch, registry):
    monkeypatch.setenv("POSTGRES_POOL_MAX_OPEN", "2")
    instance, _ = registry
    held = instance.acquire("brain_a")
    instance.get("brain_b")
    instance.get("brain_c")
    assert "brain_a" in instance._pools
    assert "brain_b" not in instance._pools
    assert "brain_c" in instance._pools
    instance.release("brain_a", held, close=False)


def test_borrow_releases_pin_after_interface_error(registry):
    instance, _ = registry
    pool = instance.get("brain_a")
    with pytest.raises(psycopg2.InterfaceError):
        with borrow(pool):
            raise psycopg2.InterfaceError("connection already closed")
    assert instance._pins.get("brain_a", 0) == 0
    with borrow(pool) as conn:
        assert conn.closed == 0


def test_putconn_on_closed_pool_does_not_raise(registry):
    instance, _ = registry
    pool = instance.get("brain_a")
    with pytest.raises(psycopg2.InterfaceError):
        with borrow(pool) as conn:
            pool.closeall()
            conn.closed = 1
            conn.rollback()
    assert instance._pins.get("brain_a", 0) == 0


def test_exhausted_pool_waits_then_succeeds(monkeypatch, registry):
    monkeypatch.setenv("POSTGRES_POOL_TIMEOUT", "1")
    instance, _ = registry
    pool = instance.get("brain_a")
    started = threading.Event()
    release_hold = threading.Event()
    second_got = []

    def holder():
        with borrow(pool):
            started.set()
            release_hold.wait(timeout=2)

    def waiter():
        with borrow(pool) as conn:
            second_got.append(conn)

    first = threading.Thread(target=holder)
    first.start()
    assert started.wait(timeout=2)
    second = threading.Thread(target=waiter)
    second.start()
    time.sleep(0.05)
    release_hold.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert len(second_got) == 1


def test_exhausted_pool_times_out_with_pool_timeout(monkeypatch, registry):
    monkeypatch.setenv("POSTGRES_POOL_TIMEOUT", "0.05")
    instance, _ = registry
    pool = instance.get("brain_a")
    with borrow(pool):
        with pytest.raises(PoolTimeout, match="timeout"):
            with borrow(pool):
                pass


def test_atlas_build_retries_closed_pool_then_succeeds():
    store = AtlasStore.__new__(AtlasStore)
    calls = {"n": 0}

    def once(_brain_id, _token, _started):
        calls["n"] += 1
        if calls["n"] == 1:
            raise psycopg2.InterfaceError("connection already closed")

    store._build_once = once
    store._mark_build_failed = Mock()
    store.build("brain", "token")
    assert calls["n"] == 2
    store._mark_build_failed.assert_not_called()


def test_atlas_build_marks_failed_after_retry_exhausted():
    store = AtlasStore.__new__(AtlasStore)
    store._build_once = Mock(
        side_effect=psycopg2.pool.PoolError("connection pool is closed")
    )
    store._mark_build_failed = Mock()
    with pytest.raises(psycopg2.pool.PoolError):
        store.build("brain", "token")
    assert store._build_once.call_count == 2
    store._mark_build_failed.assert_called_once_with("brain", "token")


def test_atlas_transaction_rollback_survives_closed_connection():
    store = AtlasStore.__new__(AtlasStore)
    conn = Mock()
    conn.cursor.side_effect = RuntimeError("boom")
    conn.rollback.side_effect = psycopg2.InterfaceError("already closed")

    @contextmanager
    def connection(_brain_id):
        yield conn

    store.graph = Mock()
    store.graph._connection = connection
    with pytest.raises(RuntimeError, match="boom"):
        with store.transaction("brain"):
            pass
    conn.rollback.assert_called_once()


def test_pool_timeout_handler_returns_503():
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/pool-fail")
    async def pool_fail():
        raise PoolTimeout("connection pool timeout")

    response = TestClient(app, raise_server_exceptions=False).get("/pool-fail")
    assert response.status_code == 503
    body = response.json()
    assert body["detail"] == "Database is busy. Retry shortly."
    assert body["error"]["code"] == "SERVICE_UNAVAILABLE"
    assert response.headers.get("retry-after") == "1"
    assert "Traceback" not in response.text
    assert "PoolError" not in response.text
    assert "PoolTimeout" not in response.text


def test_ingest_publication_middleware_returns_503_on_pool_timeout(monkeypatch):
    events = []

    @contextmanager
    def gate(_brain):
        raise PoolTimeout("connection pool timeout")
        yield

    async def app(scope, receive, send):
        events.append(scope["path"])

    monkeypatch.setenv("STAGED_INGESTION_ENABLED", "1")
    monkeypatch.setattr(middleware, "brain_access", gate)
    guarded = middleware.IngestPublicationMiddleware(app)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/retrieve",
        "state": {"brain_id": "brain"},
    }
    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(guarded(scope, None, send))
    assert events == []
    start = next(item for item in sent if item["type"] == "http.response.start")
    assert start["status"] == 503
    headers = {key.decode(): value.decode() for key, value in start["headers"]}
    assert headers["retry-after"] == "1"


def test_spreadsheet_query_pool_timeout_returns_503(monkeypatch):
    monkeypatch.setenv("SPREADSHEET_EXPERIMENT_BRAINS", "demo")
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(spreadsheet_router)
    app.dependency_overrides[get_brain_id] = lambda: "demo"
    with patch(
        "src.services.api.routes.spreadsheets.connection",
        side_effect=PoolTimeout("connection pool timeout"),
    ):
        response = TestClient(app, raise_server_exceptions=False).post(
            "/retrieve/spreadsheets/query",
            json={"workbook": "00000000-0000-0000-0000-000000000001"},
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "Database is busy. Retry shortly."
    assert response.headers.get("retry-after") == "1"
    assert response.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


def test_worker_fork_hook_resets_registry(registry):
    instance, _ = registry
    instance.get("brain_a")
    from src.workers.app import _reset_postgres_pools_after_fork

    _reset_postgres_pools_after_fork()
    assert instance._pools == {}


def test_create_pool_semaphore_matches_maxconn(monkeypatch):
    monkeypatch.setenv("POSTGRES_POOL_MAX_CONN", "7")
    monkeypatch.setenv("POSTGRES_POOL_MIN_CONN", "2")
    captured = {}

    class CapturePool:
        def __init__(self, minconn, maxconn, **kwargs):
            captured["minconn"] = minconn
            captured["maxconn"] = maxconn
            captured["kwargs"] = kwargs
            self.closed = False

        def closeall(self):
            self.closed = True

    monkeypatch.setattr(psycopg2.pool, "ThreadedConnectionPool", CapturePool)
    monkeypatch.setattr(
        provisioning.config.postgresql, "validate_credentials", lambda: None
    )
    instance = _PoolRegistry()
    pool = instance._create_pool("brain_x")
    assert captured["minconn"] == 2
    assert captured["maxconn"] == 7
    assert pool.maxconn == 7
    assert pool._wait_limiter._initial_value == 7
    named = instance.get("brain_y", maxconn=4, minconn=1)
    assert named.maxconn == 4
    assert named._wait_limiter._initial_value == 4
    assert captured["maxconn"] == 4

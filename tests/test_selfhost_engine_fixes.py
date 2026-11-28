from __future__ import annotations

import os
from unittest.mock import patch

import psycopg2
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.constants.data import Brain
from src.services.api.errors import DependencyOutageMiddleware, install_error_handlers
from src.services.api.middlewares.auth import BrainPATMiddleware
from src.services.api.routes.system import system_router
from src.services.mcp.urls import configured_url, mcp_resource_url, public_base_url


def test_configured_url_ignores_template_placeholders():
    assert configured_url("https://your-mcp-oauth-issuer-url.com") == ""
    assert configured_url("https://your-mcp-resource-server-url.com/mcp") == ""
    assert configured_url("https://kb.example") == "https://kb.example"


def test_mcp_resource_url_uses_configured_public_url(monkeypatch):
    monkeypatch.setenv("BRAINAPI_PUBLIC_URL", "https://kb.example")
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    assert public_base_url() == "https://kb.example"
    assert mcp_resource_url("", "") == "https://kb.example/mcp"
    assert mcp_resource_url("", "https://your-mcp-resource-server-url.com") == (
        "https://kb.example/mcp"
    )


def test_brains_list_requires_system_pat():
    app = FastAPI()
    install_error_handlers(app)
    app.add_middleware(BrainPATMiddleware)
    app.include_router(system_router)
    with patch.dict(os.environ, {"BRAINPAT_TOKEN": "system-token"}):
        response = TestClient(app).get("/system/brains-list")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTH_INVALID"


def test_brains_list_redacts_pats_unless_requested():
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(system_router)
    brains = [Brain(id="1", name_key="demo", pat="secret-pat-value")]
    with patch(
        "src.services.api.controllers.system.data_adapter"
    ) as adapter:
        adapter.get_brains_list.return_value = brains
        client = TestClient(app)
        redacted = client.get("/system/brains-list")
        full = client.get("/system/brains-list", params={"include_pats": True})
    assert redacted.status_code == 200
    assert redacted.json()[0]["name_key"] == "demo"
    assert redacted.json()[0]["pat"] == ""
    assert full.status_code == 200
    assert full.json()[0]["pat"] == "secret-pat-value"


def test_create_brain_persists_dashboard_pat_scopes():
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(system_router)
    created = Brain(
        id="1",
        name_key="acme",
        pat="new-pat",
        pat_scopes=["verification"],
    )
    with patch("src.services.api.controllers.system.data_adapter") as adapter:
        adapter.create_brain.return_value = created
        response = TestClient(app).post(
            "/system/brains",
            json={"brain_id": "acme", "pat_scopes": ["brainapi", "verification"]},
        )
    assert response.status_code == 200
    adapter.create_brain.assert_called_once_with("acme", ["verification"])
    assert response.json()["pat_scopes"] == ["verification"]


def test_create_brain_rejects_unknown_pat_scope():
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(system_router)
    response = TestClient(app).post(
        "/system/brains",
        json={"brain_id": "acme", "pat_scopes": ["admin"]},
    )
    assert response.status_code == 422


def test_exception_group_database_outage_returns_503():
    app = FastAPI()
    install_error_handlers(app)
    app.add_middleware(DependencyOutageMiddleware)

    @app.get("/boom")
    async def boom():
        raise ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [psycopg2.OperationalError("server closed the connection unexpectedly")],
        )

    response = TestClient(app, raise_server_exceptions=False).get("/boom")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "SERVICE_UNAVAILABLE"

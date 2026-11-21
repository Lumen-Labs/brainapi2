import os
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from pydantic import AnyUrl
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "EMBEDDINGS_LOCAL_MODEL": "local-model",
    "EMBEDDINGS_SMALL_MODEL": "small-model",
    "EMBEDDING_NODES_DIMENSION": "3",
    "EMBEDDING_TRIPLETS_DIMENSION": "3",
    "EMBEDDING_OBSERVATIONS_DIMENSION": "3",
    "EMBEDDING_DATA_DIMENSION": "3",
    "EMBEDDING_RELATIONSHIPS_DIMENSION": "3",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
    "CELERY_WORKER_CONCURRENCY": "1",
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_LLM_SMALL_MODEL": "small",
    "OLLAMA_LLM_LARGE_MODEL": "large",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull

from src.services.mcp.main import _mcp_oauth_consent
from test_mcp_oauth_redis_storage import make_provider


ISSUER = "https://brainapi.example"
CALLBACK = "https://mcp-client.example/oauth/callback"


def _consent_client(provider):
    app = Starlette(
        routes=[Route("/mcp-oauth/consent", _mcp_oauth_consent, methods=["GET", "POST"])]
    )
    return TestClient(app), provider


def _pending_provider():
    from test_mcp_oauth_redis_storage import FakeRedis

    redis_client = FakeRedis()
    provider = make_provider(redis_client, credential_secret="secret")
    client = OAuthClientInformationFull(
        client_id="client-1",
        redirect_uris=[AnyUrl(CALLBACK)],
    )
    import asyncio

    asyncio.run(provider.register_client(client))
    params = AuthorizationParams(
        state="state",
        scopes=["brainapi"],
        code_challenge="challenge",
        redirect_uri=AnyUrl(CALLBACK),
        redirect_uri_provided_explicitly=True,
        resource="https://brainapi.example/mcp",
    )
    url = asyncio.run(provider.authorize(client, params))
    request_id = parse_qs(urlparse(url).query)["request_id"][0]
    return provider, request_id


def test_consent_get_requires_request_id():
    provider, request_id = _pending_provider()
    with (
        patch("src.services.mcp.main.oauth_provider", provider),
        patch("src.services.mcp.main._oauth_issuer", ISSUER),
    ):
        client, _ = _consent_client(provider)
        missing = client.get("/mcp-oauth/consent")
        found = client.get("/mcp-oauth/consent", params={"request_id": request_id})
    assert missing.status_code == 400
    assert found.status_code == 200
    assert 'name="request_id"' in found.text
    assert "code_challenge" not in found.text
    assert found.headers["cache-control"] == "no-store"
    policy = found.headers["content-security-policy"]
    assert "form-action 'self' https://mcp-client.example" in policy
    assert "https://chatgpt.com" not in policy
    assert "form-action 'self'" in missing.headers["content-security-policy"]
    assert "mcp-client.example" not in missing.headers["content-security-policy"]


def test_consent_csp_keeps_hosted_origin_when_restricted():
    provider, request_id = _pending_provider()
    with (
        patch("src.services.mcp.main.oauth_provider", provider),
        patch("src.services.mcp.main._oauth_issuer", ISSUER),
        patch("src.services.mcp.main._chatgpt_only", True),
    ):
        client, _ = _consent_client(provider)
        found = client.get("/mcp-oauth/consent", params={"request_id": request_id})
    policy = found.headers["content-security-policy"]
    assert "form-action 'self'" in policy
    assert "https://chatgpt.com" in policy
    assert "https://mcp-client.example" in policy


def test_consent_post_rejects_wrong_origin():
    provider, request_id = _pending_provider()
    with (
        patch("src.services.mcp.main.oauth_provider", provider),
        patch("src.services.mcp.main._oauth_issuer", ISSUER),
    ):
        client, _ = _consent_client(provider)
        response = client.post(
            "/mcp-oauth/consent",
            data={"request_id": request_id, "brainpat": "brain-pat"},
            headers={"origin": "https://evil.example"},
        )
    assert response.status_code == 403


def test_consent_post_requires_brain_scoped_pat():
    provider, request_id = _pending_provider()
    with (
        patch("src.services.mcp.main.oauth_provider", provider),
        patch("src.services.mcp.main._oauth_issuer", ISSUER),
        patch("src.services.mcp.main.guard_brainpat", return_value=True),
    ):
        client, _ = _consent_client(provider)
        response = client.post(
            "/mcp-oauth/consent",
            data={"request_id": request_id, "brainpat": "system-pat"},
            headers={"origin": ISSUER},
        )
    assert response.status_code == 401


def test_consent_post_issues_code_for_brain_pat():
    provider, request_id = _pending_provider()
    with (
        patch("src.services.mcp.main.oauth_provider", provider),
        patch("src.services.mcp.main._oauth_issuer", ISSUER),
        patch("src.services.mcp.main.guard_brainpat", return_value="demo"),
        patch(
            "src.services.mcp.main.resolve_token_identity",
            return_value=("brainpat:demo", ["brainapi"]),
        ),
    ):
        client, _ = _consent_client(provider)
        response = client.post(
            "/mcp-oauth/consent",
            data={"request_id": request_id, "brainpat": "brain-pat"},
            headers={"origin": ISSUER},
            follow_redirects=False,
        )
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith(CALLBACK)
    assert "code=" in location
    assert provider.consent_request(request_id) is None

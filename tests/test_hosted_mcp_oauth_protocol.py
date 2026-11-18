import os
from pathlib import Path
import subprocess
import sys


def test_hosted_mcp_oauth_protocol():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def verify_protocol():
    import base64
    import hashlib
    import os
    import secrets
    import sys
    from types import ModuleType, SimpleNamespace
    from urllib.parse import parse_qs, urlsplit

    from starlette.testclient import TestClient

    from test_mcp_oauth_redis_storage import FakeRedis

    redis = FakeRedis()
    issuer = "https://brainapi.example"
    resource = issuer + "/mcp"
    pat = "local-test-brain-pat"
    system_pat = "local-test-system-pat"
    os.environ.update(MCP_HOSTED_OAUTH_ISSUER_URL=issuer, MCP_OAUTH_ISSUER_URL="", MCP_OAUTH_CHATGPT_ONLY="", BRAINPAT_TOKEN=system_pat)

    def stub(name, **values):
        module = ModuleType(name)
        vars(module).update(values)
        sys.modules[name] = module

    stub("src.core.instances", data_adapter=None, embeddings_adapter=None, graph_adapter=None, vector_store_adapter=None)
    stub("src.services.mcp.utils", guard_brainpat=lambda token: "test-brain" if token == pat else token == system_pat)
    stub("src.lib.redis.client", _redis_client=SimpleNamespace(client=redis))
    stub("src.utils.vector_search", VectorSearchFacade=lambda _: None)

    from src.services.mcp.main import mcp, oauth_provider

    with TestClient(mcp.streamable_http_app(), base_url=issuer) as client:
        assert client.get("/.well-known/oauth-protected-resource/mcp").json()["resource"] == resource
        metadata = client.get("/.well-known/oauth-authorization-server").json()
        assert "S256" in metadata["code_challenge_methods_supported"]
        callback = "https://chatgpt.com/connector/oauth/test"
        registration = {"redirect_uris": [callback], "grant_types": ["authorization_code", "refresh_token"], "token_endpoint_auth_method": "client_secret_post"}
        assert client.post("/register", json={**registration, "redirect_uris": ["https://example.com/callback"]}).status_code == 400
        registered = client.post("/register", json=registration)
        assert registered.status_code == 201, registered.text
        creds = registered.json()
        assert oauth_provider._client_key(creds["client_id"]) not in redis.expirations
        verifier = secrets.token_urlsafe(40)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        response = client.get("/authorize", params={"client_id": creds["client_id"], "response_type": "code", "redirect_uri": callback, "code_challenge": challenge, "code_challenge_method": "S256", "resource": resource, "state": "test-state"}, follow_redirects=False)
        consent = response.headers["location"]
        assert client.get(consent).status_code == 200
        request_id = parse_qs(urlsplit(consent).query)["request_id"][0]
        fields = {"request_id": request_id, "brainpat": system_pat}
        assert client.post("/mcp-oauth/consent", data=fields, headers={"Origin": issuer}, follow_redirects=False).status_code == 401
        fields["brainpat"] = pat
        assert client.post("/mcp-oauth/consent", data=fields, headers={"Origin": "https://example.com"}, follow_redirects=False).status_code == 403
        response = client.post("/mcp-oauth/consent", data=fields, headers={"Origin": issuer}, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"].startswith(callback + "?")
        query = parse_qs(urlsplit(response.headers["location"]).query)
        assert query["state"] == ["test-state"]
        assert client.post("/mcp-oauth/consent", data=fields, headers={"Origin": issuer}, follow_redirects=False).status_code == 400
        exchange = {"client_id": creds["client_id"], "client_secret": creds["client_secret"], "grant_type": "authorization_code", "redirect_uri": callback, "code": query["code"][0], "code_verifier": verifier, "resource": resource}
        response = client.post("/token", data=exchange)
        assert response.status_code == 200, response.text
        token = response.json()["access_token"]
        assert token != pat and all(pat not in v for v in redis.values.values())
        assert client.post("/token", data=exchange).status_code == 400
        rpc = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "oauth-smoke", "version": "1"}}}
        headers = {"Accept": "application/json, text/event-stream"}
        response = client.post("/mcp", json=rpc, headers=headers)
        assert response.status_code == 401 and "resource_metadata" in response.headers["www-authenticate"]
        for bearer in (pat, token):
            assert client.post("/mcp", json=rpc, headers={**headers, "Authorization": "Bearer " + bearer}).status_code == 200
    print("OAuth discovery, ChatGPT registration, consent, PKCE, replay rejection, encrypted storage and both bearer modes passed")

if __name__ == "__main__":
    verify_protocol()

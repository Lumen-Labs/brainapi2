import asyncio
import base64
import hmac
import os
import unittest
from urllib.parse import parse_qs, urlparse

from cryptography.fernet import Fernet
from pydantic import AnyHttpUrl, AnyUrl
from starlette.testclient import TestClient

os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")

from mcp.server import FastMCP
from mcp.server.auth.provider import AuthorizationParams, TokenError
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.shared.auth import OAuthClientInformationFull

from src.services.mcp.oauth_provider import BrainapiMcpOAuthProvider

CALLBACK = "https://mcp-client.example/oauth/callback"
ISSUER = "https://brainapi.example"
RESOURCE = "https://brainapi.example/mcp"


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.sets = {}
        self.expirations = {}

    def set(self, key, value, ex=None):
        self.values[key] = value
        if ex:
            self.expirations[key] = ex
        return True

    def get(self, key):
        return self.values.get(key)

    def delete(self, *keys):
        removed = 0
        for key in keys:
            if key in self.values:
                removed += 1
                self.values.pop(key, None)
            if key in self.sets:
                removed += 1
                self.sets.pop(key, None)
            self.expirations.pop(key, None)
        return removed

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)
        return 1

    def smembers(self, key):
        return self.sets.get(key, set())

    def srem(self, key, value):
        if key not in self.sets:
            return 0
        before = len(self.sets[key])
        self.sets[key].discard(value)
        return before - len(self.sets[key])

    def expire(self, key, seconds):
        self.expirations[key] = seconds
        return True

    def getdel(self, key):
        self.expirations.pop(key, None)
        return self.values.pop(key, None)

    def scan(self, cursor=0, match=None, count=100):
        import fnmatch

        keys = list(self.values)
        if match:
            keys = [key for key in keys if fnmatch.fnmatch(key, match)]
        return 0, keys


def make_provider(
    redis_client,
    pat_verifier=lambda _pat: False,
    valid_scopes=None,
    default_scopes=None,
    pat_scope_resolver=None,
    chatgpt_only=False,
    credential_secret=None,
):
    return BrainapiMcpOAuthProvider(
        issuer_url="https://brainapi.example",
        resource_server_url="https://brainapi.example/mcp",
        valid_scopes=valid_scopes or ["brainapi"],
        default_scopes=default_scopes,
        access_token_ttl_seconds=3600,
        refresh_token_ttl_seconds=7200,
        auth_code_ttl_seconds=600,
        redis_client=redis_client,
        pat_verifier=pat_verifier,
        pat_scope_resolver=pat_scope_resolver,
        chatgpt_only=chatgpt_only,
        credential_secret=credential_secret,
    )


def _client_info(client_id="client-1"):
    return OAuthClientInformationFull(
        client_id=client_id,
        redirect_uris=[AnyUrl(CALLBACK)],
    )


def _mint_oauth_token(redis_client, credential_secret=None, brainpat="brain-pat"):
    provider = make_provider(redis_client, credential_secret=credential_secret)
    client = _client_info()
    asyncio.run(provider.register_client(client))
    code = provider.issue_auth_code(
        client_id=client.client_id,
        redirect_uri=AnyUrl(CALLBACK),
        code_challenge="challenge",
        scopes=["brainapi"],
        resource=RESOURCE,
        state=None,
        brainpat=brainpat,
    )
    auth_code = asyncio.run(provider.load_authorization_code(client, code))
    token = asyncio.run(provider.exchange_authorization_code(client, auth_code))
    return provider, client, token


def _mcp_http(provider):
    mcp = FastMCP(
        "brainapi-mcp-test",
        stateless_http=True,
        auth_server_provider=provider,
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(ISSUER),
            resource_server_url=AnyHttpUrl(RESOURCE),
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=["brainapi"],
                default_scopes=["brainapi"],
            ),
        ),
    )
    return TestClient(mcp.streamable_http_app())


def _post_mcp(http, access_token):
    return http.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
    )


class BrainapiMcpOAuthProviderRedisTests(unittest.TestCase):
    def test_authorization_code_exchange_survives_provider_recreation(self):
        redis_client = FakeRedis()
        issuer = make_provider(redis_client)
        client = OAuthClientInformationFull(
            client_id="client-1",
            redirect_uris=[AnyUrl("https://claude.ai/api/mcp/auth_callback")],
        )
        asyncio.run(issuer.register_client(client))

        code = issuer.issue_auth_code(
            client_id="client-1",
            redirect_uri=AnyUrl("https://claude.ai/api/mcp/auth_callback"),
            code_challenge="challenge",
            scopes=["brainapi"],
            resource="https://brainapi.example/mcp",
            state="state",
            brainpat="brain-pat",
        )

        recreated = make_provider(redis_client)
        loaded_client = asyncio.run(recreated.get_client("client-1"))
        loaded_code = asyncio.run(recreated.load_authorization_code(loaded_client, code))
        token = asyncio.run(recreated.exchange_authorization_code(loaded_client, loaded_code))

        self.assertEqual(recreated.get_pat_for_access_token(token.access_token), "brain-pat")
        self.assertIsNone(asyncio.run(recreated.load_authorization_code(loaded_client, code)))
        self.assertIsNotNone(asyncio.run(recreated.load_access_token(token.access_token)))

    def test_authorization_code_preserves_implicit_redirect_uri(self):
        redis_client = FakeRedis()
        provider = make_provider(redis_client)

        code = provider.issue_auth_code(
            client_id="client-1",
            redirect_uri=AnyUrl("https://claude.ai/api/mcp/auth_callback"),
            redirect_uri_provided_explicitly=False,
            code_challenge="challenge",
            scopes=["brainapi"],
            resource="https://brainapi.example/mcp",
            state="state",
            brainpat="brain-pat",
        )
        auth_code = asyncio.run(provider.load_authorization_code(None, code))

        self.assertFalse(auth_code.redirect_uri_provided_explicitly)

    def test_authorize_carries_redirect_uri_explicitness_to_consent(self):
        provider = make_provider(FakeRedis())
        client = OAuthClientInformationFull(
            client_id="client-1",
            redirect_uris=[AnyUrl("https://claude.ai/api/mcp/auth_callback")],
        )
        params = AuthorizationParams(
            state="state",
            scopes=["brainapi"],
            code_challenge="challenge",
            redirect_uri=AnyUrl("https://claude.ai/api/mcp/auth_callback"),
            redirect_uri_provided_explicitly=False,
            resource="https://brainapi.example/mcp",
        )

        url = asyncio.run(provider.authorize(client, params))
        query = parse_qs(urlparse(url).query)
        pending = provider.consent_request(query["request_id"][0])

        self.assertEqual(list(query), ["request_id"])
        self.assertEqual(pending["redirect_uri_provided_explicitly"], "0")

    def test_valid_pat_can_be_used_as_bearer_token(self):
        provider = make_provider(FakeRedis(), pat_verifier=lambda pat: pat == "brain-pat")

        access_token = asyncio.run(provider.load_access_token("brain-pat"))

        self.assertIsNotNone(access_token)
        self.assertEqual(access_token.token, "brain-pat")
        self.assertEqual(access_token.client_id, "brainpat")
        self.assertEqual(access_token.scopes, ["brainapi"])

    def test_raw_pat_does_not_receive_verification_scope_by_default(self):
        provider = make_provider(
            FakeRedis(),
            pat_verifier=lambda pat: pat == "brain-pat",
            valid_scopes=["brainapi", "verification"],
            default_scopes=["brainapi"],
        )
        access_token = asyncio.run(provider.load_access_token("brain-pat"))
        self.assertIsNotNone(access_token)
        self.assertEqual(access_token.scopes, ["brainapi"])
        self.assertNotIn("verification", access_token.scopes)

    def test_authorize_without_requested_scopes_uses_defaults(self):
        provider = make_provider(
            FakeRedis(),
            valid_scopes=["brainapi", "verification"],
            default_scopes=["brainapi"],
        )
        client = OAuthClientInformationFull(
            client_id="client-1",
            redirect_uris=[AnyUrl("https://claude.ai/api/mcp/auth_callback")],
        )
        params = AuthorizationParams(
            state="state",
            scopes=None,
            code_challenge="challenge",
            redirect_uri=AnyUrl("https://claude.ai/api/mcp/auth_callback"),
            redirect_uri_provided_explicitly=True,
            resource="https://brainapi.example/mcp",
        )
        url = asyncio.run(provider.authorize(client, params))
        query = parse_qs(urlparse(url).query)
        pending = provider.consent_request(query["request_id"][0])
        self.assertEqual(pending["scope"], "brainapi")

    def test_invalid_pat_is_not_accepted_as_bearer_token(self):
        provider = make_provider(FakeRedis(), pat_verifier=lambda _pat: False)

        self.assertIsNone(asyncio.run(provider.load_access_token("bad-pat")))

    def test_refresh_exchange_survives_provider_recreation_and_revokes_old_access_token(self):
        redis_client = FakeRedis()
        provider = make_provider(redis_client)
        client = OAuthClientInformationFull(
            client_id="client-1",
            redirect_uris=[AnyUrl("https://claude.ai/api/mcp/auth_callback")],
        )
        asyncio.run(provider.register_client(client))
        code = provider.issue_auth_code(
            client_id="client-1",
            redirect_uri=AnyUrl("https://claude.ai/api/mcp/auth_callback"),
            code_challenge="challenge",
            scopes=["brainapi"],
            resource="https://brainapi.example/mcp",
            state=None,
            brainpat="brain-pat",
        )
        auth_code = asyncio.run(provider.load_authorization_code(client, code))
        first_token = asyncio.run(provider.exchange_authorization_code(client, auth_code))

        recreated = make_provider(redis_client)
        refresh_token = asyncio.run(
            recreated.load_refresh_token(client, first_token.refresh_token)
        )
        second_token = asyncio.run(
            recreated.exchange_refresh_token(client, refresh_token, ["brainapi"])
        )

        self.assertIsNone(asyncio.run(recreated.load_access_token(first_token.access_token)))
        self.assertEqual(
            recreated.get_pat_for_access_token(second_token.access_token),
            "brain-pat",
        )

    def test_chatgpt_only_rejects_non_connector_redirect(self):
        from mcp.server.auth.provider import AuthorizeError, RegistrationError

        provider = make_provider(FakeRedis(), chatgpt_only=True, credential_secret="secret")
        client = OAuthClientInformationFull(
            client_id="client-1",
            redirect_uris=[AnyUrl("https://mcp-client.example/oauth/callback")],
        )
        with self.assertRaises(RegistrationError):
            asyncio.run(provider.register_client(client))
        hosted = OAuthClientInformationFull(
            client_id="client-1",
            redirect_uris=[AnyUrl("https://chatgpt.com/connector_platform_oauth_redirect")],
        )
        asyncio.run(provider.register_client(hosted))
        params = AuthorizationParams(
            state="state",
            scopes=["brainapi"],
            code_challenge="challenge",
            redirect_uri=AnyUrl("https://mcp-client.example/oauth/callback"),
            redirect_uri_provided_explicitly=True,
            resource="https://brainapi.example/mcp",
        )
        with self.assertRaises(AuthorizeError):
            asyncio.run(provider.authorize(hosted, params))

    def test_chatgpt_only_requires_credential_secret(self):
        with self.assertRaises(ValueError):
            make_provider(FakeRedis(), chatgpt_only=True)

    def test_stored_pat_is_encrypted_when_secret_is_set(self):
        redis_client = FakeRedis()
        provider = make_provider(redis_client, credential_secret="secret")
        client = OAuthClientInformationFull(
            client_id="client-1",
            redirect_uris=[AnyUrl("https://mcp-client.example/oauth/callback")],
        )
        asyncio.run(provider.register_client(client))
        code = provider.issue_auth_code(
            client_id="client-1",
            redirect_uri=AnyUrl("https://mcp-client.example/oauth/callback"),
            code_challenge="challenge",
            scopes=["brainapi"],
            resource="https://brainapi.example/mcp",
            state=None,
            brainpat="brain-pat",
        )
        raw = redis_client.get(f"mcp:oauth:code_pat:{code}")
        self.assertNotEqual(raw, "brain-pat")
        auth_code = asyncio.run(provider.load_authorization_code(client, code))
        token = asyncio.run(provider.exchange_authorization_code(client, auth_code))
        self.assertEqual(provider.get_pat_for_access_token(token.access_token), "brain-pat")
        stored_access = redis_client.get(f"mcp:oauth:access_pat:{token.access_token}")
        self.assertNotEqual(stored_access, "brain-pat")

    def test_load_access_token_rejects_resource_mismatch(self):
        redis_client = FakeRedis()
        provider = make_provider(redis_client)
        client = OAuthClientInformationFull(
            client_id="client-1",
            redirect_uris=[AnyUrl("https://mcp-client.example/oauth/callback")],
        )
        asyncio.run(provider.register_client(client))
        code = provider.issue_auth_code(
            client_id="client-1",
            redirect_uri=AnyUrl("https://mcp-client.example/oauth/callback"),
            code_challenge="challenge",
            scopes=["brainapi"],
            resource="https://brainapi.example/mcp",
            state=None,
            brainpat="brain-pat",
        )
        auth_code = asyncio.run(provider.load_authorization_code(client, code))
        token = asyncio.run(provider.exchange_authorization_code(client, auth_code))
        provider._resource_server_url = "https://other.example/mcp"
        self.assertIsNone(asyncio.run(provider.load_access_token(token.access_token)))

    def test_consent_request_is_single_use(self):
        provider = make_provider(FakeRedis())
        client = OAuthClientInformationFull(
            client_id="client-1",
            redirect_uris=[AnyUrl("https://mcp-client.example/oauth/callback")],
        )
        params = AuthorizationParams(
            state="state",
            scopes=["brainapi"],
            code_challenge="challenge",
            redirect_uri=AnyUrl("https://mcp-client.example/oauth/callback"),
            redirect_uri_provided_explicitly=True,
            resource="https://brainapi.example/mcp",
        )
        url = asyncio.run(provider.authorize(client, params))
        request_id = parse_qs(urlparse(url).query)["request_id"][0]
        self.assertIsNotNone(provider.consent_request(request_id))
        consumed = provider.consent_request(request_id, consume=True)
        self.assertEqual(consumed["client_id"], "client-1")
        self.assertIsNone(provider.consent_request(request_id))
        self.assertIsNone(provider.consent_request("not-a-valid-id"))

    def test_fernet_key_is_hmac_sha256_of_secret_and_issuer(self):
        redis_client = FakeRedis()
        provider = make_provider(redis_client, credential_secret="secret")
        purpose = b"brainapi-mcp-oauth-pat:https://brainapi.example"
        key = hmac.digest(b"secret", purpose, "sha256")
        expected = Fernet(base64.urlsafe_b64encode(key))
        token = expected.encrypt(b"brain-pat")
        self.assertEqual(provider._credential_cipher.decrypt(token), b"brain-pat")

    def test_plaintext_pat_from_unencrypted_image_is_purged_and_mcp_returns_401(self):
        redis_client = FakeRedis()
        _, client, token = _mint_oauth_token(redis_client)
        access_pat_key = f"mcp:oauth:access_pat:{token.access_token}"
        refresh_pat_key = f"mcp:oauth:refresh_pat:{token.refresh_token}"
        self.assertEqual(redis_client.get(access_pat_key), "brain-pat")
        self.assertEqual(redis_client.get(refresh_pat_key), "brain-pat")
        self.assertFalse(redis_client.get(access_pat_key).startswith("gAAAAA"))

        upgraded = make_provider(redis_client, credential_secret="secret")
        self.assertIsNone(redis_client.get(access_pat_key))
        self.assertIsNone(redis_client.get(f"mcp:oauth:access:{token.access_token}"))
        self.assertIsNone(redis_client.get(refresh_pat_key))
        self.assertIsNone(redis_client.get(f"mcp:oauth:refresh:{token.refresh_token}"))
        self.assertIsNone(asyncio.run(upgraded.load_access_token(token.access_token)))
        self.assertIsNone(upgraded.get_pat_for_access_token(token.access_token))
        self.assertIsNone(
            asyncio.run(upgraded.load_refresh_token(client, token.refresh_token))
        )
        with self.assertRaises(TokenError) as raised:
            asyncio.run(
                upgraded.exchange_refresh_token(
                    client,
                    type("RT", (), {"token": token.refresh_token, "client_id": client.client_id})(),
                    ["brainapi"],
                )
            )
        self.assertEqual(raised.exception.error, "invalid_grant")

        with _mcp_http(upgraded) as http:
            response = _post_mcp(http, token.access_token)
        self.assertEqual(response.status_code, 401)

    def test_decrypt_failure_deletes_plaintext_pat_without_startup_purge(self):
        redis_client = FakeRedis()
        provider, _client, token = _mint_oauth_token(redis_client)
        access_pat_key = f"mcp:oauth:access_pat:{token.access_token}"
        self.assertEqual(redis_client.get(access_pat_key), "brain-pat")
        purpose = b"brainapi-mcp-oauth-pat:https://brainapi.example"
        key = hmac.digest(b"secret", purpose, "sha256")
        provider._credential_cipher = Fernet(base64.urlsafe_b64encode(key))

        self.assertIsNone(asyncio.run(provider.load_access_token(token.access_token)))
        self.assertIsNone(redis_client.get(access_pat_key))
        self.assertIsNone(redis_client.get(f"mcp:oauth:access:{token.access_token}"))
        self.assertIsNone(provider.get_pat_for_access_token(token.access_token))

        with _mcp_http(provider) as http:
            response = _post_mcp(http, token.access_token)
        self.assertEqual(response.status_code, 401)

    def test_wrong_key_ciphertext_is_deleted_on_use(self):
        redis_client = FakeRedis()
        _, _client, token = _mint_oauth_token(redis_client, credential_secret="secret-a")
        access_pat_key = f"mcp:oauth:access_pat:{token.access_token}"
        stored = redis_client.get(access_pat_key)
        self.assertTrue(stored.startswith("gAAAAA"))

        other = make_provider(redis_client, credential_secret="secret-b")
        self.assertEqual(redis_client.get(access_pat_key), stored)
        self.assertIsNone(asyncio.run(other.load_access_token(token.access_token)))
        self.assertIsNone(redis_client.get(access_pat_key))
        self.assertIsNone(redis_client.get(f"mcp:oauth:access:{token.access_token}"))

        with _mcp_http(other) as http:
            response = _post_mcp(http, token.access_token)
        self.assertEqual(response.status_code, 401)

    def test_encrypted_pats_survive_plaintext_cleanup(self):
        redis_client = FakeRedis()
        _, _client, token = _mint_oauth_token(redis_client, credential_secret="secret")
        access_pat_key = f"mcp:oauth:access_pat:{token.access_token}"
        stored = redis_client.get(access_pat_key)
        self.assertTrue(stored.startswith("gAAAAA"))

        recreated = make_provider(redis_client, credential_secret="secret")
        self.assertEqual(redis_client.get(access_pat_key), stored)
        self.assertIsNotNone(asyncio.run(recreated.load_access_token(token.access_token)))
        self.assertEqual(recreated.get_pat_for_access_token(token.access_token), "brain-pat")


if __name__ == "__main__":
    unittest.main()

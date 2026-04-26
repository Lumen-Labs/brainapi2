import json
import secrets
import time
from urllib.parse import urlencode

from pydantic import AnyUrl

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

class BrainapiMcpOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    def __init__(
        self,
        *,
        issuer_url: str,
        resource_server_url: str,
        valid_scopes: list[str],
        access_token_ttl_seconds: int,
        refresh_token_ttl_seconds: int | None,
        auth_code_ttl_seconds: int,
        redis_client=None,
    ):
        self._issuer_url = issuer_url.rstrip("/")
        self._resource_server_url = resource_server_url.rstrip("/")
        self._valid_scopes = valid_scopes
        self._access_token_ttl = access_token_ttl_seconds
        self._refresh_token_ttl = refresh_token_ttl_seconds
        self._auth_code_ttl = auth_code_ttl_seconds
        if redis_client is None:
            from src.lib.redis.client import _redis_client

            redis_client = _redis_client.client
        self._redis = redis_client

    def _key(self, *parts: str) -> str:
        return "mcp:oauth:" + ":".join(parts)

    def _get_text(self, key: str) -> str | None:
        value = self._redis.get(key)
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    def _set_text(self, key: str, value: str, ttl: int | None = None) -> None:
        self._redis.set(key, value, **({"ex": ttl} if ttl else {}))

    def _set_model(self, key: str, value, ttl: int | None = None) -> None:
        if hasattr(value, "model_dump_json"):
            payload = value.model_dump_json()
        else:
            payload = json.dumps(value)
        self._set_text(key, payload, ttl)

    def _get_model(self, key: str, model_type):
        payload = self._get_text(key)
        if payload is None:
            return None
        if hasattr(model_type, "model_validate_json"):
            return model_type.model_validate_json(payload)
        return model_type(**json.loads(payload))

    def _client_key(self, client_id: str) -> str:
        return self._key("client", client_id)

    def _auth_code_key(self, code: str) -> str:
        return self._key("auth_code", code)

    def _code_pat_key(self, code: str) -> str:
        return self._key("code_pat", code)

    def _access_key(self, token: str) -> str:
        return self._key("access", token)

    def _access_pat_key(self, token: str) -> str:
        return self._key("access_pat", token)

    def _refresh_key(self, token: str) -> str:
        return self._key("refresh", token)

    def _refresh_pat_key(self, token: str) -> str:
        return self._key("refresh_pat", token)

    def _client_access_key(self, client_id: str) -> str:
        return self._key("client_access", client_id)

    def _remember_client_access_token(
        self, client_id: str, access_token: str, ttl: int
    ) -> None:
        key = self._client_access_key(client_id)
        self._redis.sadd(key, access_token)
        self._redis.expire(key, ttl)

    def _client_access_tokens(self, client_id: str) -> list[str]:
        values = self._redis.smembers(self._client_access_key(client_id))
        return [
            value.decode("utf-8") if isinstance(value, bytes) else value
            for value in values
        ]

    def _delete_access_token(self, token: str, client_id: str | None = None) -> None:
        self._redis.delete(self._access_key(token), self._access_pat_key(token))
        if client_id:
            self._redis.srem(self._client_access_key(client_id), token)

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._get_model(self._client_key(client_id), OAuthClientInformationFull)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        cid = client_info.client_id
        if not cid:
            raise ValueError("client_id required")
        self._set_model(self._client_key(cid), client_info)

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        if params.resource and params.resource.rstrip("/") != self._resource_server_url:
            raise AuthorizeError(
                error="invalid_request",
                error_description="resource must match this MCP server URL",
            )
        scopes = params.scopes
        if scopes is None:
            scopes = list(self._valid_scopes)
        for s in scopes:
            if s not in self._valid_scopes:
                raise AuthorizeError(error="invalid_scope", error_description=f"unknown scope: {s}")

        consent_base = f"{self._issuer_url}/mcp-oauth/consent"
        q = {
            "client_id": client.client_id or "",
            "redirect_uri": str(params.redirect_uri),
            "code_challenge": params.code_challenge,
            "scope": " ".join(scopes),
        }
        if params.resource:
            q["resource"] = params.resource
        if params.state:
            q["state"] = params.state
        return f"{consent_base}?{urlencode(q)}"

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        return self._get_model(self._auth_code_key(authorization_code), AuthorizationCode)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        self._redis.delete(self._auth_code_key(authorization_code.code))
        brainpat = self._get_text(self._code_pat_key(authorization_code.code))
        self._redis.delete(self._code_pat_key(authorization_code.code))
        if not brainpat:
            raise TokenError(error="invalid_grant", error_description="missing credentials")

        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(48)
        now = int(time.time())
        exp_access = now + self._access_token_ttl
        exp_refresh = now + self._refresh_token_ttl if self._refresh_token_ttl else None

        at = AccessToken(
            token=access,
            client_id=client.client_id or "",
            scopes=authorization_code.scopes,
            expires_at=exp_access,
            resource=self._resource_server_url,
        )
        access_ttl = max(1, exp_access - now)
        self._set_model(self._access_key(access), at, access_ttl)
        self._set_text(self._access_pat_key(access), brainpat, access_ttl)
        self._remember_client_access_token(client.client_id or "", access, access_ttl)
        rt = RefreshToken(
            token=refresh,
            client_id=client.client_id or "",
            scopes=authorization_code.scopes,
            expires_at=exp_refresh,
        )
        refresh_ttl = max(1, exp_refresh - now) if exp_refresh else None
        self._set_model(self._refresh_key(refresh), rt, refresh_ttl)
        self._set_text(self._refresh_pat_key(refresh), brainpat, refresh_ttl)
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=self._access_token_ttl,
            refresh_token=refresh,
            scope=" ".join(authorization_code.scopes),
        )

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        rt = self._get_model(self._refresh_key(refresh_token), RefreshToken)
        if rt is None or rt.client_id != (client.client_id or ""):
            return None
        return rt

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        brainpat = self._get_text(self._refresh_pat_key(refresh_token.token))
        self._redis.delete(
            self._refresh_key(refresh_token.token),
            self._refresh_pat_key(refresh_token.token),
        )
        if not brainpat:
            raise TokenError(error="invalid_grant", error_description="refresh token not found")

        client_id = client.client_id or ""
        for token in self._client_access_tokens(client_id):
            self._delete_access_token(token, client_id)
        self._redis.delete(self._client_access_key(client_id))

        access = secrets.token_urlsafe(32)
        new_refresh = secrets.token_urlsafe(48)
        now = int(time.time())
        exp_access = now + self._access_token_ttl
        exp_refresh = now + self._refresh_token_ttl if self._refresh_token_ttl else None

        at = AccessToken(
            token=access,
            client_id=client_id,
            scopes=scopes,
            expires_at=exp_access,
            resource=self._resource_server_url,
        )
        access_ttl = max(1, exp_access - now)
        self._set_model(self._access_key(access), at, access_ttl)
        self._set_text(self._access_pat_key(access), brainpat, access_ttl)
        self._remember_client_access_token(client_id, access, access_ttl)
        rt = RefreshToken(
            token=new_refresh,
            client_id=client_id,
            scopes=scopes,
            expires_at=exp_refresh,
        )
        refresh_ttl = max(1, exp_refresh - now) if exp_refresh else None
        self._set_model(self._refresh_key(new_refresh), rt, refresh_ttl)
        self._set_text(self._refresh_pat_key(new_refresh), brainpat, refresh_ttl)
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=self._access_token_ttl,
            refresh_token=new_refresh,
            scope=" ".join(scopes),
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        at = self._get_model(self._access_key(token), AccessToken)
        if at is None:
            return None
        if at.expires_at and at.expires_at < int(time.time()):
            self._delete_access_token(token, at.client_id)
            return None
        return at

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        if isinstance(token, AccessToken):
            self._delete_access_token(token.token, token.client_id)
        else:
            self._redis.delete(
                self._refresh_key(token.token),
                self._refresh_pat_key(token.token),
            )

    def get_pat_for_access_token(self, token: str) -> str | None:
        return self._get_text(self._access_pat_key(token))

    def issue_auth_code(
        self,
        *,
        client_id: str,
        redirect_uri: AnyUrl,
        code_challenge: str,
        scopes: list[str],
        resource: str | None,
        state: str | None,
        brainpat: str,
    ) -> str:
        code = secrets.token_urlsafe(48)
        ac = AuthorizationCode(
            code=code,
            scopes=scopes,
            expires_at=time.time() + self._auth_code_ttl,
            client_id=client_id,
            code_challenge=code_challenge,
            redirect_uri=redirect_uri,
            redirect_uri_provided_explicitly=True,
            resource=resource,
        )
        self._set_model(self._auth_code_key(code), ac, self._auth_code_ttl)
        self._set_text(self._code_pat_key(code), brainpat, self._auth_code_ttl)
        return code

    def redirect_after_consent(
        self,
        *,
        redirect_uri: str,
        code: str,
        state: str | None,
    ) -> str:
        params: dict[str, str] = {"code": code}
        if state:
            params["state"] = state
        return construct_redirect_uri(redirect_uri, **params)

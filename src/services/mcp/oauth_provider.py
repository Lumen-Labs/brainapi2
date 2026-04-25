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
    ):
        self._issuer_url = issuer_url.rstrip("/")
        self._resource_server_url = resource_server_url.rstrip("/")
        self._valid_scopes = valid_scopes
        self._access_token_ttl = access_token_ttl_seconds
        self._refresh_token_ttl = refresh_token_ttl_seconds
        self._auth_code_ttl = auth_code_ttl_seconds
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._auth_codes: dict[str, AuthorizationCode] = {}
        self._code_to_pat: dict[str, str] = {}
        self._refresh_tokens: dict[str, RefreshToken] = {}
        self._refresh_to_pat: dict[str, str] = {}
        self._access_tokens: dict[str, AccessToken] = {}
        self._pat_by_access: dict[str, str] = {}

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        cid = client_info.client_id
        if not cid:
            raise ValueError("client_id required")
        self._clients[cid] = client_info

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
        return self._auth_codes.get(authorization_code)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        self._auth_codes.pop(authorization_code.code, None)
        brainpat = self._code_to_pat.pop(authorization_code.code, None)
        if not brainpat:
            raise TokenError(error="invalid_grant", error_description="missing credentials")

        access = secrets.token_urlsafe(32)
        refresh = secrets.token_urlsafe(48)
        now = int(time.time())
        exp_access = now + self._access_token_ttl
        exp_refresh = now + self._refresh_token_ttl if self._refresh_token_ttl else None

        self._pat_by_access[access] = brainpat
        at = AccessToken(
            token=access,
            client_id=client.client_id or "",
            scopes=authorization_code.scopes,
            expires_at=exp_access,
            resource=self._resource_server_url,
        )
        self._access_tokens[access] = at
        rt = RefreshToken(
            token=refresh,
            client_id=client.client_id or "",
            scopes=authorization_code.scopes,
            expires_at=exp_refresh,
        )
        self._refresh_tokens[refresh] = rt
        self._refresh_to_pat[refresh] = brainpat
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=self._access_token_ttl,
            refresh_token=refresh,
            scope=" ".join(authorization_code.scopes),
        )

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        rt = self._refresh_tokens.get(refresh_token)
        if rt is None or rt.client_id != (client.client_id or ""):
            return None
        return rt

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        brainpat = self._refresh_to_pat.pop(refresh_token.token, None)
        self._refresh_tokens.pop(refresh_token.token, None)
        if not brainpat:
            raise TokenError(error="invalid_grant", error_description="refresh token not found")

        for t, v in list(self._access_tokens.items()):
            if v.client_id == (client.client_id or ""):
                self._access_tokens.pop(t, None)
                self._pat_by_access.pop(t, None)

        access = secrets.token_urlsafe(32)
        new_refresh = secrets.token_urlsafe(48)
        now = int(time.time())
        exp_access = now + self._access_token_ttl
        exp_refresh = now + self._refresh_token_ttl if self._refresh_token_ttl else None

        self._pat_by_access[access] = brainpat
        at = AccessToken(
            token=access,
            client_id=client.client_id or "",
            scopes=scopes,
            expires_at=exp_access,
            resource=self._resource_server_url,
        )
        self._access_tokens[access] = at
        rt = RefreshToken(
            token=new_refresh,
            client_id=client.client_id or "",
            scopes=scopes,
            expires_at=exp_refresh,
        )
        self._refresh_tokens[new_refresh] = rt
        self._refresh_to_pat[new_refresh] = brainpat
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=self._access_token_ttl,
            refresh_token=new_refresh,
            scope=" ".join(scopes),
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        at = self._access_tokens.get(token)
        if at is None:
            return None
        if at.expires_at and at.expires_at < int(time.time()):
            self._access_tokens.pop(token, None)
            self._pat_by_access.pop(token, None)
            return None
        return at

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        if isinstance(token, AccessToken):
            self._access_tokens.pop(token.token, None)
            self._pat_by_access.pop(token.token, None)
        else:
            self._refresh_tokens.pop(token.token, None)
            self._refresh_to_pat.pop(token.token, None)

    def get_pat_for_access_token(self, token: str) -> str | None:
        return self._pat_by_access.get(token)

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
        self._auth_codes[code] = ac
        self._code_to_pat[code] = brainpat
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

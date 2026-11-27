from __future__ import annotations

import os
from typing import Iterable, Sequence

DEFAULT_SCOPE = "brainapi"
VERIFICATION_SCOPE = "verification"
VERIFICATION_PERMISSION_DENIED = "VERIFICATION_PERMISSION_DENIED"


def default_token_scopes() -> list[str]:
    return [DEFAULT_SCOPE]


def env_system_pat_scopes() -> list[str]:
    raw = os.getenv("BRAINPAT_SCOPES", "").strip()
    if not raw:
        return default_token_scopes()
    return [part for part in raw.split() if part]


def effective_pat_scopes(brain) -> list[str]:
    scopes = default_token_scopes()
    extra = [
        str(part).strip()
        for part in (getattr(brain, "pat_scopes", None) or [])
        if str(part).strip()
    ]
    for scope in extra:
        if scope not in scopes:
            scopes.append(scope)
    return scopes


def principal_for_system() -> str:
    return "system"


def is_system_principal(principal: str | None) -> bool:
    return str(principal or "").strip() == principal_for_system()


def principal_for_user(user_id: str) -> str:
    return f"user:{user_id}"


def principal_for_brain(name_key: str) -> str:
    return f"brainpat:{name_key}"


def principal_for_oauth(client_id: str) -> str:
    return f"oauth:{client_id}"


def has_verification_scope(scopes: Iterable[str] | None) -> bool:
    return VERIFICATION_SCOPE in {str(scope).strip() for scope in (scopes or []) if scope}


def grantable_oauth_scopes(
    requested: Sequence[str] | None,
    pat_scopes: Sequence[str] | None,
    *,
    valid_scopes: Sequence[str] | None = None,
) -> list[str]:
    requested_list = [str(scope).strip() for scope in (requested or []) if str(scope).strip()]
    if not requested_list:
        requested_list = default_token_scopes()
    allowed = set(default_token_scopes())
    for scope in pat_scopes or []:
        text = str(scope).strip()
        if text:
            allowed.add(text)
    valid = {str(scope).strip() for scope in (valid_scopes or []) if str(scope).strip()}
    granted: list[str] = []
    for scope in requested_list:
        if valid and scope not in valid:
            continue
        if scope not in allowed:
            continue
        if scope not in granted:
            granted.append(scope)
    return granted or default_token_scopes()


def resolve_token_identity(pat: str | None) -> tuple[str | None, list[str]]:
    if not pat:
        return None, []
    from src.config import config
    from src.services.data.main import data_adapter
    from src.utils.tokens import pat_matches

    if pat_matches(pat, config.brainpat_token):
        return principal_for_system(), env_system_pat_scopes()
    brain = data_adapter.get_brain_by_pat(pat)
    if not brain:
        return None, []
    return principal_for_brain(brain.name_key), effective_pat_scopes(brain)


def scopes_tuple(scopes: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(scope for scope in (scopes or []) if scope)

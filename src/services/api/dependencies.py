from fastapi import Request

from src.core.auth.scopes import (
    VERIFICATION_PERMISSION_DENIED,
    VERIFICATION_SCOPE,
    has_verification_scope,
    is_system_principal,
    principal_for_user,
)
from src.services.api.errors import CodedHTTPException

USAGE_CONTEXT_HEADER = "X-Usage-Context"


def get_brain_id(request: Request) -> str:
    return request.state.brain_id


def get_auth_principal(request: Request) -> str | None:
    principal = getattr(request.state, "principal", None)
    if principal:
        return str(principal)
    return None


def get_auth_scopes(request: Request) -> list[str]:
    scopes = getattr(request.state, "scopes", None) or []
    return [str(scope) for scope in scopes if scope]


def verification_denied_exception() -> CodedHTTPException:
    return CodedHTTPException(
        status_code=403,
        detail="verification scope is required",
        code=VERIFICATION_PERMISSION_DENIED,
        message="This token cannot verify, reject, or reset facts.",
        resolution=(
            "Use an OAuth token that includes the verification scope, or grant "
            f"{VERIFICATION_SCOPE} on the brain PAT. Existing tokens do not "
            "receive this scope by default."
        ),
    )


def _usage_context_user_id(request: Request) -> str | None:
    token = request.headers.get(USAGE_CONTEXT_HEADER) or request.headers.get(
        "Usage-Context"
    )
    if not token:
        return None
    from src.core.saving.ingest_manifest import decode_signed_usage_context

    body = decode_signed_usage_context(token)
    if not body:
        return None
    for key in ("actorUserId", "actor_user_id", "userId", "user_id"):
        value = body.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    user = body.get("user")
    if isinstance(user, dict):
        for key in ("id", "userId", "user_id", "actorUserId"):
            value = user.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return None


def resolve_verification_actor(request: Request, principal: str) -> str:
    if not is_system_principal(principal):
        return principal
    user_id = _usage_context_user_id(request)
    if not user_id:
        return principal
    return principal_for_user(user_id)


def resolve_request_actor(request: Request) -> str:
    from src.core.auth.scopes import principal_for_system

    principal = get_auth_principal(request) or principal_for_system()
    return resolve_verification_actor(request, principal)


def require_verification_permission(request: Request) -> str:
    if not has_verification_scope(get_auth_scopes(request)):
        raise verification_denied_exception()
    principal = get_auth_principal(request)
    if not principal:
        raise verification_denied_exception()
    return resolve_verification_actor(request, principal)

from fastapi import Request

from src.core.auth.scopes import (
    VERIFICATION_PERMISSION_DENIED,
    VERIFICATION_SCOPE,
    has_verification_scope,
)
from src.services.api.errors import CodedHTTPException


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


def require_verification_permission(request: Request) -> str:
    if not has_verification_scope(get_auth_scopes(request)):
        raise CodedHTTPException(
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
    principal = get_auth_principal(request)
    if not principal:
        raise CodedHTTPException(
            status_code=403,
            detail="verification scope is required",
            code=VERIFICATION_PERMISSION_DENIED,
            message="This token cannot verify, reject, or reset facts.",
            resolution=(
                "Authenticate with a principal that holds the verification scope."
            ),
        )
    return principal

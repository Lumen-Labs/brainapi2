from __future__ import annotations

from typing import Any


class BrainResolutionError(ValueError):
    def __init__(self, code: str, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.hint = hint


def resolve_brain(
    token: str | None,
    brain_id: str | None = None,
    *,
    data_adapter: Any | None = None,
    guard=None,
) -> str:
    from src.services.mcp.utils import guard_brainpat

    checker = guard or guard_brainpat
    requested = (brain_id or "").strip() or None
    if not token:
        raise BrainResolutionError(
            "unauthorized",
            "Sign in with OAuth or pass a valid BrainPAT.",
            hint="Authorize the MCP client, then retry without writing a query.",
        )

    access = checker(token, requested)
    if access is False:
        if requested:
            raise BrainResolutionError(
                "unauthorized",
                f"This token cannot access brain '{requested}'.",
                hint="Use a BrainPAT for that brain, or omit brain_id to use the token default.",
            )
        raise BrainResolutionError(
            "unauthorized",
            "This token is not valid.",
            hint="Re-authorize the MCP client with a BrainPAT.",
        )

    if isinstance(access, str) and access.strip():
        return access.strip()

    adapter = data_adapter
    if adapter is None:
        from src.core.instances import data_adapter as default_data

        adapter = default_data
    brains = list(adapter.get_brains_list() or [])
    keys = [getattr(brain, "name_key", None) or str(brain) for brain in brains]
    keys = [key for key in keys if key]
    if requested:
        if requested in keys or access is True:
            return requested
        raise BrainResolutionError(
            "not_found",
            f"Brain '{requested}' was not found.",
            hint="Call list_brains and pass one of those ids.",
        )
    if len(keys) == 1:
        return keys[0]
    if not keys:
        raise BrainResolutionError(
            "no_brain",
            "This token is not tied to a brain.",
            hint="Create a brain and authorize with its BrainPAT, or pass brain_id.",
        )
    listed = ", ".join(keys[:12])
    raise BrainResolutionError(
        "ambiguous_brain",
        "This token can access more than one brain. Pass brain_id.",
        hint=f"Choose one of: {listed}",
    )


def writer_from_token(token: str | None, brain_id: str) -> str:
    import hashlib

    if not token:
        return f"{brain_id}:anonymous"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    return f"{brain_id}:{digest}"

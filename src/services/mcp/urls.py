"""Resolve public MCP URLs and ignore env-template placeholders."""

from __future__ import annotations

import os


_PLACEHOLDER_MARKERS = (
    "your-mcp-",
    "your-app-host",
    "example.invalid",
)


def configured_url(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return ""
    return text.rstrip("/")


def public_base_url() -> str:
    for key in ("BRAINAPI_PUBLIC_URL", "PUBLIC_BASE_URL"):
        value = configured_url(os.getenv(key))
        if value:
            return value
    return ""


def mcp_resource_url(issuer: str | None, resource: str | None) -> str:
    resolved_resource = configured_url(resource)
    if resolved_resource:
        return resolved_resource
    base = public_base_url() or configured_url(issuer)
    if not base:
        return ""
    if base.endswith("/mcp"):
        return base
    return base + "/mcp"

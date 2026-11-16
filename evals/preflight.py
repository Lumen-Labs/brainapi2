from __future__ import annotations

from typing import Any, Iterable

import httpx

from evals.config import (
    ALLOWED_LLM_ALIASES,
    DEFAULT_LLM_MODEL,
    Settings,
    canonicalize_llm_model,
)

PROBE_PATHS = (
    "/health",
    "/config",
    "/system/config",
    "/system/info",
    "/info",
    "/status",
    "/llm",
    "/models",
)
_EMBEDDING_MARKERS = (
    "embed",
    "minilm",
    "e5-",
    "bge-",
    "text-embedding",
    "cross-encoder",
)
_CHAT_MARKERS = (
    "claude",
    "gpt-",
    "gpt4",
    "o1-",
    "o3-",
    "deepseek",
    "sonnet",
    "haiku",
    "opus",
    "fable",
    "gemini",
    "mistral",
    "llama",
)
def is_haiku_llm(value: str) -> bool:
    key = (value or "").strip().lower().replace("_", "-")
    return key in ALLOWED_LLM_ALIASES


def looks_like_embedding(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _EMBEDDING_MARKERS)


def looks_like_chat_model(value: str) -> bool:
    if looks_like_embedding(value):
        return False
    lowered = value.lower()
    return any(marker in lowered for marker in _CHAT_MARKERS)


def _walk_strings(obj: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            yield from _walk_strings(value, child)
        return
    if isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from _walk_strings(value, f"{path}[{index}]")
        return
    if isinstance(obj, str) and obj.strip():
        yield path, obj.strip()


def collect_reported_chat_models(payloads: dict[str, Any]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source, payload in payloads.items():
        for path, value in _walk_strings(payload):
            if looks_like_embedding(value) or not looks_like_chat_model(value):
                continue
            key = (f"{source}:{path}", value)
            if key in seen:
                continue
            seen.add(key)
            found.append({"source": source, "path": path, "model": value})
    return found


def fetch_probe_payloads(settings: Settings) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if settings.brainpat_token:
        headers["Authorization"] = f"Bearer {settings.brainpat_token}"
    payloads: dict[str, Any] = {}
    with httpx.Client(
        base_url=settings.brainapi_url,
        headers=headers,
        timeout=10.0,
    ) as client:
        for path in PROBE_PATHS:
            try:
                response = client.get(path)
            except httpx.HTTPError:
                continue
            if response.status_code >= 400 or not response.content:
                continue
            try:
                data = response.json()
            except Exception:
                continue
            if data is not None:
                payloads[path] = data
    return payloads


def preflight_brainapi_haiku(
    settings: Settings,
    *,
    payloads: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probes = payloads if payloads is not None else fetch_probe_payloads(settings)
    reported = collect_reported_chat_models(probes)
    if reported:
        bad = [row for row in reported if not is_haiku_llm(row["model"])]
        if bad:
            detail = ", ".join(f"{row['source']} {row['path']}={row['model']!r}" for row in bad)
            raise SystemExit(
                "BrainAPI preflight failed: target reports a chat LLM that is not "
                f"Claude Haiku 5.5 ({DEFAULT_LLM_MODEL}): {detail}"
            )
        return {
            "ok": True,
            "mode": "reported",
            "llm_model": DEFAULT_LLM_MODEL,
            "reported": reported,
            "probed_paths": sorted(probes),
        }

    confirmed = (settings.brainapi_llm_confirmed or "").strip()
    if confirmed:
        canonicalize_llm_model(confirmed, source="EVAL_BRAINAPI_LLM_CONFIRMED")
        return {
            "ok": True,
            "mode": "operator_confirmed",
            "llm_model": DEFAULT_LLM_MODEL,
            "reported": [],
            "probed_paths": sorted(probes),
            "note": (
                "Target /health and config endpoints did not report a chat model; "
                "EVAL_BRAINAPI_LLM_CONFIRMED=claude-haiku-5-5 was accepted."
            ),
        }
    raise SystemExit(
        "BrainAPI preflight failed: enrichment is on for real comparisons, but the "
        "target did not report Claude Haiku 5.5 on /health or config endpoints. "
        "Set EVAL_BRAINAPI_LLM_CONFIRMED=claude-haiku-5-5 after confirming the "
        "stack's server LLM routing is Haiku 5.5, or wait for that health field "
        f"on the Haiku routing PR. Required model id: {DEFAULT_LLM_MODEL}."
    )

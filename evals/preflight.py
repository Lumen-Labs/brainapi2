from __future__ import annotations

from typing import Any, Iterable

import httpx

from evals.config import Settings

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
    "opus",
    "fable",
    "gemini",
    "mistral",
    "llama",
)


def normalize_model_id(value: str) -> str:
    key = (value or "").strip().lower().replace("_", "-")
    for sep in ("/", ":"):
        if sep in key:
            key = key.rsplit(sep, 1)[-1]
    return key


def model_ids_match(left: str, right: str) -> bool:
    a = normalize_model_id(left)
    b = normalize_model_id(right)
    return bool(a) and a == b


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


def preflight_brainapi_model(
    settings: Settings,
    *,
    payloads: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probes = payloads if payloads is not None else fetch_probe_payloads(settings)
    reported = collect_reported_chat_models(probes)
    expected = (settings.expected_brainapi_model or "").strip()
    base = {
        "ok": True,
        "llm_model": settings.llm_model,
        "expected": expected,
        "reported": reported,
        "probed_paths": sorted(probes),
    }
    if not expected:
        return {
            **base,
            "mode": "unchecked",
            "note": (
                "EVAL_EXPECTED_BRAINAPI_MODEL is unset; BrainAPI chat model "
                "was not verified."
            ),
        }
    if not reported:
        raise SystemExit(
            "BrainAPI preflight failed: "
            f"EVAL_EXPECTED_BRAINAPI_MODEL={expected!r} is set, but /health "
            "and config endpoints did not report a chat model."
        )
    mismatches = [
        row for row in reported if not model_ids_match(row["model"], expected)
    ]
    if mismatches:
        detail = ", ".join(
            f"{row['source']} {row['path']}={row['model']!r}" for row in mismatches
        )
        raise SystemExit(
            "BrainAPI preflight failed: expected chat model "
            f"{expected!r}, target reported: {detail}"
        )
    return {**base, "mode": "reported"}

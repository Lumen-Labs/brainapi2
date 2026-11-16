"""Provider-neutral LLM routing: optional shared model/provider, per-role overrides."""

from __future__ import annotations

import logging
import os
from typing import Iterable

logger = logging.getLogger("brainapi.llm")

LLM_ROLES = ("small", "large")

_PROVIDER_ROLE_ENV = {
    "anthropic": {
        "small": "ANTHROPIC_SMALL_LLM_MODEL",
        "large": "ANTHROPIC_LARGE_LLM_MODEL",
    },
    "openai": {
        "small": "OPENAI_SMALL_LLM_MODEL",
        "large": "OPENAI_LARGE_LLM_MODEL",
    },
    "azure": {
        "small": "AZURE_SMALL_LLM_MODEL",
        "large": "AZURE_LARGE_LLM_MODEL",
    },
    "deepseek": {
        "small": "DEEPSEEK_SMALL_LLM_MODEL",
        "large": "DEEPSEEK_LARGE_LLM_MODEL",
    },
    "gcp_vertex": {
        "small": "GCP_SMALL_LLM_MODEL",
        "large": "GCP_LARGE_LLM_MODEL",
    },
    "anthropic_vertex": {
        "small": "GCP_SMALL_LLM_MODEL",
        "large": "GCP_LARGE_LLM_MODEL",
    },
    "amazon_bedrock": {
        "small": "BEDROCK_SMALL_LLM_MODEL",
        "large": "BEDROCK_LARGE_LLM_MODEL",
    },
    "ollama": {
        "small": "OLLAMA_LLM_SMALL_MODEL",
        "large": "OLLAMA_LLM_LARGE_MODEL",
    },
}

# Adapter defaults that already existed on main when a provider is selected
# and neither a per-role env nor LLM_MODEL is set. Providers with no entry
# (GCP Vertex, Bedrock, Ollama) still require their env model ids.
_PROVIDER_ROLE_DEFAULTS = {
    "openai": {"small": "gpt-4o-mini", "large": "gpt-4o"},
    "azure": {"small": "gpt-4o-mini"},
    "anthropic": {
        "small": "claude-3-5-haiku-latest",
        "large": "claude-sonnet-4-20250514",
    },
    "deepseek": {"small": "deepseek-v4-flash", "large": "deepseek-v4-pro"},
}


def configured_llm_model() -> str | None:
    raw = (os.getenv("LLM_MODEL") or "").strip()
    return raw or None


def allowed_llm_models(raw: str | None = None) -> list[str]:
    value = raw if raw is not None else os.getenv("LLM_ALLOWED_MODELS", "")
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def model_is_allowed(model: str | None, allowed: Iterable[str] | None = None) -> bool:
    names = list(allowed) if allowed is not None else allowed_llm_models()
    if not names:
        return True
    candidate = (model or "").strip().lower()
    return candidate in {item.strip().lower() for item in names}


def default_provider_for_mode(models_mode: str, role: str | None = None) -> str:
    if (models_mode or "").strip() == "local":
        return "ollama"
    shared = (os.getenv("LLM_PROVIDER") or "").strip()
    if role == "large":
        return (os.getenv("LLM_LARGE_PROVIDER") or shared or "azure").strip()
    if role == "small":
        return (os.getenv("LLM_SMALL_PROVIDER") or shared or "gcp_vertex").strip()
    return shared or "gcp_vertex"


def resolve_role_provider(role: str, models_mode: str) -> str:
    return default_provider_for_mode(models_mode, role)


def resolve_role_model(
    role: str, provider: str, llm_model: str | None = None
) -> str | None:
    if llm_model is None:
        llm_model = configured_llm_model()
    else:
        llm_model = llm_model.strip() or None
    env_key = (_PROVIDER_ROLE_ENV.get(provider) or {}).get(role)
    if env_key:
        override = (os.getenv(env_key) or "").strip()
        if override:
            return override
    if llm_model:
        return llm_model
    return (_PROVIDER_ROLE_DEFAULTS.get(provider) or {}).get(role)


def resolve_llm_roles(
    models_mode: str, llm_model: str | None = None
) -> list[tuple[str, str, str | None]]:
    rows: list[tuple[str, str, str | None]] = []
    for role in LLM_ROLES:
        provider = resolve_role_provider(role, models_mode)
        model = resolve_role_model(role, provider, llm_model)
        rows.append((role, provider, model))
    return rows


def format_role_line(role: str, provider: str, model: str | None) -> str:
    return f"LLM role {role}: provider={provider} model={model}"


def log_and_enforce_llm_roles(
    rows: Iterable[tuple[str, str, str | None]] | None = None,
    *,
    allowed: Iterable[str] | None = None,
) -> list[tuple[str, str, str | None]]:
    if rows is None:
        from src.core.saving.pipeline_runtime import configured_model

        resolved = [(role, *configured_model(role)) for role in LLM_ROLES]
    else:
        resolved = list(rows)
    allowed_names = (
        list(allowed) if allowed is not None else allowed_llm_models()
    )
    offenders: list[str] = []
    for role, provider, model in resolved:
        line = format_role_line(role, provider, model)
        logger.info(line)
        if not model_is_allowed(model, allowed_names):
            message = (
                f"{line} is not in LLM_ALLOWED_MODELS "
                f"({', '.join(allowed_names)})"
            )
            logger.warning(message)
            offenders.append(message)
    if offenders:
        raise RuntimeError(
            "LLM_ALLOWED_MODELS is set; refusing to start with other models: "
            + "; ".join(offenders)
        )
    return resolved

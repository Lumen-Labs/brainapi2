"""Single-model LLM routing: one default id, existing per-role overrides."""

from __future__ import annotations

import logging
import os
from typing import Iterable

logger = logging.getLogger("brainapi.llm")

DEFAULT_LLM_MODEL = "claude-haiku-5-5"
DEFAULT_REMOTE_PROVIDER = "anthropic"
HAIKU_MODEL_ID = "claude-haiku-5-5"
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

_PROVIDER_ROLE_DEFAULTS = {
    "openai": {"small": "gpt-4o-mini", "large": "gpt-4o"},
    "azure": {"small": "gpt-4o-mini", "large": "gpt-4o"},
    "deepseek": {"small": "deepseek-v4-flash", "large": "deepseek-v4-pro"},
}


def configured_llm_model() -> str:
    return (os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL


def llm_require_haiku(flag: str | None = None) -> bool:
    raw = (flag if flag is not None else os.getenv("LLM_REQUIRE_HAIKU", "false")).strip()
    return raw.lower() in {"1", "true", "yes", "on"}


def is_haiku_5_5(model: str | None) -> bool:
    name = (model or "").strip().lower()
    if name == HAIKU_MODEL_ID:
        return True
    return name.startswith(f"{HAIKU_MODEL_ID}-")


def default_provider_for_mode(models_mode: str) -> str:
    if (models_mode or "").strip() == "local":
        return "ollama"
    return (
        os.getenv("LLM_PROVIDER") or DEFAULT_REMOTE_PROVIDER
    ).strip() or DEFAULT_REMOTE_PROVIDER


def resolve_role_provider(role: str, models_mode: str) -> str:
    env_key = "LLM_LARGE_PROVIDER" if role == "large" else "LLM_SMALL_PROVIDER"
    return (os.getenv(env_key) or default_provider_for_mode(models_mode)).strip()


def resolve_role_model(
    role: str, provider: str, llm_model: str | None = None
) -> str:
    llm_model = (llm_model or configured_llm_model()).strip() or DEFAULT_LLM_MODEL
    env_key = (_PROVIDER_ROLE_ENV.get(provider) or {}).get(role)
    if env_key:
        override = (os.getenv(env_key) or "").strip()
        if override:
            return override
    if provider in {"anthropic", "anthropic_vertex", "gcp_vertex"}:
        return llm_model
    fallback = (_PROVIDER_ROLE_DEFAULTS.get(provider) or {}).get(role)
    return fallback or llm_model


def resolve_llm_roles(
    models_mode: str, llm_model: str | None = None
) -> list[tuple[str, str, str]]:
    llm_model = llm_model or configured_llm_model()
    rows: list[tuple[str, str, str]] = []
    for role in LLM_ROLES:
        provider = resolve_role_provider(role, models_mode)
        model = resolve_role_model(role, provider, llm_model)
        rows.append((role, provider, model))
    return rows


def format_role_line(role: str, provider: str, model: str) -> str:
    return f"LLM role {role}: provider={provider} model={model}"


def non_haiku_messages(rows: Iterable[tuple[str, str, str]]) -> list[str]:
    messages: list[str] = []
    for role, provider, model in rows:
        if not is_haiku_5_5(model):
            messages.append(
                f"{format_role_line(role, provider, model)} is not Claude Haiku 5.5 "
                f"({HAIKU_MODEL_ID})"
            )
    return messages


def log_and_enforce_llm_roles(
    rows: Iterable[tuple[str, str, str]] | None = None,
    *,
    strict: bool | None = None,
) -> list[tuple[str, str, str]]:
    if rows is None:
        from src.core.saving.pipeline_runtime import configured_model

        resolved = [(role, *configured_model(role)) for role in LLM_ROLES]
    else:
        resolved = list(rows)
    if strict is None:
        strict = llm_require_haiku()
    offenders: list[str] = []
    for role, provider, model in resolved:
        line = format_role_line(role, provider, model)
        logger.info(line)
        if not is_haiku_5_5(model):
            message = (
                f"{line} is not Claude Haiku 5.5 ({HAIKU_MODEL_ID})"
            )
            logger.warning(message)
            offenders.append(message)
    if offenders and strict:
        raise RuntimeError(
            "LLM_REQUIRE_HAIKU is enabled; refusing to start with non-Haiku models: "
            + "; ".join(offenders)
        )
    return resolved

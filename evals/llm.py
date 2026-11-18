from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from evals.config import Settings


def is_anthropic_api_host(host: str) -> bool:
    name = (host or "").lower().rstrip(".")
    return name == "anthropic.com" or name.endswith(".anthropic.com")


def chat_model(settings: Settings) -> str:
    settings.require_llm()
    return settings.llm_model.strip()


def chat_client(settings: Settings):
    from openai import OpenAI

    settings.require_llm()
    kwargs: dict[str, Any] = {"api_key": settings.llm_api_key}
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    host = urlparse(settings.llm_base_url or "").hostname or ""
    provider = (settings.llm_provider or "").strip().lower()
    if is_anthropic_api_host(host) or provider == "anthropic":
        kwargs["default_headers"] = {"anthropic-version": "2023-06-01"}
    return OpenAI(**kwargs)

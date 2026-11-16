from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from evals.config import Settings, canonicalize_llm_model


def chat_model(settings: Settings) -> str:
    return canonicalize_llm_model(settings.llm_model, source="llm_model")


def chat_client(settings: Settings):
    from openai import OpenAI

    settings.require_llm()
    chat_model(settings)
    kwargs: dict[str, Any] = {"api_key": settings.llm_api_key}
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    host = (urlparse(settings.llm_base_url or "").hostname or "").lower()
    if host.endswith("anthropic.com"):
        kwargs["default_headers"] = {"anthropic-version": "2023-06-01"}
    return OpenAI(**kwargs)

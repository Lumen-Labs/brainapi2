import json
import os
import threading
from typing import Any

from anthropic import Anthropic
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.adapters.interfaces.llm import LLM
from src.config import config


def _anthropic_openai_kwargs() -> dict[str, Any]:
    return {
        "api_key": config.anthropic.api_key,
        "base_url": config.anthropic.openai_compatible_base_url(),
        "timeout": 120.0,
        "max_retries": 3,
        "default_headers": {"anthropic-version": "2023-06-01"},
    }


def _anthropic_sdk_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": config.anthropic.api_key}
    native = config.anthropic.native_sdk_base_url()
    if native:
        kwargs["base_url"] = native
    return kwargs


class _AnthropicLLMBase(LLM):
    def __init__(self, model: str):
        self.model = model
        self._client: Anthropic | None = None
        self._openai_client: OpenAI | None = None
        self._langchain_model = None
        self._lock = threading.Lock()
        self._pid = os.getpid()

    def _check_fork(self) -> None:
        if os.getpid() != self._pid:
            self._client = None
            self._openai_client = None
            self._langchain_model = None
            self._pid = os.getpid()

    @property
    def client(self) -> Anthropic:
        self._check_fork()
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = Anthropic(**_anthropic_sdk_kwargs())
        return self._client

    @property
    def openai_client(self) -> OpenAI:
        self._check_fork()
        if self._openai_client is None:
            with self._lock:
                if self._openai_client is None:
                    self._openai_client = OpenAI(**_anthropic_openai_kwargs())
        return self._openai_client

    def _build_langchain_model(self):
        try:
            from langchain_anthropic import ChatAnthropic

            kwargs: dict[str, Any] = {
                "model": self.model,
                "api_key": config.anthropic.api_key,
                "timeout": 120,
                "max_retries": 3,
            }
            native = config.anthropic.native_sdk_base_url()
            if native:
                kwargs["base_url"] = native
            return ChatAnthropic(**kwargs)
        except ImportError:
            openai_kwargs = dict(_anthropic_openai_kwargs())
            try:
                return ChatOpenAI(model=self.model, **openai_kwargs)
            except TypeError:
                openai_kwargs.pop("default_headers", None)
                return ChatOpenAI(model=self.model, **openai_kwargs)

    @property
    def langchain_model(self):
        self._check_fork()
        if self._langchain_model is None:
            with self._lock:
                if self._langchain_model is None:
                    self._langchain_model = self._build_langchain_model()
        return self._langchain_model

    def generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens or 1024,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [
            block.text for block in response.content if getattr(block, "text", None)
        ]
        return "".join(parts)

    def generate_json(
        self, prompt: str, max_new_tokens: int = None, max_retries: int = 3
    ) -> dict:
        response = None
        while max_retries > 0 and response is None:
            try:
                raw = self.generate_text(prompt, max_new_tokens=max_new_tokens)
                response = json.loads(raw.strip().strip("```json").strip("```"))
            except Exception as e:
                max_retries -= 1
                if max_retries <= 0:
                    raise e
        return response


class AnthropicLLMClientSmall(_AnthropicLLMBase):
    def __init__(self):
        super().__init__(config.anthropic.small_llm_model)


class AnthropicLLMClientLarge(_AnthropicLLMBase):
    def __init__(self):
        super().__init__(config.anthropic.large_llm_model)


if getattr(config, "anthropic", None) is not None:
    _llm_small_client_anthropic = AnthropicLLMClientSmall()
    _llm_large_client_anthropic = AnthropicLLMClientLarge()
else:
    _llm_small_client_anthropic = None
    _llm_large_client_anthropic = None

import json
import os
import threading
from typing import Any

from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.adapters.interfaces.llm import LLM
from src.config import config


def _openai_client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "api_key": config.openai.api_key,
        "timeout": 120.0,
        "max_retries": 3,
    }
    if config.openai.base_url:
        kwargs["base_url"] = config.openai.base_url
    return kwargs


class _OpenAILLMBase(LLM):
    def __init__(self, model: str):
        self.model = model
        self._client: OpenAI | None = None
        self._langchain_model: ChatOpenAI | None = None
        self._lock = threading.Lock()
        self._pid = os.getpid()

    def _check_fork(self) -> None:
        if os.getpid() != self._pid:
            self._client = None
            self._langchain_model = None
            self._pid = os.getpid()

    @property
    def client(self) -> OpenAI:
        self._check_fork()
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = OpenAI(**_openai_client_kwargs())
        return self._client

    @property
    def langchain_model(self) -> ChatOpenAI:
        self._check_fork()
        if self._langchain_model is None:
            with self._lock:
                if self._langchain_model is None:
                    self._langchain_model = ChatOpenAI(
                        model=self.model, **_openai_client_kwargs()
                    )
        return self._langchain_model

    def generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **({"max_tokens": max_new_tokens} if max_new_tokens else {}),
        )
        return response.choices[0].message.content

    def generate_json(
        self, prompt: str, max_new_tokens: int = None, max_retries: int = 3
    ) -> dict:
        response = None
        while max_retries > 0 and response is None:
            try:
                _response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **({"max_tokens": max_new_tokens} if max_new_tokens else {}),
                    response_format={"type": "json_object"},
                )
                _response = (
                    _response.choices[0].message.content.strip("```json").strip("```")
                )
                response = json.loads(_response)
            except Exception as e:
                max_retries -= 1
                if max_retries <= 0:
                    raise e
        return response


class OpenAILLMClientSmall(_OpenAILLMBase):
    def __init__(self):
        super().__init__(config.openai.small_llm_model)


class OpenAILLMClientLarge(_OpenAILLMBase):
    def __init__(self):
        super().__init__(config.openai.large_llm_model)


_llm_small_client_openai = OpenAILLMClientSmall()
_llm_large_client_openai = OpenAILLMClientLarge()

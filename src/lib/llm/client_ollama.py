"""
File: /client_ollama.py
Project: llm
Created Date: Sunday February 22nd 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday February 22nd 2026 5:22:16 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
import logging
from typing import Any

from openai import OpenAI
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from src.adapters.interfaces.llm import LLM
from src.config import config

logger = logging.getLogger(__name__)


def _ollama_base_url():
    return f"http://{config.ollama.host}:{config.ollama.port}/v1"


class ChatOllamaWithThinking(ChatOpenAI):
    def __init__(self, **kwargs: Any):
        kwargs.setdefault("model_kwargs", {})
        kwargs["extra_body"] = kwargs.get("extra_body") or {}
        super().__init__(**kwargs)

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        for i, choice in enumerate(response_dict.get("choices") or []):
            msg_dict = choice.get("message") or {}
            if "thinking" in msg_dict and i < len(result.generations):
                thinking = msg_dict["thinking"]
                result.generations[i].message.additional_kwargs["thinking"] = thinking
                if thinking:
                    logger.info("thinking: %s", thinking)
        return result


class LLMClientLarge(LLM):
    def __init__(self):
        self.model = config.ollama.llm_large_model
        self.client = OpenAI(
            base_url=f"{_ollama_base_url()}/",
            api_key="ollama",
        )
        self.langchain_model = ChatOllamaWithThinking(
            base_url=f"{_ollama_base_url()}/",
            api_key="ollama",
            model=self.model,
        )

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


_llm_large_client = LLMClientLarge()


class LLMClientSmall(LLM):
    def __init__(self):
        self.model = config.ollama.llm_small_model
        self.client = OpenAI(
            base_url=f"{_ollama_base_url()}/",
            api_key="ollama",
        )
        self.langchain_model = ChatOllamaWithThinking(
            base_url=f"{_ollama_base_url()}/",
            api_key="ollama",
            model=self.model,
        )
        self.default_timeout = 120

    def generate_text(
        self, prompt: str, max_new_tokens: int = None, timeout: int = None
    ) -> str:
        if not prompt or len(prompt.strip()) == 0:
            return "Input prompt is empty"
        timeout = timeout or self.default_timeout
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **({"max_tokens": max_new_tokens} if max_new_tokens else {}),
            timeout=timeout,
        )
        return response.choices[0].message.content

    def generate_json(
        self,
        prompt: str,
        max_new_tokens: int = None,
        max_retries: int = 3,
        timeout: int = None,
    ) -> dict:
        timeout = timeout or self.default_timeout
        response = None
        last_error = None
        for _ in range(max_retries):
            try:
                _response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **({"max_tokens": max_new_tokens} if max_new_tokens else {}),
                    response_format={"type": "json_object"},
                    timeout=timeout,
                )
                raw = (
                    _response.choices[0]
                    .message.content.strip("```json")
                    .strip("```")
                    .strip()
                )
                return json.loads(raw)
            except (json.JSONDecodeError, Exception) as e:
                last_error = e
        raise last_error


_llm_small_client = LLMClientSmall()

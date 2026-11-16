import json
import os
import threading

import boto3

from src.adapters.interfaces.llm import LLM
from src.config import config


class _BedrockLLMBase(LLM):
    def __init__(self, model: str):
        self.model = model
        self._client = None
        self._langchain_model = None
        self._lock = threading.Lock()
        self._pid = os.getpid()

    def _check_fork(self) -> None:
        if os.getpid() != self._pid:
            self._client = None
            self._langchain_model = None
            self._pid = os.getpid()

    def _session(self):
        return boto3.Session(**config.bedrock.session_kwargs())

    @property
    def client(self):
        self._check_fork()
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._session().client(
                        "bedrock-runtime", **config.bedrock.runtime_client_kwargs()
                    )
        return self._client

    def _build_langchain_model(self):
        from langchain_aws import ChatBedrockConverse

        kwargs = {
            "model": self.model,
            "region_name": config.bedrock.region,
        }
        if config.bedrock.endpoint_url:
            kwargs["endpoint_url"] = config.bedrock.endpoint_url
        if config.bedrock.access_key_id and config.bedrock.secret_access_key:
            kwargs["aws_access_key_id"] = config.bedrock.access_key_id
            kwargs["aws_secret_access_key"] = config.bedrock.secret_access_key
            if config.bedrock.session_token:
                kwargs["aws_session_token"] = config.bedrock.session_token
        return ChatBedrockConverse(**kwargs)

    @property
    def langchain_model(self):
        self._check_fork()
        if self._langchain_model is None:
            with self._lock:
                if self._langchain_model is None:
                    self._langchain_model = self._build_langchain_model()
        return self._langchain_model

    def _converse(self, prompt: str, max_new_tokens: int = None) -> str:
        payload = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
        }
        if max_new_tokens:
            payload["inferenceConfig"] = {"maxTokens": max_new_tokens}
        response = self.client.converse(**payload)
        from src.core.saving.pipeline_runtime import record_call

        usage = response.get("usage") or {}
        record_call(
            {
                "usage_metadata": {
                    "input_tokens": usage.get("inputTokens"),
                    "output_tokens": usage.get("outputTokens"),
                    "cache_read_input_tokens": usage.get("cacheReadInputTokens", 0),
                    "cache_write_input_tokens": usage.get("cacheWriteInputTokens", 0),
                }
            }
            if usage
            else None,
            identity=("amazon_bedrock", self.model),
        )
        content = response["output"]["message"]["content"]
        texts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(texts).strip()

    def generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        return self._converse(prompt, max_new_tokens=max_new_tokens)

    def generate_json(
        self, prompt: str, max_new_tokens: int = None, max_retries: int = 3
    ) -> dict:
        while max_retries > 0:
            try:
                response = self._converse(prompt, max_new_tokens=max_new_tokens)
                raw = response.strip().strip("```json").strip("```").strip()
                return json.loads(raw)
            except Exception:
                max_retries -= 1
                if max_retries <= 0:
                    raise
        return {}


class BedrockLLMClientSmall(_BedrockLLMBase):
    def __init__(self):
        super().__init__(config.bedrock.small_llm_model)


class BedrockLLMClientLarge(_BedrockLLMBase):
    def __init__(self):
        super().__init__(config.bedrock.large_llm_model)


if getattr(config, "bedrock", None) is not None:
    _llm_small_client_bedrock = BedrockLLMClientSmall()
    _llm_large_client_bedrock = BedrockLLMClientLarge()
else:
    _llm_small_client_bedrock = None
    _llm_large_client_bedrock = None

import os
import threading
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    RetryError,
)

from src.adapters.interfaces.embeddings import EmbeddingsClient
from src.config import config
from src.lib.embeddings.client import EmbeddingError


def _openai_client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "api_key": config.openai.api_key,
        "timeout": 60.0,
        "max_retries": 3,
    }
    if config.openai.base_url:
        kwargs["base_url"] = config.openai.base_url
    return kwargs


class OpenAIEmbeddingsClient(EmbeddingsClient):
    def __init__(self):
        self._client: OpenAI | None = None
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self.model = config.openai.embedding_model

    def _check_fork(self) -> None:
        if os.getpid() != self._pid:
            self._client = None
            self._pid = os.getpid()

    @property
    def client(self) -> OpenAI:
        self._check_fork()
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = OpenAI(**_openai_client_kwargs())
        return self._client

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
        reraise=True,
    )
    def _embed_text_with_retry(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
        reraise=True,
    )
    def _embed_texts_with_retry(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        sorted_data = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in sorted_data]

    def embed_text(self, text: str) -> list[float]:
        try:
            return self._embed_text_with_retry(text)
        except RetryError as e:
            last = e.last_attempt.exception()
            raise EmbeddingError(
                f"OpenAI embedding failed after retries: {last}"
            ) from last
        except (APIConnectionError, APITimeoutError) as e:
            raise EmbeddingError(f"OpenAI embedding connection error: {e}") from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._embed_texts_with_retry(texts)
        except RetryError as e:
            last = e.last_attempt.exception()
            raise EmbeddingError(
                f"OpenAI embedding failed after retries: {last}"
            ) from last
        except (APIConnectionError, APITimeoutError) as e:
            raise EmbeddingError(f"OpenAI embedding connection error: {e}") from e


_embeddings_openai_client = OpenAIEmbeddingsClient()

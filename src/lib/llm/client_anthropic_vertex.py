import json
import os
import threading

from anthropic import AnthropicVertex
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from src.adapters.interfaces.llm import LLM
from src.config import config


def _vertex_region() -> str:
    return getattr(config.gcp, "location", None) or os.getenv(
        "GCP_VERTEX_REGION", "global"
    )


class _AnthropicVertexLLMBase(LLM):
    def __init__(self, model: str):
        self.model = model
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._client = None
        self._langchain_model = None

    def _check_fork(self):
        if os.getpid() != self._pid:
            self._client = None
            self._langchain_model = None
            self._pid = os.getpid()

    def _prepare_credentials(self) -> None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.gcp.credentials_path

    @property
    def client(self):
        self._check_fork()
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._prepare_credentials()
                    self._client = AnthropicVertex(
                        region=_vertex_region(),
                        project_id=config.gcp.project_id,
                    )
        return self._client

    @property
    def langchain_model(self):
        self._check_fork()
        if self._langchain_model is None:
            with self._lock:
                if self._langchain_model is None:
                    self._prepare_credentials()
                    self._langchain_model = ChatAnthropicVertex(
                        model_name=self.model,
                        project=config.gcp.project_id,
                        location=_vertex_region(),
                        max_retries=5,
                        timeout=120,
                    )
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


class AnthropicVertexLLMClientSmall(_AnthropicVertexLLMBase):
    def __init__(self):
        super().__init__(config.gcp.small_llm_model)


class AnthropicVertexLLMClientLarge(_AnthropicVertexLLMBase):
    def __init__(self):
        super().__init__(config.gcp.large_llm_model or config.gcp.small_llm_model)


_llm_small_client_anthropic_vertex = AnthropicVertexLLMClientSmall()
_llm_large_client_anthropic_vertex = AnthropicVertexLLMClientLarge()

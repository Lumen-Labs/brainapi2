import json
import os

from langchain_google_vertexai import ChatVertexAI
from google import genai
from google.genai.types import GenerateContentConfig

from src.adapters.interfaces.llm import LLM
from src.config import config


class _VertexLLMBase(LLM):
    def __init__(self, model: str):
        self.model = model
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.gcp.credentials_path
        self.client = genai.Client(
            vertexai=True,
            project=config.gcp.project_id,
            location="global",
            http_options={"api_version": "v1"},
        )
        self.langchain_model = ChatVertexAI(
            model_name=model,
            project=config.gcp.project_id,
            location="global",
            max_retries=5,
            request_timeout=120,
            streaming=False,
        )

    def generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=GenerateContentConfig(
                response_mime_type="text/plain",
                **({"max_output_tokens": max_new_tokens} if max_new_tokens else {}),
            ),
        )
        return response.text

    def generate_json(
        self, prompt: str, max_new_tokens: int = None, max_retries: int = 3
    ) -> dict:
        while max_retries > 0:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt],
                    config=GenerateContentConfig(
                        response_mime_type="application/json",
                        **({"max_output_tokens": max_new_tokens} if max_new_tokens else {}),
                    ),
                )
                raw = response.text.strip().strip("```json").strip("```").strip()
                return json.loads(raw)
            except Exception:
                max_retries -= 1
                if max_retries <= 0:
                    raise
        return {}


class VertexLLMClientSmall(_VertexLLMBase):
    def __init__(self):
        super().__init__(config.gcp.small_llm_model)


class VertexLLMClientLarge(_VertexLLMBase):
    def __init__(self):
        super().__init__(config.gcp.large_llm_model)


_llm_small_client_vertex = VertexLLMClientSmall()
_llm_large_client_vertex = VertexLLMClientLarge()

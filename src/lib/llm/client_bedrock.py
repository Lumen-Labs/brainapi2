import json

import boto3
from langchain_aws import ChatBedrockConverse

from src.adapters.interfaces.llm import LLM
from src.config import config


class _BedrockLLMBase(LLM):
    def __init__(self, model: str):
        self.model = model
        session = boto3.Session(
            aws_access_key_id=config.bedrock.access_key_id,
            aws_secret_access_key=config.bedrock.secret_access_key,
            aws_session_token=config.bedrock.session_token,
            region_name=config.bedrock.region,
        )
        self.client = session.client("bedrock-runtime")
        self.langchain_model = ChatBedrockConverse(
            model=model,
            region_name=config.bedrock.region,
            aws_access_key_id=config.bedrock.access_key_id,
            aws_secret_access_key=config.bedrock.secret_access_key,
            aws_session_token=config.bedrock.session_token,
        )

    def _converse(self, prompt: str, max_new_tokens: int = None) -> str:
        payload = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
        }
        if max_new_tokens:
            payload["inferenceConfig"] = {"maxTokens": max_new_tokens}
        response = self.client.converse(**payload)
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


_llm_small_client_bedrock = BedrockLLMClientSmall()
_llm_large_client_bedrock = BedrockLLMClientLarge()

import json

from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

from src.adapters.interfaces.llm import LLM
from src.config import config


class _AzureLLMBase(LLM):
    def __init__(self, model: str):
        self.model = model
        self.client = AzureOpenAI(
            api_version=config.azure.llm_api_version,
            azure_endpoint=config.azure.llm_endpoint,
            api_key=config.azure.llm_subscription_key,
        )
        self.langchain_model = AzureChatOpenAI(
            azure_deployment=model,
            api_version=config.azure.llm_api_version,
            azure_endpoint=config.azure.llm_endpoint,
            api_key=config.azure.llm_subscription_key,
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


class AzureLLMClientSmall(_AzureLLMBase):
    def __init__(self):
        super().__init__(config.azure.small_llm_model)


class AzureLLMClientLarge(_AzureLLMBase):
    def __init__(self):
        super().__init__(config.azure.large_llm_model)


_llm_small_client_azure = AzureLLMClientSmall()
_llm_large_client_azure = AzureLLMClientLarge()

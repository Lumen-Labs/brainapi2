"""
File: /client_large.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 10:39:17 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import json
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

from src.adapters.interfaces.llm import LLM
from src.config import config


class LLMClientLarge(LLM):
    """
    Large language model client, used for main agents and other high-level tasks.
    """

    def __init__(self):
        self.model = config.azure.large_llm_model
        self.client = AzureOpenAI(
            api_version=config.azure.large_llm_api_version,
            azure_endpoint=config.azure.large_llm_endpoint,
            api_key=config.azure.large_llm_subscription_key,
        )

        self.langchain_model = AzureChatOpenAI(
            azure_deployment=config.azure.large_llm_model,
            api_version=config.azure.large_llm_api_version,
            azure_endpoint=config.azure.large_llm_endpoint,
            api_key=config.azure.large_llm_subscription_key,
        )

    def generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        """
        Generate a text response from the model and return it as a string.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content

    def generate_json(
        self, prompt: str, max_new_tokens: int = None, max_retries: int = 3
    ) -> dict:
        """
        Generate a JSON response from the model and return it as a dictionary.
        """
        response = None
        while max_retries > 0 and response is None:
            try:
                _response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    response_format={"type": "json_object"},
                )
                _response = (
                    _response.choices[0].message.content.strip("```json").strip("```")
                )
                response = json.loads(_response)
            except Exception as e:  # pylint: disable=broad-exception-caught
                max_retries -= 1
                if max_retries <= 0:
                    print(f"Failed to generate JSON response: {e}")
                    raise e
        return response


_llm_large_client = LLMClientLarge()

"""
File: /llm.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 10:23:50 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from .interfaces.llm import LLM


class LLMAdapter:
    """
    Adapter for the LLM client.
    """

    def __init__(self):
        self.llm = None

    def add_client(self, client: LLM) -> None:
        """
        Add a LLM client to the adapter.
        """
        self.llm = client

    def generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        """
        Generate a text response from the model and return it as a string.
        """
        return self.llm.generate_text(prompt, max_new_tokens)

    def generate_json(
        self, prompt: str, max_new_tokens: int = None, max_retries: int = 3
    ) -> dict:
        """
        Generate a JSON response from the model and return it as a dictionary.
        """
        return self.llm.generate_json(prompt, max_new_tokens, max_retries)

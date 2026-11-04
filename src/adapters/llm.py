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

    def __init__(self, role="small"):
        self._role = role
        self._llm = None

    @property
    def llm(self):
        from src.core.saving.pipeline_runtime import routed_client
        return routed_client(self._llm, self._role)

    @llm.setter
    def llm(self, value):
        self._llm = value

    def add_client(self, client: LLM) -> None:
        """
        Add a LLM client to the adapter.
        """
        self.llm = client

    def generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        """
        Generate a text response from the model and return it as a string.
        """
        client = self.llm
        try:
            result = client.generate_text(prompt, max_new_tokens)
        except Exception as error:
            from src.core.saving.pipeline_runtime import record_call
            record_call(None, error)
            raise
        if "bedrock" not in type(client).__module__:
            from src.core.saving.pipeline_runtime import record_call
            from src.core.saving.ingest_cost import estimate_tokens
            record_call({"usage_metadata":{"input_tokens":estimate_tokens(prompt),"output_tokens":estimate_tokens(result)}}, quality="estimated")
        return result

    def generate_json(
        self, prompt: str, max_new_tokens: int = None, max_retries: int = 3
    ) -> dict:
        """
        Generate a JSON response from the model and return it as a dictionary.
        """
        return self.llm.generate_json(prompt, max_new_tokens, max_retries)

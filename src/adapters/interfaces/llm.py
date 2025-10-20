"""
File: /llm.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 10:23:54 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from abc import ABC, abstractmethod

from langchain.chat_models.base import BaseChatModel


class LLM(ABC):
    """
    Abstract base class for LLM clients.
    """

    langchain_model: BaseChatModel

    @abstractmethod
    def generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        """
        Generate a text response from the model and return it as a string.
        """
        raise NotImplementedError("generate_text method not implemented")

    @abstractmethod
    def generate_json(
        self, prompt: str, max_new_tokens: int = None, max_retries: int = 3
    ) -> dict:
        """
        Generate a JSON response from the model and return it as a dictionary.
        """
        raise NotImplementedError("generate_json method not implemented")

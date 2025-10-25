"""
File: /embeddings.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 9:01:10 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from abc import ABC, abstractmethod

from src.constants.embeddings import Vector


class EmbeddingsClient(ABC):
    """
    Abstract base class for embeddings clients.
    """

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """
        Embed a text and return a list of floats.
        """
        raise NotImplementedError("embed method not implemented")


class VectorStoreClient(ABC):
    """
    Abstract base class for vector store clients.
    """

    @abstractmethod
    def add_vectors(self, vectors: list[Vector], store: str) -> None:
        """
        Add vectors to the vector store.
        """
        raise NotImplementedError("add_vectors method not implemented")

    @abstractmethod
    def search_vectors(self, query: str, store: str, k: int = 10) -> list[Vector]:
        """
        Search vectors in the vector store and return the top k vectors.
        """
        raise NotImplementedError("search_vectors method not implemented")

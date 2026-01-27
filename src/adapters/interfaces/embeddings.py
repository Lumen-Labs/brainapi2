"""
File: /embeddings.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 5th 2026 9:57:30 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
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
    def add_vectors(
        self, vectors: list[Vector], store: str, brain_id: str
    ) -> list[str]:
        """
        Add vectors to the vector store.
        """
        raise NotImplementedError("add_vectors method not implemented")

    @abstractmethod
    def search_vectors(
        self, data_vector: list[float], brain_id: str, store: str, k: int = 10
    ) -> list[Vector]:
        """
        Search vectors in the vector store and return the top k vectors.
        """
        raise NotImplementedError("search_vectors method not implemented")

    @abstractmethod
    def get_by_ids(self, ids: list[str], store: str, brain_id: str) -> list[Vector]:
        """
        Get vectors by their IDs.
        """
        raise NotImplementedError("get_by_ids method not implemented")

    @abstractmethod
    def search_similar_by_ids(
        self,
        vector_ids: list[str],
        brain_id: str,
        store: str,
        min_similarity: float,
        limit: int = 10,
    ) -> dict[str, list[Vector]]:
        """
        Finds vectors similar to the vectors identified by the given IDs.
        
        Parameters:
            vector_ids (list[str]): Identifiers of the vectors to find similarities for.
            brain_id (str): Identifier for the brain/context containing the vectors.
            store (str): Name of the vector store to query.
            min_similarity (float): Minimum similarity threshold (e.g., 0.0–1.0) for returned results.
            limit (int): Maximum number of similar vectors to return per input ID.
        
        Returns:
            dict[str, list[Vector]]: Mapping from each input vector ID to a list of similar `Vector` objects that meet or exceed `min_similarity`, with up to `limit` results per ID.
        """
        raise NotImplementedError("search_similar_by_ids method not implemented")
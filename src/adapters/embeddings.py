"""
File: /embeddings.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:43:59 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import uuid
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from src.adapters.interfaces.embeddings import EmbeddingsClient, VectorStoreClient
from src.constants.embeddings import Vector
from src.lib.embeddings.client import EmbeddingError


class EmbeddingsAdapter:
    def __init__(self):
        self.embeddings = None

    def add_client(self, client: EmbeddingsClient) -> None:
        """
        Add a embeddings client to the adapter.
        """
        self.embeddings = client

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(EmbeddingError),
    )
    def _embed_text_with_retry(self, text: str) -> list[float]:
        return self.embeddings.embed_text(text)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(EmbeddingError),
    )
    def _embed_texts_with_retry(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_texts(texts)

    def embed_text(self, text: str) -> Vector:
        """
        Embed a text and return a vector.
        """
        try:
            embeddings = self._embed_text_with_retry(text)
            return Vector(id=str(uuid.uuid4()), embeddings=embeddings, metadata={})
        except EmbeddingError as e:
            print(f"Embedding failed in adapter, returning empty vector: {e}")
            return Vector(id=str(uuid.uuid4()), embeddings=[], metadata={})

    def embed_texts(self, texts: list[str]) -> list[Vector]:
        """
        Embed a list of texts and return a list of vectors.
        """
        try:
            embeddings_list = self._embed_texts_with_retry(texts)
            return [
                Vector(id=str(uuid.uuid4()), embeddings=embeddings, metadata={})
                for embeddings in embeddings_list
            ]
        except EmbeddingError as e:
            print(f"Embedding failed in adapter, returning empty vectors: {e}")
            return [
                Vector(id=str(uuid.uuid4()), embeddings=[], metadata={}) for _ in texts
            ]


class VectorStoreAdapter:
    def __init__(self):
        self.vector_store = None

    def add_client(self, client: VectorStoreClient) -> None:
        """
        Add a vector store client to the adapter.
        """
        self.vector_store = client

    def add_vectors(
        self, vectors: list[Vector], store: str, brain_id: str = "default"
    ) -> list[str]:
        """
        Add vectors to the vector store.
        """
        return self.vector_store.add_vectors(vectors, store, brain_id)

    def search_vectors(
        self,
        data_vector: list[float],
        brain_id: str = "default",
        store: str = "default",
        k: int = 10,
    ) -> list[Vector]:
        """
        Search vectors in the vector store and return the top k vectors.
        """
        vectors = self.vector_store.search_vectors(data_vector, brain_id, store, k)
        return sorted(vectors, key=lambda x: x.distance, reverse=True)

    def get_by_ids(
        self, ids: list[str], store: str, brain_id: str = "default"
    ) -> list[Vector]:
        """
        Get vectors by their IDs.
        """
        return self.vector_store.get_by_ids(ids, store, brain_id)

    def search_similar_by_ids(
        self,
        vector_ids: list[str],
        brain_id: str,
        store: str,
        min_similarity: float,
        limit: int = 10,
    ) -> dict[str, list[Vector]]:
        """
        Finds vectors similar to the provided vector IDs within a specified store and brain.

        Parameters:
            vector_ids (list[str]): IDs of the source vectors to find similarities for.
            brain_id (str): Identifier of the brain/namespace to search within.
            store (str): Name of the vector store to query.
            min_similarity (float): Minimum similarity threshold for returned vectors (inclusive).
            limit (int): Maximum number of similar vectors to return per source ID.

        Returns:
            dict[str, list[Vector]]: Mapping from each source vector ID to a list of similar Vectors.
            Each list contains at most `limit` items, includes only vectors with similarity >= `min_similarity`,
            and is ordered by descending similarity.
        """
        return self.vector_store.search_similar_by_ids(
            vector_ids, brain_id, store, min_similarity, limit
        )

    def remove_vectors(
        self, ids: list[str], store: str, brain_id: str = "default"
    ) -> None:
        """
        Remove vectors from the vector store.
        """
        return self.vector_store.remove_vectors(ids, store, brain_id)


_embeddings_adapter = EmbeddingsAdapter()
_vector_store_adapter = VectorStoreAdapter()

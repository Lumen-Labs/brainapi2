"""
File: /embeddings.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 5th 2026 9:57:30 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import uuid
from src.adapters.interfaces.embeddings import EmbeddingsClient, VectorStoreClient
from src.constants.embeddings import Vector


class EmbeddingsAdapter:
    def __init__(self):
        self.embeddings = None

    def add_client(self, client: EmbeddingsClient) -> None:
        """
        Add a embeddings client to the adapter.
        """
        self.embeddings = client

    def embed_text(self, text: str) -> Vector:
        """
        Embed a text and return a vector.
        """
        from src.lib.embeddings.client import EmbeddingError

        try:
            embeddings = self.embeddings.embed_text(text)
            return Vector(id=str(uuid.uuid4()), embeddings=embeddings, metadata={})
        except EmbeddingError as e:
            print(f"Embedding failed in adapter, returning empty vector: {e}")
            return Vector(id=str(uuid.uuid4()), embeddings=[], metadata={})


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
        Search similar vectors by their IDs.
        """
        return self.vector_store.search_similar_by_ids(
            vector_ids, brain_id, store, min_similarity, limit
        )


_embeddings_adapter = EmbeddingsAdapter()
_vector_store_adapter = VectorStoreAdapter()

"""
File: /embeddings.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 9:00:59 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
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
        embeddings = self.embeddings.embed_text(text)
        return Vector(id=str(uuid.uuid4()), embeddings=embeddings, metadata={})


class VectorStoreAdapter:
    def __init__(self):
        self.vector_store = None

    def add_client(self, client: VectorStoreClient) -> None:
        """
        Add a vector store client to the adapter.
        """
        self.vector_store = client

    def add_vectors(self, vectors: list[Vector], store: str) -> None:
        """
        Add vectors to the vector store.
        """
        return self.vector_store.add_vectors(vectors, store)

    def search_vectors(self, query: str, store: str, k: int = 10) -> list[Vector]:
        """
        Search vectors in the vector store and return the top k vectors.
        """
        return self.vector_store.search_vectors(query, store, k)

    def get_by_ids(self, ids: list[str], store: str) -> list[Vector]:
        """
        Get vectors by their IDs.
        """
        return self.vector_store.get_by_ids(ids, store)

    def search_similar_by_ids(
        self, vector_ids: list[str], store: str, min_similarity: float, limit: int = 10
    ) -> list[Vector]:
        """
        Search similar vectors by their IDs.
        """
        return self.vector_store.search_similar_by_ids(
            vector_ids, store, min_similarity, limit
        )


_embeddings_adapter = EmbeddingsAdapter()
_vector_store_adapter = VectorStoreAdapter()

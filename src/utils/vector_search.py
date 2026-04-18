from src.adapters.embeddings import VectorStoreAdapter
from src.constants.embeddings import Vector


class VectorSearchFacade:
    def __init__(self, vector_store: VectorStoreAdapter):
        self._vector_store = vector_store

    def search(
        self,
        data_vector: list[float],
        *,
        store: str,
        brain_id: str = "default",
        k: int = 10,
    ) -> list[Vector]:
        return self._vector_store.search_vectors(
            data_vector,
            store=store,
            brain_id=brain_id,
            k=k,
        )

    def search_nodes(
        self,
        data_vector: list[float],
        *,
        brain_id: str = "default",
        k: int = 10,
    ) -> list[Vector]:
        return self.search(data_vector, store="nodes", brain_id=brain_id, k=k)

    def search_triplets(
        self,
        data_vector: list[float],
        *,
        brain_id: str = "default",
        k: int = 10,
    ) -> list[Vector]:
        return self.search(data_vector, store="triplets", brain_id=brain_id, k=k)

    def search_relationships(
        self,
        data_vector: list[float],
        *,
        brain_id: str = "default",
        k: int = 10,
    ) -> list[Vector]:
        return self.search(data_vector, store="relationships", brain_id=brain_id, k=k)

    def search_data(
        self,
        data_vector: list[float],
        *,
        brain_id: str = "default",
        k: int = 10,
    ) -> list[Vector]:
        return self.search(data_vector, store="data", brain_id=brain_id, k=k)

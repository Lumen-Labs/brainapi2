"""
File: /client.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 8:41:10 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import os
import hashlib

os.environ["GRPC_DNS_RESOLVER"] = "native"

if os.getenv("MILVUS_URI") and not os.getenv("MILVUS_URI").startswith(
    ("http://", "https://")
):
    os.environ["MILVUS_URI"] = f"https://{os.getenv('MILVUS_URI')}"

from pymilvus import MilvusClient as Milvus

from src.adapters.interfaces.embeddings import VectorStoreClient
from src.config import config
from src.constants.embeddings import EMBEDDING_STORES_SIZES, Vector


def string_to_int64(s: str) -> int:
    sha256 = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(sha256[:8], byteorder="big") % (2**63)


class MilvusClient(VectorStoreClient):
    """
    Milvus client.
    """

    def __init__(self):
        self._client = None
        self._lock = None

    def _get_lock(self):
        if self._lock is None:
            import threading

            self._lock = threading.Lock()
        return self._lock

    @property
    def client(self):
        """
        Lazy initialization of the Milvus client.
        """
        if self._client is None:
            with self._get_lock():
                if self._client is None:
                    if config.milvus.uri and config.milvus.token:
                        print("using uri", config.milvus.uri)
                        self._client = Milvus(
                            uri=config.milvus.uri,
                            token=config.milvus.token,
                        )
                    elif config.milvus.host and config.milvus.port:
                        self._client = Milvus(
                            host=config.milvus.host,
                            port=config.milvus.port,
                            token=config.milvus.token if config.milvus.token else None,
                        )
                    else:
                        raise ValueError("Invalid Milvus configuration")
        return self._client

    def _ensure_store(self, store: str) -> None:
        """
        Ensure the store exists and is loaded.
        """
        if store not in EMBEDDING_STORES_SIZES:
            raise ValueError(f"Store {store} not available")

        if not self.client.has_collection(store):
            self.client.create_collection(
                store,
                dimension=EMBEDDING_STORES_SIZES[store],
                vector_field_name="embeddings",
            )

        try:
            has_partition = self.client.has_partition(store, "_default")
            if not has_partition:
                self.client.load_collection(store)
        except Exception:
            try:
                self.client.load_collection(store)
            except Exception:
                pass

    def add_vectors(self, vectors: list[Vector], store: str) -> None:
        """
        Add vectors to the vector store.
        """
        self._ensure_store(store)
        self.client.insert(
            store,
            [
                {
                    "id": hash(vector.id) % (2**63),
                    "embeddings": vector.embeddings,
                    "uuid": vector.id,
                    **(vector.metadata or {}),
                }
                for vector in vectors
            ],
        )

    def search_vectors(
        self, data_vector: list[float], store: str, k: int = 10
    ) -> list[Vector]:
        """
        Search vectors in the vector store and return the top k vectors.
        """
        self._ensure_store(store)

        try:
            self.client.load_collection(store)
        except Exception:
            pass

        _results = self.client.search(
            store, data=[data_vector], limit=k, output_fields=["$meta"]
        )
        results = []
        for query_results in _results:
            for result in query_results:
                results.append(
                    Vector(
                        id=str(result["id"]),
                        metadata={
                            k: v
                            for k, v in result.get("entity").items()
                            if k not in ["id", "embeddings", "distance"]
                        },
                        distance=result["distance"],
                    )
                )
        return results

    def get_by_ids(self, ids: list[str], store: str) -> list[Vector]:
        """
        Get vectors by their IDs.
        """
        self._ensure_store(store)
        collection = self.client.query(
            collection_name=store,
            ids=[hash(id) % (2**63) for id in ids],
            output_fields=["$meta"],
        )
        return [
            Vector(id=result["id"], metadata=result["entity"]) for result in collection
        ]


_milvus_client = MilvusClient()

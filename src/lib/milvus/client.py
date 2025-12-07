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
        self._pid = None

    def _get_lock(self):
        if self._lock is None:
            import threading

            self._lock = threading.Lock()
        return self._lock

    def _reset_client_if_forked(self):
        import os as os_module

        current_pid = os_module.getpid()
        if self._pid is not None and self._pid != current_pid:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._client = None
        self._pid = current_pid

    @property
    def client(self):
        """
        Lazy initialization of the Milvus client.
        """
        self._reset_client_if_forked()
        if self._client is None:
            with self._get_lock():
                self._reset_client_if_forked()
                if self._client is None:
                    if config.milvus.uri and config.milvus.token:
                        self._client = Milvus(
                            uri=config.milvus.uri,
                            token=config.milvus.token,
                        )
                    elif config.milvus.host and config.milvus.port:
                        uri = f"http://{config.milvus.host}:{config.milvus.port}"
                        self._client = Milvus(
                            uri=uri,
                            token=config.milvus.token if config.milvus.token else None,
                        )
                    else:
                        raise ValueError("Invalid Milvus configuration")
        return self._client

    def _ensure_store(self, store: str, brain_id: str) -> None:
        """
        Ensure the store exists and is loaded.
        """
        if store not in EMBEDDING_STORES_SIZES:
            raise ValueError(f"Store {store} not available")

        collection_created = False
        if not self.client.has_collection(store):
            self.client.create_collection(
                store,
                dimension=EMBEDDING_STORES_SIZES[store],
                vector_field_name="embeddings",
            )
            collection_created = True

        if collection_created:
            try:
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "AUTOINDEX",
                }
                self.client.create_index(
                    collection_name=store,
                    field_name="embeddings",
                    index_params=index_params,
                )
            except Exception as e:
                pass

        try:
            has_partition = self.client.has_partition(store, brain_id)
            if not has_partition:
                self.client.load_collection(store)
        except Exception:
            try:
                self.client.load_collection(store)
            except Exception:
                pass

    def add_vectors(self, vectors: list[Vector], store: str, brain_id: str) -> None:
        """
        Add vectors to the vector store.
        """
        self._ensure_store(store, brain_id)
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
        self, data_vector: list[float], store: str, brain_id: str, k: int = 10
    ) -> list[Vector]:
        """
        Search vectors in the vector store and return the top k vectors.
        """
        self._ensure_store(store, brain_id)

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

    def get_by_ids(self, ids: list[str], store: str, brain_id: str) -> list[Vector]:
        """
        Get vectors by their IDs.
        """
        self._ensure_store(store, brain_id)
        collection = self.client.query(
            collection_name=store,
            ids=[hash(id) % (2**63) for id in ids],
            output_fields=["$meta"],
        )
        return [
            Vector(id=result["id"], metadata=result["entity"]) for result in collection
        ]

    def search_similar_by_ids(
        self,
        vector_ids: list[str],
        brain_id: str,
        store: str,
        min_similarity: float,
        limit: int = 10,
    ) -> dict[str, list[Vector]]:
        self._ensure_store(store, brain_id)
        try:
            self.client.load_collection(store)
        except Exception:
            pass

        if not vector_ids:
            return {}

        expr = "uuid in [" + ",".join(f"'{v}'" for v in vector_ids) + "]"
        queried = self.client.query(
            collection_name=store,
            filter=expr,
            output_fields=["embeddings", "$meta"],
        )
        id_to_emb: dict[str, list[float]] = {}

        for item in queried or []:
            emb = item.get("embeddings") or item.get("embeddings")
            uuid = item.get("uuid")
            if emb is not None and uuid:
                id_to_emb[uuid] = emb

        data_batch = [id_to_emb.get(v_id) for v_id in vector_ids]
        if not any(data_batch):
            return {v_id: [] for v_id in vector_ids}

        batch_indices = [i for i, emb in enumerate(data_batch) if emb is not None]
        batch_embeddings = [data_batch[i] for i in batch_indices]
        batch_ids = [vector_ids[i] for i in batch_indices]

        out: dict[str, list[Vector]] = {v_id: [] for v_id in vector_ids}
        chunk_size = 10
        for start in range(0, len(batch_embeddings), chunk_size):
            end = start + chunk_size
            chunk_embeddings = batch_embeddings[start:end]
            chunk_ids = batch_ids[start:end]

            raw_results = self.client.search(
                collection_name=store,
                data=chunk_embeddings,
                limit=max(limit * 5, limit),
                output_fields=["$meta"],
            )

            for idx, query_results in enumerate(raw_results or []):
                origin_uuid = chunk_ids[idx]
                if out.get(origin_uuid):
                    continue

                collected: list[Vector] = []
                for r in query_results or []:
                    r_entity = r.get("entity", {})
                    if r_entity.get("uuid") == origin_uuid:
                        continue
                    sim = 1.0 - float(r.get("distance", 0.0))
                    if sim >= min_similarity:
                        collected.append(
                            Vector(
                                id=str(r["id"]),
                                metadata={
                                    k: v
                                    for k, v in r.get("entity", {}).items()
                                    if k not in ["id", "embeddings", "distance"]
                                },
                                distance=r["distance"],
                            )
                        )
                        if len(collected) >= limit:
                            break
                out[origin_uuid] = collected

        return out


_milvus_client = MilvusClient()

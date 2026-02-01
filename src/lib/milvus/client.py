"""
File: /client.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 5th 2026 9:57:30 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
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
from pymilvus import connections, db
from pymilvus.milvus_client.index import IndexParams

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
        self._clients = {}
        self._default_client = None
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
        if hasattr(self, "_pid") and self._pid is not None and self._pid != current_pid:
            for brain_id in list(self._clients.keys()):
                try:
                    self._clients[brain_id].close()
                except Exception:
                    pass
                del self._clients[brain_id]
            if self._default_client is not None:
                try:
                    self._default_client.close()
                except Exception:
                    pass
                self._default_client = None
        self._pid = current_pid

    def _get_default_client(self):
        """
        Get or create a default Milvus client (without db_name) for database management operations.
        """
        self._reset_client_if_forked()
        if self._default_client is None:
            with self._get_lock():
                self._reset_client_if_forked()
                if self._default_client is None:
                    if config.milvus.uri and config.milvus.token:
                        self._default_client = Milvus(
                            uri=config.milvus.uri,
                            token=config.milvus.token,
                            timeout=30,
                        )
                    elif config.milvus.host and config.milvus.port:
                        uri = f"http://{config.milvus.host}:{config.milvus.port}"
                        self._default_client = Milvus(
                            uri=uri,
                            token=config.milvus.token if config.milvus.token else None,
                            timeout=30,
                        )
                    else:
                        raise ValueError("Invalid Milvus configuration")
        return self._default_client

    def _ensure_database(self, brain_id: str):
        """
        Ensure the database exists, creating it if necessary.
        """
        try:
            if config.milvus.uri and config.milvus.token:
                connections.connect(
                    uri=config.milvus.uri,
                    token=config.milvus.token,
                    timeout=30,
                )
            elif config.milvus.host and config.milvus.port:
                uri = f"http://{config.milvus.host}:{config.milvus.port}"
                connections.connect(
                    uri=uri,
                    token=config.milvus.token if config.milvus.token else None,
                    timeout=30,
                )
            else:
                raise ValueError("Invalid Milvus configuration")

            databases = db.list_database()
            if brain_id not in databases:
                db.create_database(brain_id)
        except Exception as e:
            error_msg = str(e).lower()
            if (
                "already exists" not in error_msg
                and "database not found" not in error_msg
                and "already connected" not in error_msg
            ):
                raise

    def _get_client(self, brain_id: str):
        """
        Get or create a Milvus client for the specified database (brain_id).
        """
        self._reset_client_if_forked()
        if brain_id not in self._clients:
            with self._get_lock():
                self._reset_client_if_forked()
                if brain_id not in self._clients:
                    self._ensure_database(brain_id)
                    if config.milvus.uri and config.milvus.token:
                        self._clients[brain_id] = Milvus(
                            uri=config.milvus.uri,
                            token=config.milvus.token,
                            db_name=brain_id,
                        )
                    elif config.milvus.host and config.milvus.port:
                        uri = f"http://{config.milvus.host}:{config.milvus.port}"
                        self._clients[brain_id] = Milvus(
                            uri=uri,
                            token=config.milvus.token if config.milvus.token else None,
                            db_name=brain_id,
                        )
                    else:
                        raise ValueError("Invalid Milvus configuration")
        return self._clients[brain_id]

    def _ensure_store(self, store: str, brain_id: str) -> None:
        """
        Ensure the store exists and is loaded in the specified database.
        """
        client = self._get_client(brain_id)
        if store not in EMBEDDING_STORES_SIZES:
            raise ValueError(f"Store {store} not available")

        collection_created = False
        if not client.has_collection(store):
            client.create_collection(
                store,
                dimension=EMBEDDING_STORES_SIZES[store],
                vector_field_name="embeddings",
            )
            collection_created = True

        if collection_created:
            try:
                index_params = IndexParams()
                index_params.add_index(
                    field_name="embeddings",
                    index_type="AUTOINDEX",
                    metric_type="COSINE",
                )
                client.create_index(
                    collection_name=store,
                    index_params=index_params,
                )
            except Exception as e:
                print(f"[Milvus] Error creating index for collection {store}: {e}")

        try:
            client.load_collection(store)
        except Exception:
            pass

    def add_vectors(
        self, vectors: list[Vector], store: str, brain_id: str
    ) -> list[str]:
        """
        Add vectors to the vector store.
        """
        client = self._get_client(brain_id)
        self._ensure_store(store, brain_id)
        vs = [
            {
                "id": hash(vector.id) % (2**63),
                "embeddings": vector.embeddings,
                "uuid": vector.id,
                **(vector.metadata or {}),
            }
            for vector in vectors
        ]
        client.insert(
            store,
            vs,
        )
        return [v["id"] for v in vs]

    def search_vectors(
        self, data_vector: list[float], brain_id: str, store: str, k: int = 10
    ) -> list[Vector]:
        """
        Search vectors in the vector store and return the top k vectors.
        """
        client = self._get_client(brain_id)
        self._ensure_store(store, brain_id)

        try:
            client.load_collection(store)
        except Exception:
            pass

        _results = client.search(
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
        results.sort(key=lambda x: x.distance)
        return results

    def get_by_ids(self, ids: list[str], store: str, brain_id: str) -> list[Vector]:
        """
        Get vectors by their IDs from the specified database (brain_id).
        """
        if not ids:
            return []
        client = self._get_client(brain_id)
        self._ensure_store(store, brain_id)
        collection = client.query(
            collection_name=store,
            ids=ids,
            output_fields=["embeddings", "$meta"],
        )
        return [
            Vector(
                id=result.get("uuid", str(result["id"])),
                embeddings=result.get("embeddings"),
                metadata={
                    k: v
                    for k, v in result.items()
                    if k not in ["id", "embeddings", "distance"]
                },
            )
            for result in collection
        ]

    def search_similar_by_ids(
        self,
        vector_ids: list[str],
        brain_id: str,
        store: str,
        min_similarity: float,
        limit: int = 10,
    ) -> dict[str, list[Vector]]:
        client = self._get_client(brain_id)
        self._ensure_store(store, brain_id)
        try:
            client.load_collection(store)
        except Exception:
            pass

        if not vector_ids:
            return {}

        expr = "id in [" + ",".join(str(v) for v in vector_ids) + "]"
        queried = client.query(
            collection_name=store,
            filter=expr,
            output_fields=["embeddings", "$meta"],
        )
        id_to_emb: dict[str, list[float]] = {}

        for item in queried or []:
            emb = item.get("embeddings") or item.get("embeddings")
            uuid = item.get("id")
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

            raw_results = client.search(
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

    def remove_vectors(self, ids: list[str], store: str, brain_id: str) -> None:
        """
        Remove vectors from the vector store.
        """
        if not ids:
            return None
        client = self._get_client(brain_id)
        self._ensure_store(store, brain_id)
        try:
            client.delete(store, ids=ids)
        except Exception:
            pass
        return None


_milvus_client = MilvusClient()

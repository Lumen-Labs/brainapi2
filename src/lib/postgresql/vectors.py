"""
File: /vectors.py
Project: postgresql
Created Date: Sunday May 24th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import logging
import re
import threading
from typing import Any, Iterator, Optional

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from src.adapters.interfaces.embeddings import VectorStoreClient
from src.constants.embeddings import EMBEDDING_STORES_SIZES, Vector

from ._provisioning import borrow, ensure_brain_database, get_brain_pool


logger = logging.getLogger(__name__)


_STORE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def string_to_int64(s: str) -> int:
    sha256 = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(sha256[:8], byteorder="big") % (2**63)


def _to_int_id(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return string_to_int64(value)
    return string_to_int64(str(value))


def _safe_store(store: str) -> str:
    if not _STORE_NAME_RE.match(store):
        raise ValueError(f"Invalid store name: {store}")
    return store


def _table_name(store: str) -> str:
    return f"vectors_{_safe_store(store)}"


def _vector_index_ddl(table: str, dimension: int) -> str:
    dim = int(dimension)
    if dim > 2000:
        return ""
    return f"""
            CREATE INDEX IF NOT EXISTS idx_{table}_embeddings
                ON {table} USING hnsw (embeddings vector_cosine_ops);
            """


def _table_vector_dimension(cur: psycopg2.extensions.cursor, table: str) -> Optional[int]:
    cur.execute(
        """
        SELECT a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE c.relname = %s
          AND a.attname = 'embeddings'
          AND NOT a.attisdropped
          AND n.nspname = current_schema()
        """,
        (table,),
    )
    row = cur.fetchone()
    if not row or row[0] is None or row[0] < 1:
        return None
    return int(row[0])


class PostgreSQLVectorStoreClient(VectorStoreClient):
    """
    Vector store backed by PostgreSQL with the pgvector extension.

    Each brain owns its own database (`brain_<sanitized_id>`); per-store tables
    live inside that database with no `brain_id` column. The `vector` extension
    is enabled lazily, once per brain database.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._brain_extensions_ready: set[str] = set()
        self._initialized_stores: set[str] = set()

    @contextmanager
    def _connection(self, brain_id: str) -> Iterator[psycopg2.extensions.connection]:
        self._ensure_extension(brain_id)
        with borrow(get_brain_pool(brain_id)) as conn:
            register_vector(conn)
            yield conn

    def _ensure_extension(self, brain_id: str) -> None:
        if brain_id in self._brain_extensions_ready:
            return
        with self._lock:
            if brain_id in self._brain_extensions_ready:
                return
            ensure_brain_database(brain_id)
            with borrow(get_brain_pool(brain_id)) as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
            self._brain_extensions_ready.add(brain_id)

    def _ensure_store(self, store: str, brain_id: str) -> None:
        if store not in EMBEDDING_STORES_SIZES:
            raise ValueError(f"Store {store} not available")
        key = f"{brain_id}:{store}"
        if key in self._initialized_stores:
            return
        with self._lock:
            if key in self._initialized_stores:
                return
            dimension = EMBEDDING_STORES_SIZES[store]
            table = _table_name(store)
            ddl = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id BIGINT PRIMARY KEY,
                uuid TEXT NOT NULL,
                embeddings vector({dimension}),
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
            );
            CREATE INDEX IF NOT EXISTS idx_{table}_uuid
                ON {table} (uuid);
            {_vector_index_ddl(table, dimension)}
            """
            with self._connection(brain_id) as conn:
                with conn.cursor() as cur:
                    existing_dim = _table_vector_dimension(cur, table)
                    if existing_dim is not None and existing_dim != dimension:
                        logger.warning(
                            "Recreating %s for brain %s: vector dimension %s -> %s",
                            table,
                            brain_id,
                            existing_dim,
                            dimension,
                        )
                        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                    cur.execute(ddl)
                conn.commit()
            self._initialized_stores.add(key)

    def add_vectors(
        self, vectors: list[Vector], store: str, brain_id: str
    ) -> list[str]:
        if not vectors:
            return []
        self._ensure_store(store, brain_id)
        table = _table_name(store)
        rows = []
        ids: list[int] = []
        for vector in vectors:
            int_id = string_to_int64(vector.id)
            ids.append(int_id)
            metadata = dict(vector.metadata or {})
            metadata.pop("id", None)
            metadata.pop("embeddings", None)
            metadata.pop("distance", None)
            rows.append(
                (
                    int_id,
                    vector.id,
                    vector.embeddings,
                    json.dumps(metadata, default=str),
                )
            )
        with self._connection(brain_id) as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    f"""
                    INSERT INTO {table}
                        (id, uuid, embeddings, metadata)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        uuid = EXCLUDED.uuid,
                        embeddings = EXCLUDED.embeddings,
                        metadata = EXCLUDED.metadata
                    """,
                    rows,
                    template="(%s, %s, %s, %s::jsonb)",
                )
            conn.commit()
        return ids

    def search_vectors(
        self, data_vector: list[float], brain_id: str, store: str, k: int = 10
    ) -> list[Vector]:
        self._ensure_store(store, brain_id)
        table = _table_name(store)
        with self._connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT id, uuid, metadata,
                           (embeddings <=> %s::vector) AS distance
                    FROM {table}
                    ORDER BY embeddings <=> %s::vector
                    LIMIT %s
                    """,
                    (data_vector, data_vector, k),
                )
                rows = cur.fetchall()
        results = [
            Vector(
                id=str(row["id"]),
                metadata=dict(row["metadata"] or {}),
                distance=float(row["distance"]),
            )
            for row in rows
        ]
        results.sort(key=lambda x: x.distance)
        return results

    def get_by_ids(self, ids: list[str], store: str, brain_id: str) -> list[Vector]:
        if not ids:
            return []
        self._ensure_store(store, brain_id)
        table = _table_name(store)
        int_ids = [_to_int_id(i) for i in ids]
        with self._connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT id, uuid, embeddings, metadata
                    FROM {table}
                    WHERE id = ANY(%s)
                    """,
                    (int_ids,),
                )
                rows = cur.fetchall()
        results: list[Vector] = []
        for row in rows:
            embeddings = row["embeddings"]
            if embeddings is not None and not isinstance(embeddings, list):
                embeddings = list(embeddings)
            results.append(
                Vector(
                    id=row["uuid"] or str(row["id"]),
                    embeddings=embeddings,
                    metadata=dict(row["metadata"] or {}),
                )
            )
        return results

    def search_similar_by_ids(
        self,
        vector_ids: list[str],
        brain_id: str,
        store: str,
        min_similarity: float,
        limit: int = 10,
    ) -> dict[str, list[Vector]]:
        if not vector_ids:
            return {}
        self._ensure_store(store, brain_id)
        table = _table_name(store)
        int_ids = [_to_int_id(i) for i in vector_ids]

        with self._connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT id, uuid, embeddings
                    FROM {table}
                    WHERE id = ANY(%s)
                    """,
                    (int_ids,),
                )
                origin_rows = cur.fetchall()
        embeddings_by_id: dict[int, list[float]] = {}
        for row in origin_rows:
            embedding = row["embeddings"]
            if embedding is None:
                continue
            if not isinstance(embedding, list):
                embedding = list(embedding)
            embeddings_by_id[row["id"]] = embedding

        out: dict[str, list[Vector]] = {str(v_id): [] for v_id in vector_ids}
        if not embeddings_by_id:
            return out

        fetch_limit = max(limit * 5, limit) + 1
        with self._connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                for origin_id, embedding in embeddings_by_id.items():
                    cur.execute(
                        f"""
                        SELECT id, uuid, metadata,
                               (embeddings <=> %s::vector) AS distance
                        FROM {table}
                        ORDER BY embeddings <=> %s::vector
                        LIMIT %s
                        """,
                        (embedding, embedding, fetch_limit),
                    )
                    rows = cur.fetchall()
                    collected: list[Vector] = []
                    for row in rows:
                        if row["id"] == origin_id:
                            continue
                        distance = float(row["distance"])
                        similarity = 1.0 - distance
                        if similarity < min_similarity:
                            continue
                        collected.append(
                            Vector(
                                id=str(row["id"]),
                                metadata=dict(row["metadata"] or {}),
                                distance=distance,
                            )
                        )
                        if len(collected) >= limit:
                            break
                    out[str(origin_id)] = collected
        return out

    def remove_vectors(self, ids: list[str], store: str, brain_id: str) -> None:
        if not ids:
            return
        self._ensure_store(store, brain_id)
        table = _table_name(store)
        int_ids = [_to_int_id(i) for i in ids]
        try:
            with self._connection(brain_id) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        DELETE FROM {table}
                        WHERE id = ANY(%s)
                        """,
                        (int_ids,),
                    )
                conn.commit()
        except Exception as exc:
            print(f"[PostgreSQLVectorStore] Failed to remove vectors from {store}: {exc}")

    def list_vectors(
        self,
        store: str,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        include_embeddings: bool = False,
    ) -> tuple[list[Vector], int]:
        self._ensure_store(store, brain_id)
        table = _table_name(store)
        embedding_col = ", embeddings" if include_embeddings else ""
        with self._connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"SELECT COUNT(*) AS total FROM {table}")
                total = int(cur.fetchone()["total"])
                cur.execute(
                    f"""
                    SELECT id, uuid, metadata{embedding_col}
                    FROM {table}
                    ORDER BY id
                    LIMIT %s OFFSET %s
                    """,
                    (limit, skip),
                )
                rows = cur.fetchall()
        results: list[Vector] = []
        for row in rows:
            embeddings = None
            if include_embeddings:
                embeddings = row.get("embeddings")
                if embeddings is not None and not isinstance(embeddings, list):
                    embeddings = list(embeddings)
            results.append(
                Vector(
                    id=row["uuid"] or str(row["id"]),
                    embeddings=embeddings,
                    metadata=dict(row["metadata"] or {}),
                )
            )
        return results, total


_postgresql_vector_store_client: Optional[PostgreSQLVectorStoreClient] = None


def get_postgresql_vector_store_client() -> PostgreSQLVectorStoreClient:
    global _postgresql_vector_store_client
    if _postgresql_vector_store_client is None:
        _postgresql_vector_store_client = PostgreSQLVectorStoreClient()
    return _postgresql_vector_store_client

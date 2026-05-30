"""
File: /data.py
Project: postgresql
Created Date: Sunday May 24th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import threading
from typing import Any, Iterator, List, Optional, Tuple

import psycopg2
import psycopg2.extras

from src.adapters.interfaces.data import DataClient, SearchResult
from src.config import config
from src.constants.data import (
    Brain,
    KGChanges,
    KGChangesType,
    Observation,
    StructuredData,
    TextChunk,
)

from ._provisioning import (
    borrow,
    ensure_brain_database,
    ensure_database_exists,
    get_brain_pool,
    get_system_pool,
)


_SYSTEM_DDL = """
CREATE TABLE IF NOT EXISTS data_brains (
    id TEXT PRIMARY KEY,
    name_key TEXT UNIQUE NOT NULL,
    pat TEXT NOT NULL,
    document JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


_BRAIN_DDL = """
CREATE TABLE IF NOT EXISTS data_text_chunks (
    id TEXT PRIMARY KEY,
    text TEXT,
    metadata JSONB,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    document JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_data_text_chunks_inserted_at
    ON data_text_chunks (inserted_at DESC);

CREATE TABLE IF NOT EXISTS data_observations (
    id TEXT PRIMARY KEY,
    text TEXT,
    resource_id TEXT,
    metadata JSONB,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    document JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_data_observations_resource
    ON data_observations (resource_id);
CREATE INDEX IF NOT EXISTS idx_data_observations_labels
    ON data_observations USING gin ((metadata -> 'labels'));

CREATE TABLE IF NOT EXISTS data_structured_data (
    id TEXT PRIMARY KEY,
    data JSONB,
    types TEXT[],
    metadata JSONB,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    document JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_data_structured_data_types
    ON data_structured_data USING gin (types);
CREATE INDEX IF NOT EXISTS idx_data_structured_data_inserted_at
    ON data_structured_data (inserted_at DESC);

CREATE TABLE IF NOT EXISTS data_kg_changes (
    id TEXT PRIMARY KEY,
    type TEXT,
    change JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    document JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_data_kg_changes_timestamp
    ON data_kg_changes (timestamp DESC);
"""


def _ilike_pattern(query_text: str) -> str:
    escaped = query_text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _validate_changelog(document: dict) -> Optional[KGChanges]:
    if "change" in document and isinstance(document["change"], dict):
        change_type = document["change"].get("type")
        if isinstance(change_type, str):
            document["change"]["type"] = KGChangesType(change_type)
    if isinstance(document.get("type"), str):
        document["type"] = KGChangesType(document["type"])
    return KGChanges.model_validate(document)


class PostgreSQLDataClient(DataClient):
    """
    Data client backed by PostgreSQL with JSONB-stored documents.

    Mirrors the Mongo client's `database = brain_id` layout: a single registry
    database (``config.postgresql.system_database``) holds the ``data_brains``
    catalog, and every brain owns its own Postgres database (named
    ``brain_<sanitized_id>``) containing all of its tables.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._system_initialized = False
        self._initialized_brains: set[str] = set()

    @contextmanager
    def _system_connection(self) -> Iterator[psycopg2.extensions.connection]:
        self._ensure_system_schema()
        with borrow(get_system_pool()) as conn:
            yield conn

    @contextmanager
    def _brain_connection(
        self, brain_id: str
    ) -> Iterator[psycopg2.extensions.connection]:
        self._ensure_brain_schema(brain_id)
        with borrow(get_brain_pool(brain_id)) as conn:
            yield conn

    def _ensure_system_schema(self) -> None:
        if self._system_initialized:
            return
        with self._lock:
            if self._system_initialized:
                return
            ensure_database_exists(config.postgresql.system_database)
            with borrow(get_system_pool()) as conn:
                with conn.cursor() as cur:
                    cur.execute(_SYSTEM_DDL)
                conn.commit()
            self._system_initialized = True

    def _ensure_brain_schema(self, brain_id: str) -> None:
        if brain_id in self._initialized_brains:
            return
        with self._lock:
            if brain_id in self._initialized_brains:
                return
            ensure_brain_database(brain_id)
            with borrow(get_brain_pool(brain_id)) as conn:
                with conn.cursor() as cur:
                    cur.execute(_BRAIN_DDL)
                conn.commit()
            self._initialized_brains.add(brain_id)

    def save_text_chunk(self, text_chunk: TextChunk, brain_id: str) -> TextChunk:
        document = text_chunk.model_dump(mode="json")
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO data_text_chunks
                        (id, text, metadata, inserted_at, document)
                    VALUES (%s, %s, %s::jsonb, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        inserted_at = EXCLUDED.inserted_at,
                        document = EXCLUDED.document
                    """,
                    (
                        text_chunk.id,
                        text_chunk.text,
                        json.dumps(text_chunk.metadata or {}, default=str),
                        text_chunk.inserted_at,
                        json.dumps(document, default=str),
                    ),
                )
            conn.commit()
        return text_chunk

    def save_observations(
        self, observations: List[Observation], brain_id: str
    ) -> List[Observation]:
        if not observations:
            return observations
        rows = [
            (
                observation.id,
                observation.text,
                observation.resource_id,
                json.dumps(observation.metadata or {}, default=str),
                observation.inserted_at,
                json.dumps(observation.model_dump(mode="json"), default=str),
            )
            for observation in observations
        ]
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO data_observations
                        (id, text, resource_id, metadata, inserted_at, document)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        resource_id = EXCLUDED.resource_id,
                        metadata = EXCLUDED.metadata,
                        inserted_at = EXCLUDED.inserted_at,
                        document = EXCLUDED.document
                    """,
                    rows,
                    template="(%s, %s, %s, %s::jsonb, %s, %s::jsonb)",
                )
            conn.commit()
        return observations

    def search(
        self,
        text: str,
        brain_id: str,
        collection: str = "text_chunks",
        limit: int = 10,
    ) -> SearchResult:
        table = (
            "data_text_chunks"
            if collection in ("*", "text_chunks")
            else f"data_{collection}"
        )
        pattern = _ilike_pattern(text)
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT document
                    FROM {table}
                    WHERE text ILIKE %s
                    LIMIT %s
                    """,
                    (pattern, limit),
                )
                text_rows = cur.fetchall()
                text_chunks = [
                    TextChunk.model_validate(row["document"]) for row in text_rows
                ]
                ids = [chunk.id for chunk in text_chunks]
                observations: list[Observation] = []
                if ids:
                    cur.execute(
                        """
                        SELECT document
                        FROM data_observations
                        WHERE resource_id = ANY(%s)
                        """,
                        (ids,),
                    )
                    observations = [
                        Observation.model_validate(row["document"])
                        for row in cur.fetchall()
                    ]
        return SearchResult(text_chunks=text_chunks, observations=observations)

    def get_text_chunks_by_ids(
        self, ids: List[str], with_observations: bool = False, brain_id: str = "default"
    ) -> Tuple[List[TextChunk], List[Observation]]:
        if not ids:
            return ([], [])
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT document FROM data_text_chunks
                    WHERE id = ANY(%s)
                    """,
                    (ids,),
                )
                text_chunks = [
                    TextChunk.model_validate(row["document"]) for row in cur.fetchall()
                ]
                observations: list[Observation] = []
                if with_observations:
                    cur.execute(
                        """
                        SELECT document FROM data_observations
                        WHERE resource_id = ANY(%s)
                        """,
                        (ids,),
                    )
                    observations = [
                        Observation.model_validate(row["document"])
                        for row in cur.fetchall()
                    ]
        return (text_chunks, observations)

    def save_structured_data(
        self, structured_data: StructuredData, brain_id: str
    ) -> StructuredData:
        document = structured_data.model_dump(mode="json")
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO data_structured_data
                        (id, data, types, metadata, inserted_at, document)
                    VALUES (%s, %s::jsonb, %s, %s::jsonb, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        data = EXCLUDED.data,
                        types = EXCLUDED.types,
                        metadata = EXCLUDED.metadata,
                        inserted_at = EXCLUDED.inserted_at,
                        document = EXCLUDED.document
                    """,
                    (
                        structured_data.id,
                        json.dumps(structured_data.data or {}, default=str),
                        list(structured_data.types or []),
                        json.dumps(structured_data.metadata or {}, default=str),
                        structured_data.inserted_at,
                        json.dumps(document, default=str),
                    ),
                )
            conn.commit()
        return structured_data

    def create_brain(self, name_key: str) -> Brain:
        brain = Brain(name_key=name_key)
        document = brain.model_dump(mode="json")
        with self._system_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO data_brains (id, name_key, pat, document)
                    VALUES (%s, %s, %s, %s::jsonb)
                    """,
                    (
                        brain.id,
                        brain.name_key,
                        brain.pat,
                        json.dumps(document, default=str),
                    ),
                )
            conn.commit()
        self._ensure_brain_schema(brain.id)
        return brain

    def get_brain(self, name_key: str) -> Optional[Brain]:
        with self._system_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT document FROM data_brains WHERE name_key = %s",
                    (name_key,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return Brain.model_validate(row["document"])

    def get_brains_list(self) -> List[Brain]:
        with self._system_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT document FROM data_brains")
                rows = cur.fetchall()
        return [Brain.model_validate(row["document"]) for row in rows]

    def save_kg_changes(self, kg_changes: KGChanges, brain_id: str) -> KGChanges:
        document = kg_changes.model_dump(mode="json")
        type_value = (
            kg_changes.type.value
            if isinstance(kg_changes.type, KGChangesType)
            else kg_changes.type
        )
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO data_kg_changes
                        (id, type, change, timestamp, document)
                    VALUES (%s, %s, %s::jsonb, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        type = EXCLUDED.type,
                        change = EXCLUDED.change,
                        timestamp = EXCLUDED.timestamp,
                        document = EXCLUDED.document
                    """,
                    (
                        kg_changes.id,
                        type_value,
                        json.dumps(document.get("change") or {}, default=str),
                        kg_changes.timestamp,
                        json.dumps(document, default=str),
                    ),
                )
            conn.commit()
        return kg_changes

    def get_structured_data_by_id(
        self, id: str, brain_id: str
    ) -> Optional[StructuredData]:
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT document FROM data_structured_data
                    WHERE id = %s
                    """,
                    (id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return StructuredData.model_validate(row["document"])

    def get_structured_data_list(
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        types: list[str] = None,
        query_text: str = None,
    ) -> Tuple[list[StructuredData], int]:
        clauses: list[str] = []
        params: list[Any] = []
        if types:
            clauses.append("types && %s")
            params.append(list(types))
        if query_text:
            pattern = _ilike_pattern(query_text)
            clauses.append(
                "(data::text ILIKE %s OR metadata::text ILIKE %s "
                "OR EXISTS (SELECT 1 FROM unnest(types) t WHERE t ILIKE %s))"
            )
            params.extend([pattern, pattern, pattern])
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT document FROM data_structured_data
                    {where_sql}
                    ORDER BY inserted_at DESC
                    OFFSET %s LIMIT %s
                    """,
                    (*params, skip, limit),
                )
                rows = cur.fetchall()
                cur.execute(
                    f"SELECT COUNT(*) AS total FROM data_structured_data {where_sql}",
                    tuple(params),
                )
                total = cur.fetchone()["total"]
        return (
            [StructuredData.model_validate(row["document"]) for row in rows],
            int(total or 0),
        )

    def get_text_chunks(
        self, brain_id: str, limit: int = 10, skip: int = 0, query_text: str = None
    ) -> Tuple[List[TextChunk], int]:
        clauses: list[str] = []
        params: list[Any] = []
        if query_text:
            pattern = _ilike_pattern(query_text)
            clauses.append("(text ILIKE %s OR metadata::text ILIKE %s)")
            params.extend([pattern, pattern])
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT document FROM data_text_chunks
                    {where_sql}
                    ORDER BY inserted_at DESC
                    OFFSET %s LIMIT %s
                    """,
                    (*params, skip, limit),
                )
                rows = cur.fetchall()
                cur.execute(
                    f"SELECT COUNT(*) AS total FROM data_text_chunks {where_sql}",
                    tuple(params),
                )
                total = cur.fetchone()["total"]
        return (
            [TextChunk.model_validate(row["document"]) for row in rows],
            int(total or 0),
        )

    def get_structured_data_types(self, brain_id: str) -> list[str]:
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT unnest(types) AS t
                    FROM data_structured_data
                    WHERE types IS NOT NULL
                    ORDER BY t
                    """
                )
                return [row[0] for row in cur.fetchall() if row[0] is not None]

    def get_observation_by_id(self, id: str, brain_id: str) -> Optional[Observation]:
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT document FROM data_observations
                    WHERE id = %s
                    """,
                    (id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return Observation.model_validate(row["document"])

    def get_observations_list(
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        resource_id: str = None,
        labels: list[str] = None,
        query_text: str = None,
    ) -> list[Observation]:
        clauses: list[str] = []
        params: list[Any] = []
        if resource_id:
            clauses.append("resource_id = %s")
            params.append(resource_id)
        if labels:
            clauses.append("metadata -> 'labels' ?| %s")
            params.append(list(labels))
        if query_text:
            pattern = _ilike_pattern(query_text)
            clauses.append(
                "(text ILIKE %s OR resource_id ILIKE %s OR metadata::text ILIKE %s)"
            )
            params.extend([pattern, pattern, pattern])
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT document FROM data_observations
                    {where_sql}
                    ORDER BY inserted_at DESC
                    OFFSET %s LIMIT %s
                    """,
                    (*params, skip, limit),
                )
                rows = cur.fetchall()
        return [Observation.model_validate(row["document"]) for row in rows]

    def get_observation_labels(self, brain_id: str) -> list[str]:
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT jsonb_array_elements_text(metadata -> 'labels') AS label
                    FROM data_observations
                    WHERE metadata ? 'labels'
                      AND jsonb_typeof(metadata -> 'labels') = 'array'
                    ORDER BY label
                    """
                )
                return [row[0] for row in cur.fetchall() if row[0] is not None]

    def get_changelog_by_id(self, id: str, brain_id: str) -> Optional[KGChanges]:
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT document FROM data_kg_changes
                    WHERE id = %s
                    """,
                    (id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return _validate_changelog(dict(row["document"]))

    def get_changelogs_list(
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        types: list[str] = None,
        query_text: str = None,
    ) -> list[KGChanges]:
        clauses: list[str] = []
        params: list[Any] = []
        if types:
            clauses.append("type = ANY(%s)")
            params.append(list(types))
        if query_text:
            pattern = _ilike_pattern(query_text)
            clauses.append("(change::text ILIKE %s OR type ILIKE %s)")
            params.extend([pattern, pattern])
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT document FROM data_kg_changes
                    {where_sql}
                    ORDER BY timestamp DESC
                    OFFSET %s LIMIT %s
                    """,
                    (*params, skip, limit),
                )
                rows = cur.fetchall()
        changelogs: list[KGChanges] = []
        for row in rows:
            try:
                changelogs.append(_validate_changelog(dict(row["document"])))
            except Exception as exc:
                print(f"Error parsing changelog {row['document'].get('id')}: {exc}")
                continue
        return changelogs

    def get_changelog_types(self, brain_id: str) -> list[str]:
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT type FROM data_kg_changes
                    WHERE type IS NOT NULL
                    ORDER BY type
                    """
                )
                return [row[0] for row in cur.fetchall()]

    def update_structured_data(
        self, structured_data: StructuredData, brain_id: str
    ) -> StructuredData:
        document = structured_data.model_dump(mode="json")
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE data_structured_data SET
                        data = %s::jsonb,
                        types = %s,
                        metadata = %s::jsonb,
                        inserted_at = %s,
                        document = %s::jsonb
                    WHERE id = %s
                    """,
                    (
                        json.dumps(structured_data.data or {}, default=str),
                        list(structured_data.types or []),
                        json.dumps(structured_data.metadata or {}, default=str),
                        structured_data.inserted_at,
                        json.dumps(document, default=str),
                        structured_data.id,
                    ),
                )
            conn.commit()
        return structured_data

    def get_last_text_chunks(self, brain_id: str, limit: int = 10) -> list[TextChunk]:
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT document FROM data_text_chunks
                    ORDER BY inserted_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [TextChunk.model_validate(row["document"]) for row in rows]

    def get_last_structured_data(
        self, brain_id: str, limit: int = 10
    ) -> list[StructuredData]:
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT document FROM data_structured_data
                    ORDER BY inserted_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [StructuredData.model_validate(row["document"]) for row in rows]


_postgresql_data_client: Optional[PostgreSQLDataClient] = None


def get_postgresql_data_client() -> PostgreSQLDataClient:
    global _postgresql_data_client
    if _postgresql_data_client is None:
        _postgresql_data_client = PostgreSQLDataClient()
    return _postgresql_data_client

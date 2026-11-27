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
from datetime import datetime
from typing import Any, Iterator, List, Literal, Optional, Tuple

import psycopg2
import psycopg2.extras

from src.adapters.interfaces.data import DataClient, SearchResult
from src.config import SEARCH_FTS_REGCONFIGS, config
from src.constants.conflicts import FactConflict, conflict_from_document, conflict_to_document
from src.constants.data import (
    Brain,
    KGChanges,
    KGChangesType,
    Observation,
    StructuredData,
    TextChunk,
)

from ._naming import (
    brain_db_name,
    brain_id_from_db_name,
    is_internal_brain_db_suffix,
)
from ._provisioning import (
    borrow,
    ensure_brain_database,
    ensure_database_exists,
    get_brain_pool,
    get_system_pool,
    clear_brain_database,
    drop_brain_database,
    list_brain_database_names,
)


_SYSTEM_DDL = """
CREATE TABLE IF NOT EXISTS data_brains (
    id TEXT PRIMARY KEY,
    name_key TEXT UNIQUE NOT NULL,
    pat TEXT NOT NULL,
    document JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS data_crawl_specs (
    brain_id TEXT PRIMARY KEY,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    last_crawled_at TIMESTAMPTZ,
    last_started_at TIMESTAMPTZ,
    document JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_data_crawl_specs_due
    ON data_crawl_specs (enabled, last_crawled_at, last_started_at);
CREATE TABLE IF NOT EXISTS data_source_sync (
    brain_id TEXT NOT NULL,
    bucket TEXT NOT NULL,
    item_id TEXT NOT NULL,
    document JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (brain_id, bucket, item_id)
);
CREATE INDEX IF NOT EXISTS idx_data_source_sync_bucket
    ON data_source_sync (brain_id, bucket);
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

_SEARCH_DDL = """
ALTER TABLE data_text_chunks
    ADD COLUMN IF NOT EXISTS search_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED;
ALTER TABLE data_text_chunks
    ADD COLUMN IF NOT EXISTS search_len integer
    GENERATED ALWAYS AS (
        GREATEST(length(to_tsvector('english', coalesce(text, ''))), 0)
    ) STORED;
CREATE INDEX IF NOT EXISTS idx_data_text_chunks_search_tsv
    ON data_text_chunks USING gin (search_tsv);
"""


def _search_alt_ddl(regconfig: str) -> str:
    if regconfig not in SEARCH_FTS_REGCONFIGS:
        raise ValueError(f"Unsupported FTS regconfig {regconfig!r}")
    return f"""
ALTER TABLE data_text_chunks
    ADD COLUMN IF NOT EXISTS search_tsv_alt tsvector
    GENERATED ALWAYS AS (to_tsvector('{regconfig}', coalesce(text, ''))) STORED;
ALTER TABLE data_text_chunks
    ADD COLUMN IF NOT EXISTS search_len_alt integer
    GENERATED ALWAYS AS (
        GREATEST(length(to_tsvector('{regconfig}', coalesce(text, ''))), 0)
    ) STORED;
CREATE INDEX IF NOT EXISTS idx_data_text_chunks_search_tsv_alt
    ON data_text_chunks USING gin (search_tsv_alt);
"""


def _bm25_sql(
    regconfig: str,
    tsv_col: str,
    len_col: str,
    *,
    query_match: str | None = None,
) -> str:
    if tsv_col not in {"search_tsv", "search_tsv_alt"}:
        raise ValueError(f"Unsupported tsv column {tsv_col!r}")
    if len_col not in {"search_len", "search_len_alt"}:
        raise ValueError(f"Unsupported len column {len_col!r}")
    if regconfig not in SEARCH_FTS_REGCONFIGS and regconfig != "english":
        raise ValueError(f"Unsupported FTS regconfig {regconfig!r}")
    if query_match is None:
        query_match = "or" if tsv_col == "search_tsv_alt" else "and"
    if query_match not in {"and", "or"}:
        raise ValueError(f"Unsupported BM25 query_match {query_match!r}")
    if query_match == "and":
        match_sql = f"c.{tsv_col} @@ plainto_tsquery('{regconfig}', %s)"
    else:
        match_sql = "TRUE"
    return f"""
                    WITH query_lexemes AS (
                        SELECT unnest(tsvector_to_array(to_tsvector('{regconfig}', %s)))
                            AS lexeme
                    ),
                    coll AS (
                        SELECT
                            COUNT(*)::double precision AS n_docs,
                            COALESCE(
                                AVG(GREATEST({len_col}, 1)),
                                1
                            )::double precision AS avgdl
                        FROM data_text_chunks
                    ),
                    idf AS (
                        SELECT
                            s.word AS lexeme,
                            ln(
                                1.0 + (coll.n_docs - s.ndoc + 0.5)
                                / (s.ndoc + 0.5)
                            ) AS idf
                        FROM ts_stat(
                            'SELECT {tsv_col} FROM data_text_chunks'
                        ) AS s
                        CROSS JOIN coll
                        WHERE s.word IN (SELECT lexeme FROM query_lexemes)
                    ),
                    docs AS (
                        SELECT
                            c.id,
                            c.document,
                            c.text,
                            GREATEST(c.{len_col}, 1)::double precision AS dl,
                            lex.lexeme,
                            COALESCE(
                                array_length(lex.positions, 1),
                                1
                            )::double precision AS tf
                        FROM data_text_chunks c
                        CROSS JOIN LATERAL unnest(c.{tsv_col})
                            AS lex(lexeme, positions)
                        WHERE {match_sql}
                          AND lex.lexeme IN (SELECT lexeme FROM query_lexemes)
                    )
                    SELECT
                        d.id,
                        d.document,
                        d.text,
                        SUM(
                            COALESCE(i.idf, 0)
                            * (d.tf * (%s + 1.0))
                            / (
                                d.tf
                                + %s * (
                                    1.0 - %s
                                    + %s * d.dl
                                    / NULLIF(c.avgdl, 0)
                                )
                            )
                        ) AS bm25
                    FROM docs d
                    CROSS JOIN coll c
                    LEFT JOIN idf i ON i.lexeme = d.lexeme
                    GROUP BY d.id, d.document, d.text
                    ORDER BY bm25 DESC, d.id ASC
                    LIMIT %s
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


_CONFLICTS_DDL = """
CREATE TABLE IF NOT EXISTS data_conflicts (
    id TEXT PRIMARY KEY,
    pair_key TEXT UNIQUE NOT NULL,
    claim_key TEXT,
    status TEXT NOT NULL,
    subject_name TEXT,
    source_keys TEXT[],
    fact_ids TEXT[],
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    document JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_data_conflicts_status ON data_conflicts (status);
CREATE INDEX IF NOT EXISTS idx_data_conflicts_claim ON data_conflicts (claim_key);
CREATE INDEX IF NOT EXISTS idx_data_conflicts_subject ON data_conflicts (subject_name);
CREATE INDEX IF NOT EXISTS idx_data_conflicts_detected ON data_conflicts (detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_conflicts_fact_ids ON data_conflicts USING gin (fact_ids);
CREATE INDEX IF NOT EXISTS idx_data_conflicts_source_keys ON data_conflicts USING gin (source_keys);

CREATE TABLE IF NOT EXISTS data_conflict_scan_state (
    id TEXT PRIMARY KEY,
    after_subject TEXT,
    after_predicate TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    document JSONB NOT NULL
);
"""


_INGEST_GROUP_DDL = """
CREATE TABLE IF NOT EXISTS data_ingest_group_credits (
    group_tracking TEXT NOT NULL,
    task_id TEXT NOT NULL,
    status TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (group_tracking, task_id)
);

CREATE TABLE IF NOT EXISTS data_ingest_group_meters (
    group_tracking TEXT PRIMARY KEY,
    document JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


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
        self._search_ready_brains: set[str] = set()
        self._search_alt_brains: set[str] = set()

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
        alt_regconfig = config.search_fts_regconfig_for_brain(brain_id)
        needs_base = brain_id not in self._initialized_brains
        needs_search = (
            config.search_enabled and brain_id not in self._search_ready_brains
        )
        needs_alt = (
            config.search_enabled
            and alt_regconfig is not None
            and brain_id not in self._search_alt_brains
        )
        if not needs_base and not needs_search and not needs_alt:
            return
        with self._lock:
            alt_regconfig = config.search_fts_regconfig_for_brain(brain_id)
            needs_base = brain_id not in self._initialized_brains
            needs_search = (
                config.search_enabled and brain_id not in self._search_ready_brains
            )
            needs_alt = (
                config.search_enabled
                and alt_regconfig is not None
                and brain_id not in self._search_alt_brains
            )
            if not needs_base and not needs_search and not needs_alt:
                return
            ensure_brain_database(brain_id)
            with borrow(get_brain_pool(brain_id)) as conn:
                with conn.cursor() as cur:
                    if needs_base:
                        cur.execute(_BRAIN_DDL)
                        cur.execute(_INGEST_GROUP_DDL)
                        cur.execute(_CONFLICTS_DDL)
                        from src.core.saving.pipeline_store import DDL as PIPELINE_DDL
                        cur.execute(PIPELINE_DDL)
                    if needs_search:
                        cur.execute(_SEARCH_DDL)
                    if needs_alt and alt_regconfig is not None:
                        cur.execute(_search_alt_ddl(alt_regconfig))
                conn.commit()
            if needs_base:
                self._initialized_brains.add(brain_id)
            if needs_search:
                self._search_ready_brains.add(brain_id)
            if needs_alt:
                self._search_alt_brains.add(brain_id)

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

    def delete_text_chunks(self, ids: List[str], brain_id: str) -> None:
        if not ids:
            return
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM data_text_chunks WHERE id = ANY(%s)",
                    (list(ids),),
                )
            conn.commit()

    def set_text_chunk_task_ids(
        self, chunk_id: str, task_ids: List[str], brain_id: str
    ) -> None:
        payload = json.dumps(list(task_ids))
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE data_text_chunks
                    SET metadata = jsonb_set(coalesce(metadata, '{}'::jsonb), '{ingest_task_ids}', %s::jsonb, true),
                        document = jsonb_set(document, '{metadata,ingest_task_ids}', %s::jsonb, true)
                    WHERE id = %s
                    """,
                    (payload, payload, chunk_id),
                )
            conn.commit()

    def delete_observations(
        self, ids: List[str], brain_id: str, resource_ids: Optional[List[str]] = None
    ) -> None:
        if not ids and not resource_ids:
            return
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM data_observations
                    WHERE id = ANY(%s) OR resource_id = ANY(%s)
                    """,
                    (list(ids or []), list(resource_ids or [])),
                )
            conn.commit()

    def delete_structured_data(self, structured_id: str, brain_id: str) -> None:
        if not structured_id:
            return
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM data_structured_data WHERE id = %s",
                    (structured_id,),
                )
            conn.commit()

    def mark_text_chunks_superseded(
        self,
        ids: List[str],
        superseded_at: str,
        brain_id: str,
        superseded_by: str | None = None,
        valid_to: str | None = None,
    ) -> None:
        if not ids:
            return
        patch = {"superseded_at": superseded_at}
        if valid_to:
            patch["valid_to"] = valid_to
        if superseded_by:
            patch["superseded_by"] = superseded_by
        payload = json.dumps(patch)
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE data_text_chunks
                    SET metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb,
                        document = jsonb_set(
                            COALESCE(document, '{}'::jsonb),
                            '{metadata}',
                            (CASE WHEN jsonb_typeof(document->'metadata') = 'object'
                                  THEN document->'metadata' ELSE '{}'::jsonb END) || %s::jsonb,
                            true
                        )
                    WHERE id = ANY(%s)
                    """,
                    (payload, payload, list(ids)),
                )
            conn.commit()

    def fill_text_chunk_validity(
        self, chunk_id: str, brain_id: str, patch: dict[str, Any]
    ) -> None:
        if not chunk_id or not patch:
            return
        payload = json.dumps(patch)
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE data_text_chunks
                    SET metadata = %s::jsonb || COALESCE(metadata, '{}'::jsonb),
                        document = jsonb_set(
                            COALESCE(document, '{}'::jsonb),
                            '{metadata}',
                            %s::jsonb || CASE WHEN jsonb_typeof(document->'metadata') = 'object'
                                              THEN document->'metadata' ELSE '{}'::jsonb END,
                            true
                        )
                    WHERE id = %s
                    """,
                    (payload, payload, chunk_id),
                )
            conn.commit()

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

    def search_bm25(
        self,
        text: str,
        brain_id: str,
        collection: str = "text_chunks",
        limit: int = 10,
    ) -> List[Tuple[TextChunk, float]]:
        if not text or not text.strip() or limit <= 0:
            return []
        if collection not in ("*", "text_chunks"):
            return []
        k1 = float(config.search_bm25_k1)
        b = float(config.search_bm25_b)
        alt_regconfig = config.search_fts_regconfig_for_brain(brain_id)
        if alt_regconfig:
            sql = _bm25_sql(alt_regconfig, "search_tsv_alt", "search_len_alt")
            params = (text, k1, k1, b, b, limit)
        else:
            sql = _bm25_sql("english", "search_tsv", "search_len")
            params = (text, text, k1, k1, b, b, limit)
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    sql,
                    params,
                )
                rows = cur.fetchall()
        results: List[Tuple[TextChunk, float]] = []
        for row in rows:
            chunk = TextChunk.model_validate(row["document"])
            results.append((chunk, float(row["bm25"] or 0.0)))
        return results

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
        self._ensure_brain_schema(name_key)
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

    def get_brain_by_pat(self, pat: str) -> Optional[Brain]:
        with self._system_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT document FROM data_brains WHERE pat = %s",
                    (pat,),
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
        registered = [Brain.model_validate(row["document"]) for row in rows]
        by_key = {brain.name_key: brain for brain in registered}
        known_db_names = {
            brain_db_name(brain.name_key) for brain in registered
        } | {brain_db_name(brain.id) for brain in registered}

        for db_name in list_brain_database_names():
            if db_name in known_db_names:
                continue
            suffix = brain_id_from_db_name(db_name)
            if not suffix or is_internal_brain_db_suffix(suffix):
                continue
            if suffix not in by_key:
                by_key[suffix] = Brain(name_key=suffix)

        return sorted(by_key.values(), key=lambda brain: brain.name_key)

    def clear_brain_data(self, brain_id: str) -> None:
        clear_brain_database(brain_id)

    def delete_brain_registry(self, brain_id: str) -> bool:
        with self._system_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM data_brains WHERE name_key = %s",
                    (brain_id,),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        drop_brain_database(brain_id)
        return deleted

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
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        query_text: str = None,
        metadata_eq: dict[str, str] | None = None,
        order: Literal["asc", "desc"] = "desc",
    ) -> Tuple[List[TextChunk], int]:
        clauses: list[str] = []
        params: list[Any] = []
        if query_text:
            pattern = _ilike_pattern(query_text)
            clauses.append("(text ILIKE %s OR metadata::text ILIKE %s)")
            params.extend([pattern, pattern])
        if metadata_eq:
            for key, value in metadata_eq.items():
                clauses.append("metadata ->> %s = %s")
                params.extend([key, value])
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_sql = "ASC" if order == "asc" else "DESC"
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT document FROM data_text_chunks
                    {where_sql}
                    ORDER BY inserted_at {order_sql}
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
        since: datetime | None = None,
        until: datetime | None = None,
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
        if since is not None:
            clauses.append("timestamp >= %s")
            params.append(since)
        if until is not None:
            clauses.append("timestamp < %s")
            params.append(until)
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


    def credit_ingest_group(
        self,
        brain_id: str,
        group_tracking: str,
        task_id: str,
        status: str,
        cost: Optional[dict[str, Any]] = None,
    ) -> bool:
        from src.core.saving.ingest_groups import (
            apply_credit_to_meter,
            empty_group_meter,
        )

        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO data_ingest_group_credits
                        (group_tracking, task_id, status)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (group_tracking, task_id) DO NOTHING
                    RETURNING task_id
                    """,
                    (group_tracking, task_id, status),
                )
                inserted = cur.fetchone()
                if not inserted:
                    conn.commit()
                    return False
                cur.execute(
                    """
                    SELECT document FROM data_ingest_group_meters
                    WHERE group_tracking = %s
                    FOR UPDATE
                    """,
                    (group_tracking,),
                )
                row = cur.fetchone()
                meter = (
                    dict(row["document"])
                    if row and row.get("document")
                    else empty_group_meter(group_tracking)
                )
                meter["group_tracking"] = group_tracking
                updated = apply_credit_to_meter(
                    meter,
                    task_id=task_id,
                    status=status,
                    cost=cost or {},
                )
                cur.execute(
                    """
                    INSERT INTO data_ingest_group_meters
                        (group_tracking, document, updated_at)
                    VALUES (%s, %s::jsonb, NOW())
                    ON CONFLICT (group_tracking) DO UPDATE SET
                        document = EXCLUDED.document,
                        updated_at = NOW()
                    """,
                    (group_tracking, json.dumps(updated, default=str)),
                )
            conn.commit()
        return True


    def _conflict_row(self, conflict: FactConflict) -> tuple:
        document = conflict_to_document(conflict)
        detected = conflict.detected_at
        try:
            from src.core.saving.validity import parse_instant

            parsed = parse_instant(detected)
            if parsed is not None:
                detected = parsed
        except Exception:
            pass
        return (
            conflict.id,
            conflict.pair_key,
            conflict.claim_key,
            conflict.status,
            conflict.subject.name if conflict.subject else None,
            list(conflict.source_keys()),
            list(conflict.fact_ids()),
            detected,
            json.dumps(document, default=str),
        )

    def save_conflict(self, conflict: FactConflict, brain_id: str) -> FactConflict:
        row = self._conflict_row(conflict)
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO data_conflicts (
                        id, pair_key, claim_key, status, subject_name,
                        source_keys, fact_ids, detected_at, document
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        pair_key = EXCLUDED.pair_key,
                        claim_key = EXCLUDED.claim_key,
                        status = EXCLUDED.status,
                        subject_name = EXCLUDED.subject_name,
                        source_keys = EXCLUDED.source_keys,
                        fact_ids = EXCLUDED.fact_ids,
                        detected_at = EXCLUDED.detected_at,
                        document = EXCLUDED.document
                    """,
                    row,
                )
            conn.commit()
        return conflict

    def update_conflict(
        self,
        conflict: FactConflict,
        brain_id: str,
        *,
        if_status: str | None = None,
        if_resolution_method: str | None = None,
    ) -> FactConflict | None:
        if if_status is None and if_resolution_method is None:
            return self.save_conflict(conflict, brain_id)
        row = self._conflict_row(conflict)
        clauses = ["id = %s"]
        params: list[Any] = [
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
            conflict.id,
        ]
        if if_status is not None:
            clauses.append("status = %s")
            params.append(if_status)
        if if_resolution_method is not None:
            clauses.append("COALESCE(document #>> '{resolution,method}', '') = %s")
            params.append(if_resolution_method)
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE data_conflicts SET
                        pair_key = %s,
                        claim_key = %s,
                        status = %s,
                        subject_name = %s,
                        source_keys = %s,
                        fact_ids = %s,
                        detected_at = %s,
                        document = %s::jsonb
                    WHERE """
                    + " AND ".join(clauses),
                    tuple(params),
                )
                updated = cur.rowcount
            conn.commit()
        if not updated:
            return None
        return conflict

    def get_conflict_by_id(self, id: str, brain_id: str) -> Optional[FactConflict]:
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT document FROM data_conflicts WHERE id = %s",
                    (id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return conflict_from_document(dict(row["document"]))

    def get_conflict_by_pair_key(
        self, pair_key: str, brain_id: str
    ) -> Optional[FactConflict]:
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT document FROM data_conflicts WHERE pair_key = %s",
                    (pair_key,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return conflict_from_document(dict(row["document"]))

    def _conflict_filters(
        self,
        status: str | None,
        source: str | None,
        subject: str | None,
        since: datetime | None,
        until: datetime | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = %s")
            params.append(status)
        if source:
            pattern = _ilike_pattern(source)
            clauses.append(
                "("
                "%s = ANY(source_keys) OR "
                "EXISTS (SELECT 1 FROM unnest(COALESCE(source_keys, ARRAY[]::text[])) AS k WHERE k ILIKE %s) OR "
                "document::text ILIKE %s"
                ")"
            )
            params.extend([source, pattern, pattern])
        if subject:
            clauses.append("subject_name ILIKE %s")
            params.append(_ilike_pattern(subject))
        if since is not None:
            clauses.append("detected_at >= %s")
            params.append(since)
        if until is not None:
            clauses.append("detected_at < %s")
            params.append(until)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where_sql, params

    def get_conflicts_list(
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        status: str | None = None,
        source: str | None = None,
        subject: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Tuple[list[FactConflict], int]:
        where_sql, params = self._conflict_filters(
            status, source, subject, since, until
        )
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"SELECT COUNT(*) AS total FROM data_conflicts {where_sql}",
                    params,
                )
                total = int((cur.fetchone() or {}).get("total") or 0)
                cur.execute(
                    f"""
                    SELECT document FROM data_conflicts
                    {where_sql}
                    ORDER BY detected_at DESC, id ASC
                    OFFSET %s LIMIT %s
                    """,
                    (*params, skip, limit),
                )
                rows = cur.fetchall()
        conflicts: list[FactConflict] = []
        for row in rows:
            parsed = conflict_from_document(dict(row["document"]))
            if parsed is not None:
                conflicts.append(parsed)
        return conflicts, total

    def get_open_conflicts_for_facts(
        self, fact_ids: list[str], brain_id: str
    ) -> list[FactConflict]:
        if not fact_ids:
            return []
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT document FROM data_conflicts
                    WHERE status = 'open' AND fact_ids && %s
                    """,
                    (list(fact_ids),),
                )
                rows = cur.fetchall()
        conflicts: list[FactConflict] = []
        for row in rows:
            parsed = conflict_from_document(dict(row["document"]))
            if parsed is not None:
                conflicts.append(parsed)
        return conflicts

    def get_conflict_scan_cursor(self, brain_id: str) -> Optional[dict[str, str]]:
        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT after_subject, after_predicate
                    FROM data_conflict_scan_state
                    WHERE id = %s
                    """,
                    ("full_scan",),
                )
                row = cur.fetchone()
        if not row:
            return None
        subject = str(row.get("after_subject") or "").strip()
        predicate = str(row.get("after_predicate") or "").strip()
        if not subject and not predicate:
            return None
        return {"after_subject": subject, "after_predicate": predicate}

    def set_conflict_scan_cursor(
        self, brain_id: str, cursor: dict[str, str] | None
    ) -> None:
        subject = str((cursor or {}).get("after_subject") or "").strip()
        predicate = str((cursor or {}).get("after_predicate") or "").strip()
        with self._brain_connection(brain_id) as conn:
            with conn.cursor() as cur:
                if not cursor or (not subject and not predicate):
                    cur.execute(
                        "DELETE FROM data_conflict_scan_state WHERE id = %s",
                        ("full_scan",),
                    )
                else:
                    document = json.dumps(
                        {"after_subject": subject, "after_predicate": predicate}
                    )
                    cur.execute(
                        """
                        INSERT INTO data_conflict_scan_state (
                            id, after_subject, after_predicate, document, updated_at
                        )
                        VALUES (%s, %s, %s, %s::jsonb, NOW())
                        ON CONFLICT (id) DO UPDATE SET
                            after_subject = EXCLUDED.after_subject,
                            after_predicate = EXCLUDED.after_predicate,
                            document = EXCLUDED.document,
                            updated_at = NOW()
                        """,
                        ("full_scan", subject, predicate, document),
                    )
            conn.commit()

    def get_ingest_group(
        self, brain_id: str, group_tracking: str
    ) -> Optional[dict[str, Any]]:
        from src.core.saving.ingest_groups import normalize_meter_document

        with self._brain_connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT document FROM data_ingest_group_meters
                    WHERE group_tracking = %s
                    """,
                    (group_tracking,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return normalize_meter_document(dict(row["document"]))

    def save_crawl_spec(self, spec, brain_id: str):
        from src.core.saving.help_center_crawl import CrawlSpec, spec_from_document
        from src.core.saving.validity import parse_instant

        payload = spec if isinstance(spec, CrawlSpec) else spec_from_document(spec, brain_id)
        if payload is None:
            raise ValueError("Invalid crawl spec")
        payload.brain_id = brain_id
        document = payload.model_dump(mode="json")
        crawled = parse_instant(payload.last_crawled_at)
        started = parse_instant(payload.last_started_at)
        with self._system_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO data_crawl_specs (
                        brain_id, enabled, last_crawled_at, last_started_at, document, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                    ON CONFLICT (brain_id) DO UPDATE SET
                        enabled = EXCLUDED.enabled,
                        last_crawled_at = EXCLUDED.last_crawled_at,
                        last_started_at = EXCLUDED.last_started_at,
                        document = EXCLUDED.document,
                        updated_at = NOW()
                    """,
                    (
                        brain_id,
                        bool(payload.enabled),
                        crawled,
                        started,
                        json.dumps(document, default=str),
                    ),
                )
            conn.commit()
        return payload

    def get_crawl_spec(self, brain_id: str):
        from src.core.saving.help_center_crawl import spec_from_document

        with self._system_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT document FROM data_crawl_specs WHERE brain_id = %s",
                    (brain_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return spec_from_document(row["document"], brain_id)

    def delete_crawl_spec(self, brain_id: str) -> bool:
        with self._system_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM data_crawl_specs WHERE brain_id = %s",
                    (brain_id,),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def list_due_crawl_specs(self, now=None, force: bool = False):
        from src.core.saving.help_center_crawl import spec_from_document, spec_is_due
        from src.core.saving.validity import utc_now

        instant = now or utc_now()
        with self._system_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT brain_id, document FROM data_crawl_specs WHERE enabled = TRUE"
                )
                rows = cur.fetchall()
        due = []
        for row in rows:
            spec = spec_from_document(row["document"], row["brain_id"])
            if spec_is_due(spec, now=instant, force=force):
                due.append(spec)
        return due

    def claim_due_crawl_spec(
        self,
        brain_id: str,
        now=None,
        force: bool = False,
        task_id: str | None = None,
    ):
        from src.core.saving.help_center_crawl import (
            mark_started,
            spec_from_document,
            spec_is_due,
        )
        from src.core.saving.validity import parse_instant, utc_now

        instant = now or utc_now()
        with self._system_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT document FROM data_crawl_specs
                    WHERE brain_id = %s
                    FOR UPDATE
                    """,
                    (brain_id,),
                )
                row = cur.fetchone()
                spec = spec_from_document(row["document"], brain_id) if row else None
                if not spec_is_due(spec, now=instant, force=force):
                    conn.commit()
                    return None
                claimed = mark_started(spec, now=instant, task_id=task_id)
                document = claimed.model_dump(mode="json")
                cur.execute(
                    """
                    UPDATE data_crawl_specs
                    SET enabled = %s,
                        last_crawled_at = %s,
                        last_started_at = %s,
                        document = %s::jsonb,
                        updated_at = NOW()
                    WHERE brain_id = %s
                    """,
                    (
                        bool(claimed.enabled),
                        parse_instant(claimed.last_crawled_at),
                        parse_instant(claimed.last_started_at),
                        json.dumps(document, default=str),
                        brain_id,
                    ),
                )
            conn.commit()
        return claimed

    def finish_crawl_spec(
        self,
        brain_id: str,
        *,
        live_source_keys,
        removed_source_keys,
        conflict_scan_task_id,
        task_id: str | None = None,
        now=None,
        error: str | None = None,
        warning: str | None = None,
    ):
        from src.core.saving.help_center_crawl import mark_finished, spec_from_document
        from src.core.saving.validity import parse_instant, utc_now

        instant = now or utc_now()
        with self._system_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT document FROM data_crawl_specs
                    WHERE brain_id = %s
                    FOR UPDATE
                    """,
                    (brain_id,),
                )
                row = cur.fetchone()
                if not row:
                    conn.commit()
                    return None
                spec = spec_from_document(row["document"], brain_id)
                if spec is None:
                    conn.commit()
                    return None
                if task_id and spec.last_task_id and spec.last_task_id != task_id:
                    conn.commit()
                    return spec
                finished = mark_finished(
                    spec,
                    live_source_keys=live_source_keys,
                    removed_source_keys=removed_source_keys,
                    conflict_scan_task_id=conflict_scan_task_id,
                    now=instant,
                    error=error,
                    warning=warning,
                )
                document = finished.model_dump(mode="json")
                cur.execute(
                    """
                    UPDATE data_crawl_specs
                    SET enabled = %s,
                        last_crawled_at = %s,
                        last_started_at = %s,
                        document = %s::jsonb,
                        updated_at = NOW()
                    WHERE brain_id = %s
                    """,
                    (
                        bool(finished.enabled),
                        parse_instant(finished.last_crawled_at),
                        parse_instant(finished.last_started_at),
                        json.dumps(document, default=str),
                        brain_id,
                    ),
                )
            conn.commit()
        return finished

    def source_sync_put(self, brain_id: str, bucket: str, item_id: str, document: dict):
        with self._system_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO data_source_sync (
                        brain_id, bucket, item_id, document, updated_at
                    )
                    VALUES (%s, %s, %s, %s::jsonb, NOW())
                    ON CONFLICT (brain_id, bucket, item_id) DO UPDATE SET
                        document = EXCLUDED.document,
                        updated_at = NOW()
                    """,
                    (brain_id, bucket, item_id, json.dumps(document, default=str)),
                )
            conn.commit()

    def source_sync_get(self, brain_id: str, bucket: str, item_id: str):
        with self._system_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT document FROM data_source_sync
                    WHERE brain_id = %s AND bucket = %s AND item_id = %s
                    """,
                    (brain_id, bucket, item_id),
                )
                row = cur.fetchone()
        if not row:
            return None
        document = row["document"]
        return dict(document) if isinstance(document, dict) else None

    def source_sync_list(self, brain_id: str, bucket: str) -> dict[str, dict]:
        with self._system_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT item_id, document FROM data_source_sync
                    WHERE brain_id = %s AND bucket = %s
                    """,
                    (brain_id, bucket),
                )
                rows = cur.fetchall()
        found: dict[str, dict] = {}
        for row in rows:
            document = row["document"]
            if isinstance(document, dict):
                found[str(row["item_id"])] = dict(document)
        return found

    def source_sync_delete(self, brain_id: str, bucket: str, item_id: str) -> bool:
        with self._system_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM data_source_sync
                    WHERE brain_id = %s AND bucket = %s AND item_id = %s
                    """,
                    (brain_id, bucket, item_id),
                )
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted



_postgresql_data_client: Optional[PostgreSQLDataClient] = None


def get_postgresql_data_client() -> PostgreSQLDataClient:
    global _postgresql_data_client
    if _postgresql_data_client is None:
        _postgresql_data_client = PostgreSQLDataClient()
    return _postgresql_data_client

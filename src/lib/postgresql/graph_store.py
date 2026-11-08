"""
File: /graph_store.py
Project: postgresql
Created Date: Sunday May 24th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import re
import threading
from typing import Any, Dict, List, Optional

import networkx as nx
import psycopg2
import psycopg2.extras

from src.config import config

from ._provisioning import borrow_brain, commit_brain, current_brain_transaction, ensure_brain_database
from .read_query import (
    MAX_READ_QUERY_ROWS,
    READ_QUERY_TIMEOUT_MS,
    ReadQueryValidationError,
    validate_read_only_sql,
)


class GraphDatabaseError(Exception):
    code: Optional[str] = None

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.code = code


# Agents often write data->'field' ILIKE ..., which fails because -> returns jsonb.
_JSONB_ILIKE_RE = re.compile(
    r"\bdata\s*->\s*'([^']+)'\s+(ILIKE|LIKE)\b",
    re.IGNORECASE,
)


def normalize_read_sql(sql: str) -> str:
    return _JSONB_ILIKE_RE.sub(r"data->>'\1' \2", sql)


class _NodeProxy(dict):
    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)

    @property
    def element_id(self) -> str:
        return str(self.get("uuid", ""))


class _RelationshipProxy(dict):
    def __init__(self, data: dict, start: _NodeProxy, end: _NodeProxy):
        super().__init__(data)
        self._start = start
        self._end = end
        self.type = data.get("rel_type") or data.get("type") or ""

    @property
    def start_node(self) -> _NodeProxy:
        return self._start

    @property
    def end_node(self) -> _NodeProxy:
        return self._end

    @property
    def nodes(self) -> list:
        return [self._start, self._end]


class _BrainGraph:
    def __init__(self, brain_id: str):
        self.brain_id = brain_id
        self.graph = nx.MultiDiGraph()
        self.publication_epoch = 0

    def node_data(self, uuid: str) -> dict:
        return dict(self.graph.nodes[uuid])

    def labels(self, uuid: str) -> list[str]:
        return list(self.node_data(uuid).get("labels") or [])

    def upsert_node(self, labels: list[str], identification: dict, properties: dict) -> str:
        node_uuid = properties.get("uuid") or identification.get("uuid")
        if not node_uuid:
            raise ValueError("Unable to resolve node uuid for merge")
        match_uuid = node_uuid if node_uuid in self.graph else node_uuid
        merged = {}
        if match_uuid in self.graph:
            merged.update(self.node_data(match_uuid))
        incoming = dict(properties)
        if "description" in incoming and merged.get("description"):
            existing_desc = str(merged.get("description") or "").strip()
            incoming_desc = str(incoming.get("description") or "").strip()
            if existing_desc and incoming_desc and incoming_desc.lower() not in existing_desc.lower():
                if existing_desc.lower() not in incoming_desc.lower():
                    incoming["description"] = f"{existing_desc} | {incoming_desc}"
                else:
                    incoming["description"] = incoming_desc
            elif existing_desc and not incoming_desc:
                incoming["description"] = existing_desc
        for list_key in ("source_chunk_ids", "aliases"):
            if list_key in incoming or list_key in merged:
                seen = []
                for value in list(merged.get(list_key) or []) + list(
                    incoming.get(list_key) or []
                ):
                    item = str(value).strip()
                    if item and item not in seen:
                        seen.append(item)
                if seen:
                    incoming[list_key] = seen
        merged.update(incoming)
        if identification.get("name") is not None:
            merged["name"] = identification.get("name")
        merged["labels"] = labels or merged.get("labels") or []
        merged["uuid"] = match_uuid
        merged.setdefault("name", identification.get("name"))
        self.graph.add_node(match_uuid, **merged)
        return match_uuid

    def find_nodes(
        self,
        labels: Optional[list[str]] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        results: list[str] = []
        for uuid, data in self.graph.nodes(data=True):
            if labels and not set(labels).issubset(set(data.get("labels") or [])):
                continue
            if filters:
                matched = True
                for key, value in filters.items():
                    if data.get(key) != value:
                        matched = False
                        break
                if not matched:
                    continue
            results.append(uuid)
        return results



class PostgreSQLGraphStore:
    """
    Graph driver that persists each brain into its own Postgres database.

    The per-brain database is provisioned lazily through ``ensure_brain_database``
    and accessed through the shared LRU pool registry. Each database owns a flat
    schema (no ``brain_id`` columns, no ``kg_brains`` reference table) because
    the brain identity is encoded in the database name itself.
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS kg_publication_epoch (
        singleton BOOLEAN PRIMARY KEY CHECK (singleton),
        epoch BIGINT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS kg_nodes (
        uuid TEXT PRIMARY KEY,
        data JSONB NOT NULL
    );
    CREATE TABLE IF NOT EXISTS kg_relationships (
        uuid TEXT PRIMARY KEY,
        rel_type TEXT NOT NULL,
        source_uuid TEXT NOT NULL,
        target_uuid TEXT NOT NULL,
        data JSONB NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_kg_relationships_endpoints
        ON kg_relationships(source_uuid, target_uuid);
    CREATE TABLE IF NOT EXISTS kg_hub_bridges (
        event_a TEXT NOT NULL,
        event_b TEXT NOT NULL,
        shared_entity TEXT NOT NULL,
        shared_entity_name TEXT NOT NULL DEFAULT '',
        weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
        PRIMARY KEY (event_a, event_b, shared_entity),
        CHECK (event_a < event_b)
    );
    CREATE INDEX IF NOT EXISTS idx_kg_hub_bridges_a ON kg_hub_bridges(event_a);
    CREATE INDEX IF NOT EXISTS idx_kg_hub_bridges_b ON kg_hub_bridges(event_b);
    CREATE TABLE IF NOT EXISTS kg_topic_sessions (
        topic_id TEXT NOT NULL,
        topic_label TEXT NOT NULL DEFAULT '',
        session_id TEXT NOT NULL,
        weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
        PRIMARY KEY (topic_id, session_id)
    );
    CREATE INDEX IF NOT EXISTS idx_kg_topic_sessions_session
        ON kg_topic_sessions(session_id);
    """

    _SEARCH_DOCUMENT_SQL = (
        "coalesce(data->>'name', '') || ' ' || "
        "coalesce(data->>'description', '') || ' ' || "
        "coalesce(data->>'search_text', '')"
    )
    _SEARCH_DDL = f"""
    ALTER TABLE kg_nodes
        ADD COLUMN IF NOT EXISTS search_tsv tsvector
        GENERATED ALWAYS AS (
            to_tsvector('english'::regconfig, {_SEARCH_DOCUMENT_SQL})
        ) STORED;
    ALTER TABLE kg_nodes
        ADD COLUMN IF NOT EXISTS search_len integer
        GENERATED ALWAYS AS (
            GREATEST(
                length(to_tsvector('english'::regconfig, {_SEARCH_DOCUMENT_SQL})),
                0
            )
        ) STORED;
    CREATE INDEX IF NOT EXISTS idx_kg_nodes_search_tsv
        ON kg_nodes USING gin (search_tsv);
    """

    def __init__(self) -> None:
        config.postgresql.validate_credentials()
        self._brains: dict[str, _BrainGraph] = {}
        self._schema_ready: set[str] = set()
        self._search_ready_brains: set[str] = set()
        self._schema_lock = threading.Lock()
        self._brains_lock = threading.RLock()

    def _ensure_brain_schema(self, brain_id: str) -> None:
        if current_brain_transaction(brain_id):
            return
        needs_base = brain_id not in self._schema_ready
        needs_search = (
            bool(config.search_enabled) and brain_id not in self._search_ready_brains
        )
        if not needs_base and not needs_search:
            return
        with self._schema_lock:
            needs_base = brain_id not in self._schema_ready
            needs_search = (
                bool(config.search_enabled)
                and brain_id not in self._search_ready_brains
            )
            if not needs_base and not needs_search:
                return
            ensure_brain_database(brain_id)
            with borrow_brain(brain_id) as conn:
                if needs_base:
                    with conn.cursor() as cur:
                        cur.execute(self._DDL)
                    commit_brain(conn)
                    self._schema_ready.add(brain_id)
                if needs_search:
                    with conn.cursor() as cur:
                        cur.execute(self._SEARCH_DDL)
                    commit_brain(conn)
                    self._search_ready_brains.add(brain_id)

    @contextmanager
    def _connection(self, brain_id: str):
        self._ensure_brain_schema(brain_id)
        with borrow_brain(brain_id) as conn:
            yield conn

    def _ensure_brain_row(self, brain_id: str) -> None:
        self._ensure_brain_schema(brain_id)

    def search_nodes_bm25(
        self,
        text: str,
        brain_id: str,
        *,
        limit: int = 10,
        node_labels: list[str] | None = None,
        node_uuids: list[str] | None = None,
    ) -> list[tuple[str, float, dict]]:
        if not config.search_enabled or not text or not str(text).strip() or limit <= 0:
            return []
        self._ensure_brain_schema(brain_id)
        k1 = float(config.search_bm25_k1)
        b = float(config.search_bm25_b)
        labels = [
            str(item).strip().upper()
            for item in (node_labels or [])
            if str(item).strip()
        ]
        uuids = [str(item).strip() for item in (node_uuids or []) if str(item).strip()]
        sql = """
            WITH query_lexemes AS (
                SELECT unnest(tsvector_to_array(to_tsvector('english', %s)))
                    AS lexeme
            ),
            coll AS (
                SELECT
                    COUNT(*)::double precision AS n_docs,
                    COALESCE(AVG(GREATEST(search_len, 1)), 1)::double precision AS avgdl
                FROM kg_nodes
            ),
            idf AS (
                SELECT
                    s.word AS lexeme,
                    ln(
                        1.0 + (coll.n_docs - s.ndoc + 0.5)
                        / (s.ndoc + 0.5)
                    ) AS idf
                FROM ts_stat('SELECT search_tsv FROM kg_nodes') AS s
                CROSS JOIN coll
                WHERE s.word IN (SELECT lexeme FROM query_lexemes)
            ),
            docs AS (
                SELECT
                    n.uuid,
                    n.data,
                    GREATEST(n.search_len, 1)::double precision AS dl,
                    lex.lexeme,
                    COALESCE(array_length(lex.positions, 1), 1)::double precision AS tf
                FROM kg_nodes n
                CROSS JOIN LATERAL unnest(n.search_tsv) AS lex(lexeme, positions)
                WHERE n.search_tsv @@ plainto_tsquery('english', %s)
                  AND lex.lexeme IN (SELECT lexeme FROM query_lexemes)
                  AND (%s::text[] IS NULL OR n.uuid = ANY(%s))
                  AND (
                    %s::text[] IS NULL
                    OR EXISTS (
                        SELECT 1
                        FROM jsonb_array_elements_text(
                            coalesce(n.data->'labels', '[]'::jsonb)
                        ) AS lbl
                        WHERE upper(lbl) = ANY(%s)
                    )
                  )
            )
            SELECT
                d.uuid,
                d.data,
                SUM(
                    COALESCE(i.idf, 0)
                    * (d.tf * (%s + 1.0))
                    / (
                        d.tf
                        + %s * (
                            1.0 - %s
                            + %s * d.dl / NULLIF(c.avgdl, 0)
                        )
                    )
                ) AS bm25
            FROM docs d
            CROSS JOIN coll c
            LEFT JOIN idf i ON i.lexeme = d.lexeme
            GROUP BY d.uuid, d.data
            ORDER BY bm25 DESC, d.uuid ASC
            LIMIT %s
        """
        label_param = labels or None
        uuid_param = uuids or None
        params = (
            text,
            text,
            uuid_param,
            uuid_param,
            label_param,
            label_param,
            k1,
            k1,
            b,
            b,
            limit,
        )
        with self._connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        out: list[tuple[str, float, dict]] = []
        for row in rows:
            node_id = str(row.get("uuid") or "").strip()
            if not node_id:
                continue
            data = row.get("data") or {}
            if not isinstance(data, dict):
                data = {}
            out.append((node_id, float(row.get("bm25") or 0.0), data))
        return out

    def get_brain(self, brain_id: str) -> _BrainGraph:
        return self._load_brain(brain_id)

    def invalidate_brain(self, brain_id: str) -> None:
        with self._brains_lock:
            self._brains.pop(brain_id, None)

    def _graph_cache(self, brain_id: str):
        transaction = current_brain_transaction(brain_id)
        if transaction:
            return transaction.graphs.setdefault(self, {})
        return self._brains

    def advance_publication_epoch(self, brain_id: str) -> None:
        with self._connection(brain_id) as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO kg_publication_epoch VALUES (TRUE, 1)
                ON CONFLICT (singleton) DO UPDATE
                SET epoch = kg_publication_epoch.epoch + 1
            """)

    def _db_graph_state(self, brain_id: str) -> tuple[int, int, int]:
        with self._connection(brain_id) as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT (SELECT COUNT(*) FROM kg_nodes),
                       (SELECT COUNT(*) FROM kg_relationships),
                       COALESCE((SELECT epoch FROM kg_publication_epoch WHERE singleton), 0)
            """)
            return tuple(cur.fetchone())

    def _load_brain(self, brain_id: str) -> _BrainGraph:
        """Read path: refresh from Postgres when another process wrote data."""
        with self._brains_lock:
            cached = self._graph_cache(brain_id).get(brain_id)
            cached_counts = (
                (cached.graph.number_of_nodes(), cached.graph.number_of_edges(), cached.publication_epoch)
                if cached is not None
                else None
            )

        if cached is not None and cached_counts is not None:
            try:
                if self._db_graph_state(brain_id) == cached_counts:
                    return cached
            except Exception:
                pass

        with self._brains_lock:
            return self._reload_brain_locked(brain_id)

    def _ensure_brain_in_memory_locked(self, brain_id: str) -> _BrainGraph:
        """Write path: reuse the in-process graph; load from DB only if missing."""
        cached = self._graph_cache(brain_id).get(brain_id)
        if cached is not None:
            return cached
        return self._reload_brain_locked(brain_id)

    def _reload_brain_locked(self, brain_id: str) -> _BrainGraph:
        self._ensure_brain_schema(brain_id)
        brain = _BrainGraph(brain_id)
        with self._connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if current_brain_transaction(brain_id) is None:
                    cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                cur.execute("SELECT epoch FROM kg_publication_epoch WHERE singleton")
                epoch = cur.fetchone()
                if epoch:
                    brain.publication_epoch = epoch["epoch"]
                cur.execute("SELECT uuid, data FROM kg_nodes")
                for row in cur.fetchall():
                    data = dict(row["data"])
                    data["uuid"] = row["uuid"]
                    brain.graph.add_node(row["uuid"], **data)
                cur.execute(
                    """
                    SELECT uuid, rel_type, source_uuid, target_uuid, data
                    FROM kg_relationships
                    """
                )
                for row in cur.fetchall():
                    payload = dict(row["data"])
                    payload["uuid"] = row["uuid"]
                    payload["rel_type"] = row["rel_type"]
                    brain.graph.add_edge(
                        row["source_uuid"],
                        row["target_uuid"],
                        key=row["uuid"],
                        **payload,
                    )
        if brain.graph.number_of_edges() == 0:
            self._hydrate_relationships_from_vectors(brain_id, brain)
        self._graph_cache(brain_id)[brain_id] = brain
        return brain

    def _hydrate_relationships_from_vectors(
        self, brain_id: str, brain: _BrainGraph
    ) -> None:
        with self._connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'vectors_relationships'
                    """
                )
                if not cur.fetchone():
                    return
                cur.execute("SELECT uuid, metadata FROM vectors_relationships")
                rows = cur.fetchall()

        for row in rows:
            meta = dict(row.get("metadata") or {})
            rel_uuid = str(meta.get("uuid") or row.get("uuid") or "")
            if not rel_uuid:
                continue
            node_ids = meta.get("node_ids") or []
            if len(node_ids) < 2:
                continue
            source_uuid = str(node_ids[0])
            target_uuid = str(node_ids[1])
            if (
                source_uuid not in brain.graph
                or target_uuid not in brain.graph
            ):
                continue
            rel_type = str(meta.get("predicate") or "RELATED")
            edge_data = {"rel_type": rel_type, "deprecated": False}
            brain.graph.add_edge(
                source_uuid,
                target_uuid,
                key=rel_uuid,
                **edge_data,
            )
            transaction = current_brain_transaction(brain_id)
            if transaction and transaction.read_only:
                continue
            self._persist_relationship(
                brain_id,
                rel_uuid,
                rel_type,
                source_uuid,
                target_uuid,
                edge_data,
            )

    def _persist_node(self, brain_id: str, uuid: str, data: dict) -> None:
        payload = dict(data)
        payload.pop("uuid", None)
        with self._connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO kg_nodes (uuid, data)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT (uuid) DO UPDATE SET data = EXCLUDED.data
                    """,
                    (uuid, json.dumps(payload, default=str)),
                )
            commit_brain(conn)

    def _persist_relationship(
        self,
        brain_id: str,
        uuid: str,
        rel_type: str,
        source_uuid: str,
        target_uuid: str,
        data: dict,
    ) -> None:
        payload = dict(data)
        payload.pop("uuid", None)
        payload.pop("rel_type", None)
        with self._connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO kg_relationships
                        (uuid, rel_type, source_uuid, target_uuid, data)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (uuid) DO UPDATE SET
                        rel_type = EXCLUDED.rel_type,
                        source_uuid = EXCLUDED.source_uuid,
                        target_uuid = EXCLUDED.target_uuid,
                        data = EXCLUDED.data
                    """,
                    (
                        uuid,
                        rel_type,
                        source_uuid,
                        target_uuid,
                        json.dumps(payload, default=str),
                    ),
                )
            commit_brain(conn)

    def _delete_node(self, brain_id: str, uuid: str) -> None:
        brain = self._load_brain(brain_id)
        if uuid in brain.graph:
            brain.graph.remove_node(uuid)
        with self._connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM kg_relationships WHERE source_uuid = %s OR target_uuid = %s",
                    (uuid, uuid),
                )
                cur.execute(
                    "DELETE FROM kg_nodes WHERE uuid = %s",
                    (uuid,),
                )
            commit_brain(conn)

    def _delete_relationship(self, brain_id: str, rel_uuid: str) -> None:
        brain = self._load_brain(brain_id)
        for source, target, key in list(brain.graph.edges(keys=True)):
            if key == rel_uuid:
                brain.graph.remove_edge(source, target, key)
        with self._connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM kg_relationships WHERE uuid = %s",
                    (rel_uuid,),
                )
            commit_brain(conn)

    def ensure_database(self, database: str) -> None:
        self._ensure_brain_row(database)
        self._load_brain(database)

    def execute_read_query(
        self,
        brain_id: str,
        query: str,
        max_rows: int = MAX_READ_QUERY_ROWS,
    ) -> dict[str, Any]:
        try:
            sql = validate_read_only_sql(normalize_read_sql(query))
        except ReadQueryValidationError as exc:
            raise GraphDatabaseError(str(exc)) from exc
        self._ensure_brain_schema(brain_id)
        with self._connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"SET LOCAL statement_timeout = {READ_QUERY_TIMEOUT_MS}")
                cur.execute(sql)
                rows = cur.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        if truncated:
            rows = rows[:max_rows]
        return {
            "records": [dict(row) for row in rows],
            "truncated": truncated,
        }

    def merge_node(
        self,
        brain_id: str,
        labels: list[str],
        identification: dict[str, Any],
        properties: dict[str, Any],
    ) -> str:
        with self._brains_lock:
            brain = self._ensure_brain_in_memory_locked(brain_id)
            merged_props = {**identification, **properties}
            node_uuid = brain.upsert_node(labels, identification, merged_props)
            payload = dict(brain.node_data(node_uuid))
        self._persist_node(brain_id, node_uuid, payload)
        return node_uuid

    def _ensure_relationship_endpoint(
        self,
        brain: _BrainGraph,
        labels: list[str],
        name: Any,
        node_uuid: Optional[str],
    ) -> tuple[Optional[str], Optional[dict]]:
        if node_uuid and node_uuid in brain.graph:
            return node_uuid, None
        if node_uuid:
            resolved = brain.upsert_node(
                labels,
                {"uuid": node_uuid, "name": name},
                {"uuid": node_uuid, "name": name},
            )
            return resolved, dict(brain.node_data(resolved))
        return self.resolve_node_by_name_labels(brain, labels, name), None

    def merge_relationship(
        self,
        brain_id: str,
        subject_labels: list[str],
        subject_name: Any,
        object_labels: list[str],
        object_name: Any,
        rel_type: str,
        rel_props: dict[str, Any],
        subject_uuid: Optional[str] = None,
        object_uuid: Optional[str] = None,
    ) -> Optional[tuple[dict, dict]]:
        pending_nodes: list[tuple[str, dict]] = []
        with self._brains_lock:
            brain = self._ensure_brain_in_memory_locked(brain_id)
            source_uuid, source_payload = self._ensure_relationship_endpoint(
                brain, subject_labels, subject_name, subject_uuid
            )
            target_uuid, target_payload = self._ensure_relationship_endpoint(
                brain, object_labels, object_name, object_uuid
            )
            if source_payload is not None and source_uuid:
                pending_nodes.append((source_uuid, source_payload))
            if target_payload is not None and target_uuid:
                pending_nodes.append((target_uuid, target_payload))
            if not source_uuid or not target_uuid:
                raise GraphDatabaseError(
                    "Cannot persist relationship "
                    f"{rel_type!r}: missing endpoint nodes "
                    f"(subject_uuid={subject_uuid!r}, object_uuid={object_uuid!r}, "
                    f"subject_name={subject_name!r}, object_name={object_name!r})"
                )
            props = dict(rel_props)
            props["rel_type"] = rel_type
            rel_uuid = props.get("uuid") or f"{source_uuid}-{rel_type}-{target_uuid}"
            props["uuid"] = rel_uuid
            if brain.graph.has_edge(source_uuid, target_uuid, key=rel_uuid):
                existing = dict(
                    brain.graph.get_edge_data(source_uuid, target_uuid, key=rel_uuid)
                )
                existing.update(props)
                props = existing
                props["uuid"] = rel_uuid
                props["rel_type"] = rel_type
            brain.graph.add_edge(source_uuid, target_uuid, key=rel_uuid, **props)
            source_data = dict(brain.node_data(source_uuid))
            target_data = dict(brain.node_data(target_uuid))
            rel_payload = (
                rel_uuid,
                rel_type,
                source_uuid,
                target_uuid,
                dict(props),
            )

        for node_id, payload in pending_nodes:
            self._persist_node(brain_id, node_id, payload)
        self._persist_relationship(brain_id, *rel_payload)
        return source_data, target_data

    def resolve_node_by_name_labels(
        self, brain: _BrainGraph, labels: list[str], name: Any
    ) -> Optional[str]:
        for uuid, data in brain.graph.nodes(data=True):
            if data.get("name") != name:
                continue
            if labels and not set(labels).issubset(set(data.get("labels") or [])):
                continue
            return uuid
        return None

    def match_nodes_by_uuid(self, brain: _BrainGraph, uuids: list[str]) -> list[dict]:
        return [
            self.node_to_record(brain, node_uuid)
            for node_uuid in uuids
            if node_uuid in brain.graph
        ]

    def neighborhood_records(self, brain: _BrainGraph, node_uuid: str) -> list[dict]:
        records = []
        if node_uuid not in brain.graph:
            return records
        for source, target, key in brain.graph.edges(keys=True):
            if source != node_uuid and target != node_uuid:
                continue
            neighbor_uuid = target if source == node_uuid else source
            record = self.relationship_to_record(brain, source, target, key, node_uuid)
            record.update(
                {
                    "m_uuid": neighbor_uuid,
                    "m_name": brain.node_data(neighbor_uuid).get("name"),
                    "m_labels": brain.labels(neighbor_uuid),
                    "m_description": brain.node_data(neighbor_uuid).get("description"),
                    "m_properties": brain.node_data(neighbor_uuid),
                    "m_polarity": brain.node_data(neighbor_uuid).get("polarity"),
                    "m_metadata": brain.node_data(neighbor_uuid).get("metadata"),
                    "m_happened_at": brain.node_data(neighbor_uuid).get("happened_at"),
                    "m_last_updated": brain.node_data(neighbor_uuid).get("last_updated"),
                    "m_observations": brain.node_data(neighbor_uuid).get("observations"),
                }
            )
            records.append(record)
        return records

    def update_entity_properties(
        self,
        brain_id: str,
        entity_uuid: str,
        is_relationship: bool,
        new_properties: dict[str, Any],
        properties_to_remove: list[str],
    ) -> Optional[dict]:
        brain = self._load_brain(brain_id)
        if is_relationship:
            for source, target, key, data in brain.graph.edges(keys=True, data=True):
                if data.get("uuid") != entity_uuid:
                    continue
                for prop, value in new_properties.items():
                    data[prop] = value
                for prop in properties_to_remove:
                    data.pop(prop, None)
                self._persist_relationship(
                    brain_id, key, data.get("rel_type"), source, target, data
                )
                return {
                    "rel_type": data.get("rel_type"),
                    "rel_description": data.get("description"),
                    "properties": dict(data),
                }
            return None
        if entity_uuid not in brain.graph:
            return None
        data = brain.node_data(entity_uuid)
        for prop, value in new_properties.items():
            data[prop] = value
        for prop in properties_to_remove:
            data.pop(prop, None)
        self._persist_node(brain_id, entity_uuid, data)
        return self.node_to_record(brain, entity_uuid)

    def delete_nodes_by_uuids(self, brain_id: str, uuids: list[str]) -> list[dict]:
        brain = self._load_brain(brain_id)
        records = []
        for node_uuid in uuids:
            if node_uuid not in brain.graph:
                continue
            records.append(
                {
                    "node": {
                        "uuid": node_uuid,
                        "name": brain.node_data(node_uuid).get("name"),
                        "labels": brain.labels(node_uuid),
                        "description": brain.node_data(node_uuid).get("description"),
                        "properties": brain.node_data(node_uuid),
                        "polarity": brain.node_data(node_uuid).get("polarity"),
                        "metadata": brain.node_data(node_uuid).get("metadata"),
                        "happened_at": brain.node_data(node_uuid).get("happened_at"),
                        "last_updated": brain.node_data(node_uuid).get("last_updated"),
                        "observations": brain.node_data(node_uuid).get("observations"),
                    }
                }
            )
            self._delete_node(brain_id, node_uuid)
        return records

    def delete_relationships_by_uuids(self, brain_id: str, rel_uuids: list[str]) -> list[dict]:
        brain = self._load_brain(brain_id)
        records = []
        for rel_uuid in rel_uuids:
            for source, target, key, data in list(brain.graph.edges(keys=True, data=True)):
                if key != rel_uuid and data.get("uuid") != rel_uuid:
                    continue
                record = self.node_to_record(brain, source, "n")
                record.update(self.relationship_to_record(brain, source, target, key, source))
                record.update(self.node_to_record(brain, target, "m"))
                records.append(record)
                self._delete_relationship(brain_id, key)
        return records

    def list_labels(self, brain: _BrainGraph) -> list[str]:
        return sorted(
            {label for _, data in brain.graph.nodes(data=True) for label in data.get("labels", [])}
        )

    def list_relationship_types(self, brain: _BrainGraph) -> list[str]:
        return sorted(
            {
                edge_data.get("rel_type")
                for _, _, edge_data in brain.graph.edges(data=True)
                if edge_data.get("rel_type")
            }
        )

    def list_node_property_keys(self, brain: _BrainGraph) -> list[str]:
        return sorted(
            {
                key
                for _, data in brain.graph.nodes(data=True)
                for key in data.keys()
                if key not in {"labels", "uuid"}
            }
        )

    def event_names(self, brain: _BrainGraph) -> list[str]:
        return [
            str(data.get("name"))
            for _, data in brain.graph.nodes(data=True)
            if "EVENT" in (data.get("labels") or []) and data.get("name")
        ]

    def check_node_exists(
        self, brain: _BrainGraph, uuid: str, name: str, labels: list[str]
    ) -> bool:
        for node_uuid, data in brain.graph.nodes(data=True):
            if node_uuid != uuid:
                continue
            if data.get("name") != name:
                continue
            if not set(labels).issubset(set(data.get("labels") or [])):
                continue
            return True
        return False

    def node_to_record(self, brain: _BrainGraph, uuid: str, alias: str = "n") -> dict:
        data = brain.node_data(uuid)
        record = {
            f"{alias}_uuid" if alias != "n" else "uuid": uuid,
            f"{alias}_name" if alias != "n" else "name": data.get("name"),
            f"{alias}_labels" if alias != "n" else "labels": data.get("labels", []),
            f"{alias}_description" if alias != "n" else "description": data.get("description"),
            f"{alias}_properties" if alias != "n" else "properties": {
                k: v for k, v in data.items() if k not in {"labels"}
            },
            "properties": {k: v for k, v in data.items() if k not in {"labels"}},
            "polarity": data.get("polarity"),
            "metadata": data.get("metadata"),
            "happened_at": data.get("happened_at"),
            "last_updated": data.get("last_updated"),
            "observations": data.get("observations"),
        }
        if alias != "n":
            record[f"{alias}_polarity"] = data.get("polarity")
            record[f"{alias}_metadata"] = data.get("metadata")
            record[f"{alias}_happened_at"] = data.get("happened_at")
            record[f"{alias}_last_updated"] = data.get("last_updated")
            record[f"{alias}_observations"] = data.get("observations")
        return record

    def relationship_to_record(
        self,
        brain: _BrainGraph,
        source_uuid: str,
        target_uuid: str,
        rel_uuid: str,
        direction_from: str,
    ) -> dict:
        edge_data = brain.graph.edges[source_uuid, target_uuid, rel_uuid]
        rel_type = edge_data.get("rel_type") or edge_data.get("name") or ""
        direction = "out" if direction_from == source_uuid else "in"
        start = _NodeProxy(brain.node_data(source_uuid))
        end = _NodeProxy(brain.node_data(target_uuid))
        rel = _RelationshipProxy(
            {
                **edge_data,
                "rel_type": rel_type,
                "uuid": rel_uuid,
                "description": edge_data.get("description"),
            },
            start,
            end,
        )
        return {
            "rel": rel,
            "rel_type": rel_type,
            "rel_description": edge_data.get("description"),
            "rel_properties": dict(edge_data),
            "rel_flowkey": edge_data.get("flow_key"),
            "rel_uuid": rel_uuid,
            "rel_last_updated": edge_data.get("last_updated"),
            "rel_observations": edge_data.get("observations"),
            "rel_amount": edge_data.get("amount"),
            "direction": direction,
            "r_direction": direction,
        }

    def neighbor_records_for_uuids(
        self,
        brain: _BrainGraph,
        uuids: list[str],
        same_type_only: bool = False,
        of_types: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        records = []
        for node_uuid in uuids:
            if node_uuid not in brain.graph:
                continue
            for source, target, key in brain.graph.edges(keys=True):
                if source != node_uuid and target != node_uuid:
                    continue
                neighbor_uuid = target if source == node_uuid else source
                if same_type_only and not set(brain.labels(node_uuid)).intersection(brain.labels(neighbor_uuid)):
                    continue
                if of_types and not set(of_types).intersection(brain.labels(neighbor_uuid)):
                    continue
                record = self.node_to_record(brain, node_uuid)
                record.update(self.relationship_to_record(brain, source, target, key, node_uuid))
                record.update(
                    {
                        "c_uuid": neighbor_uuid,
                        "c_name": brain.node_data(neighbor_uuid).get("name"),
                        "c_labels": brain.labels(neighbor_uuid),
                        "c_description": brain.node_data(neighbor_uuid).get("description"),
                        "c_properties": brain.node_data(neighbor_uuid),
                    }
                )
                records.append(record)
                if limit is not None and len(records) >= limit:
                    return records
        return records

    def replace_hub_bridges(
        self,
        brain_id: str,
        bridges: list[tuple[str, str, str, str, float]],
    ) -> int:
        self._ensure_brain_schema(brain_id)
        with self._connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM kg_hub_bridges")
                for event_a, event_b, shared_entity, shared_name, weight in bridges:
                    if not event_a or not event_b or event_a == event_b:
                        continue
                    a, b = (event_a, event_b) if event_a < event_b else (event_b, event_a)
                    cur.execute(
                        """
                        INSERT INTO kg_hub_bridges
                            (event_a, event_b, shared_entity, shared_entity_name, weight)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (event_a, event_b, shared_entity) DO UPDATE SET
                            shared_entity_name = EXCLUDED.shared_entity_name,
                            weight = EXCLUDED.weight
                        """,
                        (a, b, shared_entity, shared_name or "", float(weight)),
                    )
            commit_brain(conn)
        return len(bridges)

    def delete_hub_bridges_for_entities(
        self, brain_id: str, entity_uuids: list[str]
    ) -> int:
        unique = sorted({str(u) for u in entity_uuids if u})
        if not unique:
            return 0
        self._ensure_brain_schema(brain_id)
        with self._connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM kg_hub_bridges WHERE shared_entity = ANY(%s)",
                    (unique,),
                )
                deleted = cur.rowcount or 0
            commit_brain(conn)
        return int(deleted)

    def upsert_hub_bridges(
        self,
        brain_id: str,
        bridges: list[tuple[str, str, str, str, float]],
    ) -> int:
        if not bridges:
            return 0
        self._ensure_brain_schema(brain_id)
        with self._connection(brain_id) as conn:
            with conn.cursor() as cur:
                for event_a, event_b, shared_entity, shared_name, weight in bridges:
                    if not event_a or not event_b or event_a == event_b:
                        continue
                    a, b = (event_a, event_b) if event_a < event_b else (event_b, event_a)
                    cur.execute(
                        """
                        INSERT INTO kg_hub_bridges
                            (event_a, event_b, shared_entity, shared_entity_name, weight)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (event_a, event_b, shared_entity) DO UPDATE SET
                            shared_entity_name = EXCLUDED.shared_entity_name,
                            weight = EXCLUDED.weight
                        """,
                        (a, b, shared_entity, shared_name or "", float(weight)),
                    )
            commit_brain(conn)
        return len(bridges)

    def get_hub_bridges_for_events(
        self,
        brain_id: str,
        event_uuids: list[str],
    ) -> list[tuple[str, str, str, str, float]]:
        if not event_uuids:
            return []
        self._ensure_brain_schema(brain_id)
        unique = sorted({str(u) for u in event_uuids if u})
        if not unique:
            return []
        with self._connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_a, event_b, shared_entity, shared_entity_name, weight
                    FROM kg_hub_bridges
                    WHERE event_a = ANY(%s) OR event_b = ANY(%s)
                    ORDER BY event_a, event_b, shared_entity
                    """,
                    (unique, unique),
                )
                rows = cur.fetchall()
        return [
            (
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3] or ""),
                float(row[4] or 1.0),
            )
            for row in rows
        ]

    def count_hub_bridges(self, brain_id: str) -> int:
        self._ensure_brain_schema(brain_id)
        with self._connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM kg_hub_bridges")
                row = cur.fetchone()
        return int(row[0] if row else 0)

    def replace_topic_sessions(
        self,
        brain_id: str,
        rows: list[tuple[str, str, str, float]],
    ) -> int:
        self._ensure_brain_schema(brain_id)
        with self._connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM kg_topic_sessions")
                for topic_id, topic_label, session_id, weight in rows:
                    if not topic_id or not session_id:
                        continue
                    cur.execute(
                        """
                        INSERT INTO kg_topic_sessions
                            (topic_id, topic_label, session_id, weight)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (topic_id, session_id) DO UPDATE SET
                            topic_label = EXCLUDED.topic_label,
                            weight = EXCLUDED.weight
                        """,
                        (
                            topic_id,
                            topic_label or "",
                            session_id,
                            float(weight),
                        ),
                    )
            commit_brain(conn)
        return len(rows)

    def list_topic_sessions(
        self, brain_id: str
    ) -> list[tuple[str, str, str, float]]:
        self._ensure_brain_schema(brain_id)
        with self._connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT topic_id, topic_label, session_id, weight
                    FROM kg_topic_sessions
                    ORDER BY topic_id, session_id
                    """
                )
                rows = cur.fetchall()
        return [
            (
                str(row[0]),
                str(row[1] or ""),
                str(row[2]),
                float(row[3] or 1.0),
            )
            for row in rows
        ]

    def count_topic_sessions(self, brain_id: str) -> int:
        self._ensure_brain_schema(brain_id)
        with self._connection(brain_id) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM kg_topic_sessions")
                row = cur.fetchone()
        return int(row[0] if row else 0)

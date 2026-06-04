"""
File: /graph_store.py
Project: postgresql
Created Date: Sunday May 24th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import threading
from typing import Any, Dict, List, Optional

import networkx as nx
import psycopg2
import psycopg2.extras

from src.config import config

from ._provisioning import borrow, ensure_brain_database, get_brain_pool
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

    def node_data(self, uuid: str) -> dict:
        return dict(self.graph.nodes[uuid])

    def labels(self, uuid: str) -> list[str]:
        return list(self.node_data(uuid).get("labels") or [])

    def upsert_node(self, labels: list[str], identification: dict, properties: dict) -> str:
        node_uuid = properties.get("uuid") or identification.get("uuid")
        match_uuid = None
        for candidate_uuid, data in self.graph.nodes(data=True):
            if identification.get("uuid") and candidate_uuid != identification["uuid"]:
                continue
            if identification.get("name") and data.get("name") != identification.get("name"):
                continue
            if labels and not set(labels).issubset(set(data.get("labels") or [])):
                continue
            extra_match = True
            for key, value in identification.items():
                if key in {"uuid", "name"}:
                    continue
                if data.get(key) != value:
                    extra_match = False
                    break
            if extra_match:
                match_uuid = candidate_uuid
                break
        if match_uuid is None:
            match_uuid = node_uuid or identification.get("uuid")
        if match_uuid is None:
            raise ValueError("Unable to resolve node uuid for merge")
        merged = {}
        if match_uuid in self.graph:
            merged.update(self.node_data(match_uuid))
        merged.update(properties)
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
    """

    def __init__(self) -> None:
        config.postgresql.validate_credentials()
        self._brains: dict[str, _BrainGraph] = {}
        self._schema_ready: set[str] = set()
        self._schema_lock = threading.Lock()

    def _ensure_brain_schema(self, brain_id: str) -> None:
        if brain_id in self._schema_ready:
            return
        with self._schema_lock:
            if brain_id in self._schema_ready:
                return
            ensure_brain_database(brain_id)
            with borrow(get_brain_pool(brain_id)) as conn:
                with conn.cursor() as cur:
                    cur.execute(self._DDL)
                conn.commit()
            self._schema_ready.add(brain_id)

    @contextmanager
    def _connection(self, brain_id: str):
        self._ensure_brain_schema(brain_id)
        with borrow(get_brain_pool(brain_id)) as conn:
            yield conn

    def _ensure_brain_row(self, brain_id: str) -> None:
        self._ensure_brain_schema(brain_id)

    def get_brain(self, brain_id: str) -> _BrainGraph:
        return self._load_brain(brain_id)

    def _load_brain(self, brain_id: str) -> _BrainGraph:
        if brain_id in self._brains:
            return self._brains[brain_id]
        self._ensure_brain_schema(brain_id)
        brain = _BrainGraph(brain_id)
        with self._connection(brain_id) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
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
        self._brains[brain_id] = brain
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
            conn.commit()

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
            conn.commit()

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
            conn.commit()

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
            conn.commit()

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
            sql = validate_read_only_sql(query)
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
        brain = self._load_brain(brain_id)
        merged_props = {**identification, **properties}
        node_uuid = brain.upsert_node(labels, identification, merged_props)
        self._persist_node(brain_id, node_uuid, brain.node_data(node_uuid))
        return node_uuid

    def merge_relationship(
        self,
        brain_id: str,
        subject_labels: list[str],
        subject_name: Any,
        object_labels: list[str],
        object_name: Any,
        rel_type: str,
        rel_props: dict[str, Any],
    ) -> Optional[tuple[dict, dict]]:
        brain = self._load_brain(brain_id)
        source_uuid = self.resolve_node_by_name_labels(brain, subject_labels, subject_name)
        target_uuid = self.resolve_node_by_name_labels(brain, object_labels, object_name)
        if not source_uuid or not target_uuid:
            return None
        props = dict(rel_props)
        props["rel_type"] = rel_type
        rel_uuid = props.get("uuid") or f"{source_uuid}-{rel_type}-{target_uuid}"
        props["uuid"] = rel_uuid
        brain.graph.add_edge(source_uuid, target_uuid, key=rel_uuid, **props)
        self._persist_relationship(
            brain_id, rel_uuid, rel_type, source_uuid, target_uuid, props
        )
        return brain.node_data(source_uuid), brain.node_data(target_uuid)

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

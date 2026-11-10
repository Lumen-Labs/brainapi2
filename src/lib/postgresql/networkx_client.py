"""
File: /networkx_client.py
Project: postgresql
Created Date: Sunday May 24th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.adapters.interfaces.embeddings import VectorStoreClient
from src.adapters.interfaces.graph import GraphClient, PredicateWithFlowKey
from src.constants.kg import (
    IdentificationParams,
    Node,
    NodeDict,
    Predicate,
    PredicateDict,
    SearchEntitiesResult,
    SearchRelationshipsResult,
    Triple,
)
from src.lib.postgresql.graph_store import (
    GraphDatabaseError,
    PostgreSQLGraphStore,
    _RelationshipProxy,
)
from src.lib.postgresql._provisioning import clear_brain_database
from src.utils.serialization.data import always_dict


class NetworkXGraphClient(GraphClient):
    """
    Graph client backed by PostgreSQL persistence and NetworkX in-memory traversal.
    """

    def __init__(self):
        self._store = PostgreSQLGraphStore()

    def ensure_database(self, database: str) -> None:
        self._store.ensure_database(database)

    def clear_brain_data(self, brain_id: str) -> None:
        clear_brain_database(brain_id, table_prefixes=("kg_",))
        self._store.invalidate_brain(brain_id)

    def execute_operation(self, operation: str, brain_id: str) -> Any:
        self._store.ensure_database(brain_id)
        return self._store.execute_read_query(brain_id, operation)

    @property
    def graphdb_type(self) -> str:
        return "postgresql-networkx"

    @property
    def graphdb_description(self) -> str:
        return (
            "The graph database is PostgreSQL (NetworkX-backed). "
            "You MUST write read-only SQL only. Cypher is NOT supported. "
            "Never use MATCH, RETURN, labels(n), type(r), or EXPLAIN MATCH. "
            "Never query a table named nodes or relationships — use kg_nodes and kg_relationships. "
            "IMPORTANT: for text filters use data->>'field' (text), never data->'field' (jsonb) with ILIKE/LIKE. "
            "Always send one complete SQL statement in a single tool call "
            "(never split a WITH RECURSIVE CTE across multiple calls). "
            "Tables: "
            "kg_nodes(uuid TEXT PRIMARY KEY, data JSONB) where data holds name, labels, "
            "description and other node properties; "
            "kg_relationships(uuid TEXT PRIMARY KEY, rel_type TEXT, source_uuid TEXT, "
            "target_uuid TEXT, data JSONB). "
            "Examples: "
            "SELECT COUNT(*) AS total FROM kg_nodes; "
            "SELECT uuid, data->>'name' AS name, data->'labels' AS labels FROM kg_nodes "
            "WHERE data->>'name' ILIKE '%Counsel%' OR data->>'description' ILIKE '%Mental Health%' LIMIT 50; "
            "SELECT uuid, data FROM kg_nodes WHERE uuid = '<uuid>'; "
            "SELECT n.uuid, n.data->>'name' AS name, r.rel_type, m.uuid AS target_uuid, "
            "m.data->>'name' AS target_name, m.data->'labels' AS target_labels "
            "FROM kg_relationships r "
            "JOIN kg_nodes n ON n.uuid = r.source_uuid "
            "JOIN kg_nodes m ON m.uuid = r.target_uuid "
            "WHERE n.uuid = '<uuid>'; "
            "Filter nodes by label with data->'labels' ? 'PERSON'. "
            "Prefer 1-hop joins first. For multi-hop use this single complete statement: "
            "WITH RECURSIVE walk AS ("
            "SELECT source_uuid, target_uuid, rel_type, 1 AS depth "
            "FROM kg_relationships WHERE source_uuid = '<uuid>' "
            "UNION ALL "
            "SELECT r.source_uuid, r.target_uuid, r.rel_type, w.depth + 1 "
            "FROM walk w JOIN kg_relationships r ON r.source_uuid = w.target_uuid "
            "WHERE w.depth < 3"
            ") SELECT * FROM walk LIMIT 50; "
            "Only read queries are allowed; results are capped at 100 rows."
        )

    def _clean_labels(self, labels: list[str]) -> list[str]:
        return [
            label.replace(" ", "_")
            .upper()
            .replace("-", "_")
            .replace(".", "_")
            .replace(",", "_")
            .replace(":", "_")
            .replace(";", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace("[", "_")
            .replace("]", "_")
            .replace("{", "_")
            .replace("}", "_")
            .replace("'", "_")
            for label in labels
        ]

    def _node_label_set(self, brain, node_uuid: str) -> set[str]:
        return set(self._clean_labels(list(brain.labels(node_uuid))))

    def _clean_property_key(self, property_key: str) -> str:
        words = property_key.split()
        return words[0].lower() + "".join(word.capitalize() for word in words[1:])

    def _format_property_key(self, property_key: str) -> str:
        sanitized_key = property_key.rstrip("`").lstrip("`")
        needs_quoting = any(
            char in sanitized_key
            for char in ["-", " ", ".", "+", "*", "/", "%", ":", "@", "#", "$"]
        )
        cleaned_key = self._clean_property_key(sanitized_key)
        return f"`{cleaned_key}`" if needs_quoting else cleaned_key

    def _format_value(self, value: Any) -> str:
        if isinstance(value, str):
            v = value.replace("'", "\\'")
            return f"'{v}'"
        if isinstance(value, (int, float, bool)):
            return str(value)
        if value is None:
            return "null"
        v = str(value).replace("'", "\\'")
        return f"'{v}'"

    def add_nodes(
        self,
        nodes: list[Node],
        brain_id: str,
        identification_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> list[Node] | str:
        self._store.ensure_database(brain_id)
        for node in nodes:
            if not node.uuid:
                raise ValueError("Node uuid is required for upsert")
            identification_dict: dict[str, Any] = {"uuid": node.uuid, "name": node.name}
            merged_metadata = {**(metadata or {}), **(node.metadata or {})}
            all_properties: dict[str, Any] = {
                **(node.properties or {}),
                "metadata": merged_metadata or None,
            }
            if identification_params:
                for key, value in identification_params.items():
                    normalized_key = self._clean_property_key(key)
                    if normalized_key not in {"uuid", "name"}:
                        identification_dict[normalized_key] = value
            for attr in (
                "description",
                "happened_at",
                "last_updated",
                "observations",
                "polarity",
            ):
                if getattr(node, attr, None) is not None:
                    all_properties[attr] = getattr(node, attr)
            all_properties["uuid"] = node.uuid
            self._store.merge_node(
                brain_id,
                self._clean_labels(node.labels),
                identification_dict,
                all_properties,
            )
        return [
            Node(
                uuid=node.uuid,
                labels=node.labels,
                name=node.name,
                description=node.description,
                properties=node.properties,
                metadata={**(node.metadata or {}), **(metadata or {})},
            )
            for node in nodes
        ]

    def add_relationship(
        self,
        subject: Node,
        predicate: Predicate,
        to_object: Node,
        brain_id: str,
    ) -> str:
        rel_props: dict[str, Any] = {
            "description": predicate.description,
            "uuid": predicate.uuid,
            "v_id": (predicate.properties or {}).get("v_id"),
            "flow_key": predicate.flow_key,
        }
        for attr in ("properties", "happened_at", "last_updated", "amount"):
            value = getattr(predicate, attr, None)
            if value:
                rel_props[attr] = value
        for attr in ("properties", "description", "happened_at", "flow_key", "last_updated", "amount"):
            for obj in (subject, to_object):
                value = getattr(obj, attr, None)
                if value and attr not in rel_props:
                    rel_props[attr] = value
        self._store.ensure_database(brain_id)
        self._store.merge_relationship(
            brain_id,
            self._clean_labels(subject.labels),
            subject.name,
            self._clean_labels(to_object.labels),
            to_object.name,
            self._clean_labels([predicate.name])[0],
            rel_props,
            subject_uuid=subject.uuid,
            object_uuid=to_object.uuid,
        )
        return "ok"

    def search_graph(self, nodes: list[Node], brain_id: str) -> list[Node]:
        if not nodes:
            return []
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        records = []
        for node in nodes:
            node_uuid = self._store.resolve_node_by_name_labels(
                brain, self._clean_labels(node.labels), node.name
            )
            if not node_uuid:
                continue
            records.append({"n": brain.node_data(node_uuid), "r": None, "m": None})
            for source, target, key in brain.graph.edges(keys=True):
                if source != node_uuid and target != node_uuid:
                    continue
                neighbor_uuid = target if source == node_uuid else source
                records.append(
                    {
                        "n": brain.node_data(node_uuid),
                        "r": brain.graph.edges[source, target, key],
                        "m": brain.node_data(neighbor_uuid),
                    }
                )
        return records


    def node_text_search(self, text: str, brain_id: str) -> list[Node]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        text_lower = text.lower()
        nodes = []
        for node_uuid, data in brain.graph.nodes(data=True):
            if text_lower in str(data.get("name", "")).lower():
                nodes.append(
                    Node(
                        uuid=node_uuid,
                        name=data.get("name", "") or "",
                        labels=list(data.get("labels") or []),
                        description=data.get("description", "") or "",
                        properties={k: v for k, v in data.items() if k != "labels"},
                    )
                )
        return nodes


    def get_nodes_by_uuid(
        self,
        uuids: list[str],
        brain_id: str,
        with_relationships: Optional[bool] = False,
        relationships_depth: Optional[int] = 1,
        relationships_type: Optional[list[str]] = None,
        preferred_labels: Optional[list[str]] = None,
    ) -> list[dict]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        depth = relationships_depth or 1
        rel_types = relationships_type
        pref = self._clean_labels(preferred_labels) if preferred_labels else None
        if not with_relationships:
            return [
                Node(
                    uuid=record["uuid"],
                    name=record["name"],
                    labels=record["labels"],
                    description=record["description"],
                    properties=record["properties"],
                )
                for record in self._store.match_nodes_by_uuid(brain, uuids)
            ]
        records = []
        for node_uuid in uuids:
            if node_uuid not in brain.graph:
                continue
            visited = {node_uuid}
            frontier = {node_uuid}
            for _ in range(depth):
                next_frontier: set[str] = set()
                for current_uuid in frontier:
                    for source, target, key, edge_data in brain.graph.edges(keys=True, data=True):
                        neighbor = target if source == current_uuid else source
                        if neighbor in visited:
                            continue
                        if rel_types and edge_data.get("rel_type") not in rel_types:
                            continue
                        neighbor_labels = brain.labels(neighbor)
                        if pref and not set(pref).intersection(neighbor_labels):
                            continue
                        record = self._store.node_to_record(brain, node_uuid)
                        record.update(
                            self._store.relationship_to_record(
                                brain, source, target, key, current_uuid
                            )
                        )
                        record.update(
                            {
                                "m_uuid": neighbor,
                                "m_name": brain.node_data(neighbor).get("name"),
                                "m_labels": neighbor_labels,
                                "m_description": brain.node_data(neighbor).get("description"),
                                "m_properties": brain.node_data(neighbor),
                            }
                        )
                        records.append(record)
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
                frontier = next_frontier
        return [
            {
                "node": Node(
                    uuid=record["uuid"],
                    name=record["name"],
                    labels=record["labels"],
                    description=record["description"],
                    properties=record["properties"],
                ),
                "relationships": record.get("rel"),
                "related_nodes": (
                    Node(
                        uuid=record["m_uuid"],
                        name=record["m_name"],
                        labels=record["m_labels"],
                        description=record["m_description"],
                        properties=record["m_properties"],
                    )
                    if record.get("m_uuid")
                    else None
                ),
            }
            for record in records
        ]


    def get_graph_entities(self, brain_id: str) -> list[str]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        labels: list[str] = []
        for node_uuid in brain.graph.nodes:
            labels.extend(brain.labels(node_uuid))
        return labels


    def get_graph_relationships(self, brain_id: str) -> list[str]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        return self._store.list_relationship_types(brain)


    def get_by_uuid(self, uuid: str, brain_id: str) -> Node:
        nodes = self.get_by_uuids([uuid], brain_id)
        return nodes[0] if nodes else None


    def get_by_uuids(self, uuids: list[str], brain_id: str) -> list[Node]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        return [
            Node(
                uuid=record.get("uuid", ""),
                name=record.get("name", ""),
                labels=record.get("labels", []),
                description=record.get("description", ""),
                properties=record.get("properties", {}),
                polarity=record.get("polarity", "neutral"),
                **(
                    {"happened_at": record.get("happened_at")}
                    if record.get("happened_at") is not None
                    else {}
                ),
                **(
                    {"last_updated": record.get("last_updated", datetime.now(timezone.utc))}
                    if record.get("last_updated") is not None
                    else {}
                ),
                **(
                    {"observations": record.get("observations", [])}
                    if record.get("observations") is not None
                    else {}
                ),
                **(
                    {"metadata": record.get("metadata", {})}
                    if record.get("metadata") is not None
                    else {}
                ),
            )
            for record in self._store.match_nodes_by_uuid(brain, uuids)
        ]


    def get_by_identification_params(
        self,
        identification_params: IdentificationParams,
        brain_id: str,
        entity_types: Optional[list[str]] = None,
    ) -> Node:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        params_dict = identification_params.model_dump(mode="json", exclude={"entity_types"})
        labels = self._clean_labels(entity_types) if entity_types else None
        matches = brain.find_nodes(labels=labels or None, filters=params_dict)
        if not matches:
            return None
        record = self._store.node_to_record(brain, matches[0])
        return Node(
            uuid=record.get("uuid", ""),
            name=record.get("name", "") or "",
            labels=record.get("labels", []) or [],
            description=record.get("description", "") or "",
            properties=record.get("properties", {}) or {},
        )


    def get_neighbors(
        self,
        nodes: list[Node | str],
        brain_id: str,
        same_type_only: bool = False,
        limit: int | None = None,
        of_types: Optional[list[str]] = None,
    ) -> Dict[str, List[Tuple[Predicate, Node]]]:
        if len(nodes) == 0:
            return {}
        node_uuids = nodes if isinstance(nodes[0], str) else [node.uuid for node in nodes]
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        cleaned_types = self._clean_labels(of_types) if of_types else None
        records = self._store.neighbor_records_for_uuids(
            brain, node_uuids, same_type_only, cleaned_types, limit
        )
        neighbors_dict: Dict[str, List[Tuple[Predicate, Node]]] = {uuid: [] for uuid in node_uuids}
        for record in records:
            source_uuid = record["uuid"]
            neighbor = (
                Predicate(
                    name=record.get("rel_type", "") or "",
                    description=record.get("rel_description", "") or "",
                    direction=record.get("direction", "neutral"),
                    properties=record.get("rel_properties", {}) or {},
                    flow_key=record.get("rel_flowkey", "") or "",
                    uuid=record.get("rel_uuid", "") or "",
                ),
                Node(
                    uuid=record.get("c_uuid", ""),
                    name=record.get("c_name", "") or "",
                    labels=record.get("c_labels", []) or [],
                    description=record.get("c_description", "") or "",
                    properties=record.get("c_properties", {}) or {},
                ),
            )
            if source_uuid in neighbors_dict:
                neighbors_dict[source_uuid].append(neighbor)
        return neighbors_dict


    def get_node_with_rel_by_uuid(
        self, rel_ids_with_node_ids: list[tuple[str, str]], brain_id: str
    ) -> list[dict]:
        """
        Get the node with the relationships by their UUIDs.
        """

    def get_neighbor_node_tuples(
        self, a_uuid: str, b_uuids: list[str], brain_id: str
    ) -> list[Tuple[Node, Predicate, Node]]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        b_set = set(b_uuids)
        tuples = []
        for source, target, key, edge_data in brain.graph.edges(keys=True, data=True):
            if source != a_uuid and target != a_uuid:
                continue
            neighbor = target if source == a_uuid else source
            if neighbor not in b_set:
                continue
            n_uuid, m_uuid = a_uuid, neighbor
            direction = "out" if source == a_uuid else "in"
            tuples.append(
                (
                    Node(
                        uuid=n_uuid,
                        name=brain.node_data(n_uuid).get("name", "") or "",
                        labels=brain.labels(n_uuid),
                        description=brain.node_data(n_uuid).get("description", "") or "",
                        properties=brain.node_data(n_uuid),
                    ),
                    Predicate(
                        name=edge_data.get("rel_type", "") or "",
                        description=edge_data.get("description", "") or "",
                        direction=direction,
                    ),
                    Node(
                        uuid=m_uuid,
                        name=brain.node_data(m_uuid).get("name", "") or "",
                        labels=brain.labels(m_uuid),
                        description=brain.node_data(m_uuid).get("description", "") or "",
                        properties=brain.node_data(m_uuid),
                    ),
                )
            )
        if not tuples and os.getenv("DEBUG") == "true":
            raise ValueError(
                f"No neighbor nodes found for UUID: {a_uuid} and b_uuids: {b_uuids}"
            )
        return tuples


    def get_connected_nodes(
        self,
        brain_id: str,
        node: Optional[Node] = None,
        uuids: Optional[list[str]] = None,
        limit: Optional[int] = 10,
        with_labels: Optional[list[str]] = None,
    ) -> list[Tuple[Node, Predicate, Node]]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        start_uuids: set[str] = set()
        if node:
            start_uuids.add(node.uuid)
        if uuids:
            start_uuids.update(uuids)
        label_filter = set(with_labels) if with_labels else None
        results = []
        for source, target, key, edge_data in brain.graph.edges(keys=True, data=True):
            if start_uuids and source not in start_uuids and target not in start_uuids:
                continue
            anchor = source if source in start_uuids or not start_uuids else target
            other = target if anchor == source else source
            if label_filter and not label_filter.intersection(brain.labels(other)):
                continue
            direction = "out" if anchor == source else "in"
            results.append(
                (
                    Node(
                        uuid=other,
                        name=brain.node_data(other).get("name", "") or "",
                        labels=brain.labels(other),
                        description=brain.node_data(other).get("description", "") or "",
                        properties=brain.node_data(other),
                    ),
                    Predicate(
                        name=edge_data.get("rel_type", "") or "",
                        description=edge_data.get("description", "") or "",
                        direction=direction,
                    ),
                    Node(
                        uuid=anchor,
                        name=brain.node_data(anchor).get("name", "") or "",
                        labels=brain.labels(anchor),
                        description=brain.node_data(anchor).get("description", "") or "",
                        properties=brain.node_data(anchor),
                    ),
                )
            )
            if limit is not None and len(results) >= limit:
                break
        return results


    def search_relationships(
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        relationship_types: Optional[list[str]] = None,
        from_node_labels: Optional[list[str]] = None,
        to_node_labels: Optional[list[str]] = None,
        relationship_uuids: Optional[list[str]] = None,
        query_text: Optional[str] = None,
        query_search_target: Optional[str] = "all",
    ) -> SearchRelationshipsResult:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        text_lower = query_text.lower() if query_text else None
        from_labels = set(self._clean_labels(from_node_labels)) if from_node_labels else None
        to_labels = set(self._clean_labels(to_node_labels)) if to_node_labels else None
        matched = []
        for source, target, key, edge_data in brain.graph.edges(keys=True, data=True):
            rel_type = edge_data.get("rel_type")
            if relationship_types and rel_type not in relationship_types:
                continue
            if relationship_uuids and key not in relationship_uuids and edge_data.get("uuid") not in relationship_uuids:
                continue
            if from_labels and not from_labels.intersection(
                self._node_label_set(brain, source)
            ):
                continue
            if to_labels and not to_labels.intersection(
                self._node_label_set(brain, target)
            ):
                continue
            if text_lower:
                haystacks = [
                    str(brain.node_data(source).get("name", "")).lower(),
                    str(brain.node_data(target).get("name", "")).lower(),
                    str(edge_data.get("description", "")).lower(),
                ]
                if query_search_target == "node_name":
                    haystacks = [str(brain.node_data(source).get("name", "")).lower()]
                elif query_search_target == "relationship_description":
                    haystacks = [str(edge_data.get("description", "")).lower()]
                elif query_search_target == "relationship_name":
                    haystacks = [str(rel_type or "").lower()]
                if not any(text_lower in h for h in haystacks):
                    continue
            matched.append((source, target, key, edge_data))
        total = len(matched)
        page = matched[skip : skip + limit]
        triples: list[Triple] = []
        for source, target, key, edge_data in page:
            direction = "out"
            triples.append(
                Triple(
                    subject=Node(
                        uuid=source,
                        name=str(brain.node_data(source).get("name") or source),
                        labels=list(brain.labels(source)),
                        description=brain.node_data(source).get("description"),
                        properties=brain.node_data(source),
                    ),
                    predicate=Predicate(
                        name=str(edge_data.get("rel_type") or ""),
                        description=edge_data.get("description") or "",
                        direction=direction,
                        observations=None,
                        level=None,
                        deprecated=edge_data.get("deprecated", False),
                    ),
                    object=Node(
                        uuid=target,
                        name=str(brain.node_data(target).get("name") or target),
                        labels=list(brain.labels(target)),
                        description=brain.node_data(target).get("description"),
                        properties=brain.node_data(target),
                    ),
                )
            )
        return SearchRelationshipsResult(results=triples, total=total)


    def search_entities(
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        node_labels: Optional[list[str]] = None,
        node_uuids: Optional[list[str]] = None,
        query_text: Optional[str] = None,
    ) -> SearchEntitiesResult:
        self.ensure_database(brain_id)
        brain = self._store._load_brain(brain_id)
        allowed_labels = (
            set(self._clean_labels(node_labels)) if node_labels else None
        )
        uuid_filter = set(node_uuids) if node_uuids else None
        text_lower = query_text.lower() if query_text else None

        matched: list[str] = []
        for node_uuid, data in brain.graph.nodes(data=True):
            if uuid_filter is not None and node_uuid not in uuid_filter:
                continue
            if allowed_labels is not None:
                if not allowed_labels.intersection(
                    self._node_label_set(brain, node_uuid)
                ):
                    continue
            if text_lower is not None:
                if text_lower not in str(data.get("name", "")).lower():
                    continue
            matched.append(node_uuid)

        total = len(matched)
        page = matched[skip : skip + limit]

        nodes: list[Node] = []
        for node_uuid in page:
            data = brain.node_data(node_uuid)
            properties = {k: v for k, v in data.items() if k != "labels"}
            nodes.append(
                Node(
                    uuid=str(node_uuid),
                    name=str(data.get("name") or node_uuid),
                    labels=list(brain.labels(node_uuid)),
                    description=data.get("description"),
                    properties=properties,
                )
            )
        return SearchEntitiesResult(results=nodes, total=total)

    def search_nodes_bm25(
        self,
        query_text: str,
        brain_id: str,
        *,
        limit: int = 10,
        node_labels: Optional[list[str]] = None,
        node_uuids: Optional[list[str]] = None,
    ) -> list[tuple[Node, float]]:
        rows = self._store.search_nodes_bm25(
            query_text,
            brain_id,
            limit=limit,
            node_labels=node_labels,
            node_uuids=node_uuids,
        )
        hits: list[tuple[Node, float]] = []
        for node_uuid, score, data in rows:
            labels = [str(item) for item in (data.get("labels") or []) if str(item).strip()]
            properties = {k: v for k, v in data.items() if k != "labels"}
            hits.append(
                (
                    Node(
                        uuid=str(node_uuid),
                        name=str(data.get("name") or node_uuid),
                        labels=labels,
                        description=data.get("description"),
                        properties=properties,
                    ),
                    float(score),
                )
            )
        return hits

    def deprecate_relationship(
        self,
        subject: Node,
        predicate: Predicate,
        object: Node,
        brain_id: str,
    ) -> Tuple[Node, Predicate, Node] | None:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        source_uuid = self._store.resolve_node_by_name_labels(
            brain, self._clean_labels(subject.labels), subject.name
        )
        target_uuid = self._store.resolve_node_by_name_labels(
            brain, self._clean_labels(object.labels), object.name
        )
        if not source_uuid or not target_uuid:
            return None
        rel_type = self._clean_labels([predicate.name])[0]
        for source, target, key, data in brain.graph.edges(keys=True, data=True):
            if source != source_uuid or target != target_uuid:
                continue
            if data.get("rel_type") != rel_type:
                continue
            data["deprecated"] = True
            self._store._persist_relationship(
                brain_id, key, rel_type, source, target, data
            )
            return (
                Node(
                    uuid=source_uuid,
                    name=subject.name,
                    labels=subject.labels,
                    description=subject.description,
                    properties=brain.node_data(source_uuid),
                ),
                Predicate(
                    name=rel_type,
                    description=data.get("description", "") or "",
                    direction="out",
                ),
                Node(
                    uuid=target_uuid,
                    name=object.name,
                    labels=object.labels,
                    description=object.description,
                    properties=brain.node_data(target_uuid),
                ),
            )
        return None


    def update_properties(
        self,
        uuid: str,
        updating: Literal["node", "relationship"],
        brain_id: str,
        new_properties: dict,
        properties_to_remove: list[str],
    ) -> Node | Predicate | None:
        self._store.ensure_database(brain_id)
        record = self._store.update_entity_properties(
            brain_id,
            uuid,
            updating == "relationship",
            new_properties,
            properties_to_remove,
        )
        if not record:
            return None
        if updating == "node":
            return Node(
                uuid=record.get("uuid", "") or "",
                name=record.get("name", "") or "",
                labels=record.get("labels", []) or [],
                description=record.get("description", "") or "",
                properties=record.get("properties", {}) or {},
            )
        return Predicate(
            name=record.get("rel_type", "") or "",
            description=record.get("rel_description", "") or "",
            direction=record.get("direction", "neutral"),
        )


    def get_graph_relationship_types(self, brain_id: str) -> list[str]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        return self._store.list_relationship_types(brain)


    def get_graph_node_types(self, brain_id: str) -> list[str]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        return self._store.list_labels(brain)


    def get_graph_node_properties(self, brain_id: str) -> list[str]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        return self._store.list_node_property_keys(brain)


    def update_node(
        self,
        uuid: str,
        brain_id: str,
        new_name: Optional[str] = None,
        new_description: Optional[str] = None,
        new_labels: Optional[list[str]] = None,
        new_properties: Optional[dict] = None,
        properties_to_remove: Optional[list[str]] = None,
    ) -> Node | None:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        if uuid not in brain.graph:
            return None
        existing_node = self.get_by_uuid(uuid, brain_id)
        if (
            not new_name
            and not new_description
            and not new_labels
            and not new_properties
            and not properties_to_remove
        ):
            return existing_node
        data = brain.node_data(uuid)
        if new_name:
            data["name"] = new_name
        if new_description:
            data["description"] = new_description
        if new_labels:
            data["labels"] = self._clean_labels(new_labels)
        if new_properties:
            existing_properties = (existing_node.properties or {}) if existing_node else {}
            for property_key, new_value in new_properties.items():
                key_str = self._clean_property_key(property_key)
                if property_key not in existing_properties or existing_properties.get(property_key) != new_value:
                    data[key_str] = new_value
        if properties_to_remove:
            for prop in properties_to_remove:
                data.pop(self._clean_property_key(prop), None)
        self._store._persist_node(brain_id, uuid, data)
        record = self._store.node_to_record(brain, uuid)
        return Node(
            uuid=record.get("uuid", "") or "",
            name=record.get("name", "") or "",
            labels=record.get("labels", []) or [],
            description=record.get("description", "") or "",
            properties=record.get("properties", {}) or {},
        )


    def get_schema(self, brain_id: str) -> dict:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        return {
            "labels": self._store.list_labels(brain),
            "relationships": self._store.list_relationship_types(brain),
            "event_names": self._store.event_names(brain),
        }


    def get_2nd_degree_hops(
        self,
        from_: List[str],
        flattened: bool,
        vector_store_adapter: VectorStoreClient,
        brain_id: str,
    ) -> Dict[str, List[Tuple[Predicate, Node, List[Tuple[Predicate, Node]]]]]:
        """
        Retrieve second-degree neighbor nodes from the given starting nodes.

        Parameters:
            from_ (List[str]): List of node UUIDs to start the hop from.
            flattened (bool): Whether to flatten the result structure.
            vector_store_adapter (VectorStoreClient): Adapter to the vector store.
            brain_id (str): Identifier of the brain or graph context.

        Returns:
            Dict[str, List[Tuple[Predicate, Node, List[Tuple[Predicate, Node]]]]]: Mapping from each starting node UUID to a list of tuples containing
                a predicate to a neighbor node, the neighbor node itself, and a list of tuples for that neighbor's predicates and their connected nodes.
        """
        return super().get_2nd_degree_hops(
            from_, flattened, vector_store_adapter, brain_id
        )

    def check_node_existence(
        self,
        uuid: str,
        name: str,
        labels: list[str],
        brain_id: str,
    ) -> bool:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        return self._store.check_node_exists(
            brain, uuid, name, self._clean_labels(labels)
        )


    def get_neighborhood(
        self, node: Node | str, depth: int, brain_id: str
    ) -> list[dict]:
        if depth < 1:
            return []
        node_uuid = node.uuid if isinstance(node, Node) else node
        return self._get_neighborhood_recursive(node_uuid, depth, brain_id, set())


    def _get_neighborhood_recursive(
        self, node_uuid: str, depth: int, brain_id: str, path_visited: set[str]
    ) -> list[dict]:
        if depth < 1 or node_uuid in path_visited:
            return []
        path_visited = path_visited.copy()
        path_visited.add(node_uuid)
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        records = self._store.neighborhood_records(brain, node_uuid)
        neighbors = []
        for record in records:
            neighbor_node_uuid = record.get("m_uuid", "")
            if neighbor_node_uuid in path_visited:
                continue
            neighbor_node = Node(
                uuid=neighbor_node_uuid,
                name=record.get("m_name", "") or "",
                labels=record.get("m_labels", []) or [],
                description=record.get("m_description", "") or "",
                properties=always_dict(record.get("m_properties", {})),
                polarity=record.get("m_polarity", "neutral"),
                metadata=always_dict(record.get("m_metadata", {})),
                happened_at=record.get("m_happened_at", "") or "",
                last_updated=record.get("m_last_updated", "") or "",
                observations=record.get("m_observations", []) or [],
            )
            predicate = Predicate(
                name=record.get("rel_type", "") or "",
                description=record.get("rel_description", "") or "",
                direction=record.get("direction", "neutral"),
                properties=always_dict(record.get("rel_properties", {})),
                flow_key=record.get("rel_flowkey", "") or "",
                uuid=record.get("rel_uuid", "") or "",
                last_updated=record.get("rel_last_updated", "") or "",
                observations=record.get("rel_observations", []) or [],
                amount=record.get("rel_amount"),
            )
            nested_neighbors = self._get_neighborhood_recursive(
                neighbor_node_uuid, depth - 1, brain_id, path_visited
            )
            neighbors.append(
                {
                    "predicate": predicate,
                    "node": neighbor_node,
                    "neighbors": nested_neighbors,
                }
            )
        return neighbors


    def _node_from_brain(self, brain, uuid: str) -> Node:
        data = brain.node_data(uuid) if uuid in brain.graph else {}
        node_kwargs: Dict[str, Any] = {
            "uuid": uuid,
            "name": data.get("name", "") or "",
            "labels": data.get("labels", []) or [],
            "description": data.get("description", "") or "",
            "properties": always_dict(data),
            "polarity": data.get("polarity", "neutral") or "neutral",
            "metadata": always_dict(data.get("metadata", {})),
            "happened_at": data.get("happened_at", "") or "",
            "observations": data.get("observations", []) or [],
        }
        last_updated = data.get("last_updated")
        if last_updated:
            node_kwargs["last_updated"] = last_updated
        return Node(**node_kwargs)


    def _predicate_from_edge(self, edge_data: dict, direction: str) -> Predicate:
        edge_data = always_dict(edge_data)
        predicate_kwargs: Dict[str, Any] = {
            "uuid": edge_data.get("uuid", "") or "",
            "name": edge_data.get("rel_type", "") or edge_data.get("name", "") or "",
            "description": edge_data.get("description", "") or "",
            "direction": direction,
            "properties": always_dict(edge_data),
            "flow_key": edge_data.get("flow_key", "") or "",
            "observations": edge_data.get("observations", []) or [],
            "amount": edge_data.get("amount"),
        }
        last_updated = edge_data.get("last_updated")
        if last_updated:
            predicate_kwargs["last_updated"] = last_updated
        return Predicate(**predicate_kwargs)


    def _incident_edges(
        self, brain, node_uuid: str
    ) -> List[Tuple[str, str, dict, str]]:
        if node_uuid not in brain.graph:
            return []
        edges: List[Tuple[str, str, dict, str]] = []
        for _, neighbor, key, data in brain.graph.out_edges(
            node_uuid, keys=True, data=True
        ):
            edges.append((neighbor, key, dict(data), "out"))
        for neighbor, _, key, data in brain.graph.in_edges(
            node_uuid, keys=True, data=True
        ):
            edges.append((neighbor, key, dict(data), "in"))
        return edges


    def _event_path_records(
        self,
        brain,
        node_uuids: list[str],
        predicate_uuid: Optional[str] = None,
        flow_key: Optional[str] = None,
    ):
        incident: dict[str, list[Tuple[str, str, dict, str]]] = {}
        by_flow: dict[str, dict[str, list[Tuple[str, str, dict, str]]]] = {}

        def edges_for(uuid: str) -> list[Tuple[str, str, dict, str]]:
            if uuid not in incident:
                incident[uuid] = self._incident_edges(brain, uuid)
            return incident[uuid]

        for n_uuid in node_uuids:
            if n_uuid not in brain.graph:
                continue
            for m_uuid, r1_key, r1_data, r1_direction in edges_for(n_uuid):
                if (
                    predicate_uuid is not None
                    and (r1_data.get("uuid") or r1_key) != predicate_uuid
                ):
                    continue
                required_flow_key = (
                    flow_key if flow_key is not None else r1_data.get("flow_key")
                )
                if not required_flow_key:
                    continue
                if m_uuid not in by_flow:
                    grouped: dict[str, list[Tuple[str, str, dict, str]]] = {}
                    for edge in edges_for(m_uuid):
                        edge_flow = edge[2].get("flow_key")
                        if isinstance(edge_flow, str) and edge_flow:
                            grouped.setdefault(edge_flow, []).append(edge)
                    by_flow[m_uuid] = grouped
                matching_edges = (
                    by_flow[m_uuid].get(required_flow_key, ())
                    if isinstance(required_flow_key, str)
                    else (
                        edge for edge in edges_for(m_uuid)
                        if edge[2].get("flow_key") == required_flow_key
                    )
                )
                for b_uuid, r2_key, r2_data, r2_direction in matching_edges:
                    if r2_key == r1_key:
                        continue
                    yield (
                        n_uuid,
                        m_uuid,
                        b_uuid,
                        r1_data,
                        r2_data,
                        r1_direction,
                        r2_direction,
                    )


    def get_event_centric_neighbors(
        self,
        nodes: list[Node | str],
        brain_id: str,
    ) -> List[Tuple[Node, Predicate, Node, Predicate, Node]]:
        if len(nodes) == 0:
            return []
        node_uuids = nodes if isinstance(nodes[0], str) else [node.uuid for node in nodes]
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        results = []
        for n_uuid, m_uuid, b_uuid, r1_data, r2_data, r_direction, r2_direction in self._event_path_records(brain, node_uuids):
            results.append(
                (
                    self._node_from_brain(brain, n_uuid),
                    self._predicate_from_edge(r1_data, r_direction),
                    self._node_from_brain(brain, m_uuid),
                    self._predicate_from_edge(r2_data, r2_direction),
                    self._node_from_brain(brain, b_uuid),
                )
            )
        return results


    def get_nexts_by_flow_key(
        self, predicates: list[PredicateWithFlowKey], brain_id: str
    ) -> Dict[str, List[Tuple[Node, Predicate, Node]]]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        res: Dict[str, List[Tuple[Node, Predicate, Node]]] = {}
        for predicate in predicates:
            paths = self._event_path_records(
                brain,
                list(brain.graph.nodes),
                predicate_uuid=predicate["predicate_uuid"],
                flow_key=predicate["flow_key"],
            )
            res[predicate["predicate_uuid"]] = [
                (
                    self._node_from_brain(brain, m_uuid),
                    self._predicate_from_edge(r2_data, "out"),
                    self._node_from_brain(brain, b_uuid),
                )
                for _, m_uuid, b_uuid, _, r2_data, _, _ in paths
            ]
        return res


    def get_triples_by_uuid(
        self, uuids: list[str], brain_id: str
    ) -> List[Tuple[Node, Predicate, Node]]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        uuid_set = set(uuids)
        triples = []
        for source, target, key, edge_data in brain.graph.edges(keys=True, data=True):
            if key not in uuid_set and edge_data.get("uuid") not in uuid_set:
                continue
            triples.append(
                (
                    Node(
                        uuid=source,
                        name=brain.node_data(source).get("name", "") or "",
                        labels=brain.labels(source),
                        description=brain.node_data(source).get("description", "") or "",
                        properties=brain.node_data(source),
                    ),
                    Predicate(
                        name=edge_data.get("rel_type", "") or "",
                        description=edge_data.get("description", "") or "",
                        direction="neutral",
                        properties=dict(edge_data),
                    ),
                    Node(
                        uuid=target,
                        name=brain.node_data(target).get("name", "") or "",
                        labels=brain.labels(target),
                        description=brain.node_data(target).get("description", "") or "",
                        properties=brain.node_data(target),
                    ),
                )
            )
        return triples


    def remove_nodes(self, uuids: list[str], brain_id: str) -> list[Node]:
        self._store.ensure_database(brain_id)
        records = self._store.delete_nodes_by_uuids(brain_id, uuids)
        return [Node(**record.get("node", {})) for record in records]


    def remove_relationships(
        self,
        relationships: list[Tuple[NodeDict, PredicateDict, NodeDict]],
        brain_id: str,
    ) -> list[Tuple[Node, Predicate, Node]]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        deleted: list[Tuple[Node, Predicate, Node]] = []
        by_rel_uuid: list[str] = []
        by_tip_tail: list[Tuple[NodeDict, PredicateDict, NodeDict]] = []
        for tail, pred, head in relationships:
            if pred.get("uuid"):
                by_rel_uuid.append(pred["uuid"])
            else:
                by_tip_tail.append((tail, pred, head))
        if by_rel_uuid:
            records = self._store.delete_relationships_by_uuids(brain_id, by_rel_uuid)
            for record in records:
                deleted.append(self._record_to_triple(record))
        for tail, pred, head in by_tip_tail:
            tail_uuid = self._resolve_node_dict(brain, tail)
            head_uuid = self._resolve_node_dict(brain, head)
            rel_type = pred.get("name")
            if not tail_uuid or not head_uuid:
                continue
            for source, target, key, edge_data in list(brain.graph.edges(keys=True, data=True)):
                if {source, target} != {tail_uuid, head_uuid}:
                    continue
                if rel_type and edge_data.get("rel_type") != self._clean_labels([rel_type])[0]:
                    continue
                record = self._store.node_to_record(brain, source, "n")
                record.update(self._store.relationship_to_record(brain, source, target, key, source))
                record.update(self._store.node_to_record(brain, target, "m"))
                deleted.append(self._record_to_triple(record))
                self._store._delete_relationship(brain_id, key)
        return deleted


    def _record_to_triple(self, record: Any) -> Tuple[Node, Predicate, Node]:
        return (
            Node(
                uuid=record.get("n_uuid", "") or "",
                name=record.get("n_name", "") or "",
                labels=record.get("n_labels", []) or [],
                description=record.get("n_description", "") or "",
                properties=record.get("n_properties", {}) or {},
            ),
            Predicate(
                uuid=record.get("r_uuid", "") or "",
                name=record.get("r_type", "") or "",
                description=record.get("r_description", "") or "",
                direction="neutral",
                properties=record.get("r_properties", {}) or {},
            ),
            Node(
                uuid=record.get("m_uuid", "") or "",
                name=record.get("m_name", "") or "",
                labels=record.get("m_labels", []) or [],
                description=record.get("m_description", "") or "",
                properties=record.get("m_properties", {}) or {},
            ),
        )

    def list_relationships(
        self, subject: str, object: str, brain_id: str
    ) -> list[Tuple[Node, Predicate, Node]]:
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        triples = []
        for source, target, key, edge_data in brain.graph.edges(keys=True, data=True):
            if {source, target} != {subject, object}:
                continue
            triples.append(
                (
                    Node(
                        uuid=source,
                        name=brain.node_data(source).get("name", "") or "",
                        labels=brain.labels(source),
                        description=brain.node_data(source).get("description", "") or "",
                        properties=brain.node_data(source),
                        polarity=brain.node_data(source).get("polarity", "neutral"),
                        metadata=brain.node_data(source).get("metadata", {}) or {},
                        happened_at=brain.node_data(source).get("happened_at", "") or "",
                        last_updated=brain.node_data(source).get("last_updated", "") or "",
                        observations=brain.node_data(source).get("observations", []) or [],
                    ),
                    Predicate(
                        uuid=edge_data.get("uuid", key) or "",
                        name=edge_data.get("rel_type", "") or "",
                        description=edge_data.get("description", "") or "",
                        direction="neutral",
                        properties=dict(edge_data),
                        flow_key=edge_data.get("flow_key", "") or "",
                        last_updated=edge_data.get("last_updated", "") or "",
                        observations=edge_data.get("observations", []) or [],
                        amount=edge_data.get("amount"),
                    ),
                    Node(
                        uuid=target,
                        name=brain.node_data(target).get("name", "") or "",
                        labels=brain.labels(target),
                        description=brain.node_data(target).get("description", "") or "",
                        properties=brain.node_data(target),
                        polarity=brain.node_data(target).get("polarity", "neutral"),
                        metadata=brain.node_data(target).get("metadata", {}) or {},
                        happened_at=brain.node_data(target).get("happened_at", "") or "",
                        last_updated=brain.node_data(target).get("last_updated", "") or "",
                        observations=brain.node_data(target).get("observations", []) or [],
                    ),
                )
            )
        return triples


    def get_event_hub_facts(
        self,
        event_uuids: list[str],
        brain_id: str,
    ) -> List[Tuple[Node, Predicate, Node, Predicate, Node]]:
        unique = sorted({str(u) for u in event_uuids if u})
        if not unique:
            return []
        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        event_set = set(unique)
        neighbor_seeds: list[str] = []
        seen_seeds: set[str] = set()
        for event_uuid in unique:
            for neighbor, _key, _data, _direction in self._incident_edges(
                brain, event_uuid
            ):
                if neighbor in seen_seeds or neighbor in event_set:
                    continue
                seen_seeds.add(neighbor)
                neighbor_seeds.append(neighbor)
        neighbor_seeds.sort()
        results = []
        seen_keys: set[tuple[str, str]] = set()
        for (
            n_uuid,
            m_uuid,
            b_uuid,
            r1_data,
            r2_data,
            r_direction,
            r2_direction,
        ) in self._event_path_records(brain, neighbor_seeds):
            if m_uuid not in event_set:
                continue
            key = (
                str(r1_data.get("uuid") or ""),
                str(r2_data.get("uuid") or ""),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            results.append(
                (
                    self._node_from_brain(brain, n_uuid),
                    self._predicate_from_edge(r1_data, r_direction),
                    self._node_from_brain(brain, m_uuid),
                    self._predicate_from_edge(r2_data, r2_direction),
                    self._node_from_brain(brain, b_uuid),
                )
            )
        results.sort(
            key=lambda row: (
                str(getattr(row[2], "uuid", "") or ""),
                str(getattr(row[1], "uuid", "") or ""),
                str(getattr(row[3], "uuid", "") or ""),
            )
        )
        return results

    def rebuild_hub_bridge_index(self, brain_id: str) -> int:
        from src.core.saving.hub_bridges import (
            bridges_from_memberships,
            entity_event_memberships,
            is_event_labels,
        )

        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        rows: list[tuple[str, str, str, str]] = []
        for source, target, _key, edge_data in brain.graph.edges(keys=True, data=True):
            rel_name = str(edge_data.get("rel_type") or edge_data.get("name") or "")
            source_labels = brain.labels(source)
            target_labels = brain.labels(target)
            source_is_event = is_event_labels(source_labels)
            target_is_event = is_event_labels(target_labels)
            if source_is_event and not target_is_event:
                rows.append(
                    (
                        source,
                        target,
                        str(brain.node_data(target).get("name") or ""),
                        rel_name,
                    )
                )
            elif target_is_event and not source_is_event:
                rows.append(
                    (
                        target,
                        source,
                        str(brain.node_data(source).get("name") or ""),
                        rel_name,
                    )
                )
        bridges = bridges_from_memberships(entity_event_memberships(rows))
        payload = [
            (
                br.event_a,
                br.event_b,
                br.shared_entity,
                br.shared_entity_name,
                br.weight,
            )
            for br in bridges
        ]
        return self._store.replace_hub_bridges(brain_id, payload)

    def refresh_hub_bridges_for_entities(
        self, brain_id: str, entity_uuids: list[str]
    ) -> int:
        from src.core.saving.hub_bridges import (
            bridges_from_memberships,
            entity_event_memberships,
            is_event_labels,
        )

        touched = sorted({str(u) for u in entity_uuids if u})
        if not touched:
            return self.rebuild_hub_bridge_index(brain_id)

        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        touched_set = set(touched)
        rows: list[tuple[str, str, str, str]] = []
        for source, target, _key, edge_data in brain.graph.edges(keys=True, data=True):
            rel_name = str(edge_data.get("rel_type") or edge_data.get("name") or "")
            source_labels = brain.labels(source)
            target_labels = brain.labels(target)
            source_is_event = is_event_labels(source_labels)
            target_is_event = is_event_labels(target_labels)
            if source_is_event and not target_is_event and target in touched_set:
                rows.append(
                    (
                        source,
                        target,
                        str(brain.node_data(target).get("name") or ""),
                        rel_name,
                    )
                )
            elif target_is_event and not source_is_event and source in touched_set:
                rows.append(
                    (
                        target,
                        source,
                        str(brain.node_data(source).get("name") or ""),
                        rel_name,
                    )
                )
        bridges = bridges_from_memberships(entity_event_memberships(rows))
        payload = [
            (
                br.event_a,
                br.event_b,
                br.shared_entity,
                br.shared_entity_name,
                br.weight,
            )
            for br in bridges
        ]
        self._store.delete_hub_bridges_for_entities(brain_id, touched)
        return self._store.upsert_hub_bridges(brain_id, payload)

    def get_hub_bridges(
        self,
        event_uuids: list[str],
        brain_id: str,
    ) -> list[dict]:
        from src.core.saving.hub_bridges import HubBridge

        rows = self._store.get_hub_bridges_for_events(brain_id, event_uuids)
        return [
            HubBridge(
                event_a=a,
                event_b=b,
                shared_entity=entity,
                shared_entity_name=name,
                weight=weight,
            )
            for a, b, entity, name, weight in rows
        ]

    def rebuild_topic_index(self, brain_id: str) -> int:
        import re

        from src.core.saving.hub_bridges import (
            is_context_rel,
            is_event_labels,
            is_spine_rel,
        )
        from src.core.saving.topic_hyperedges import (
            cluster_sessions_into_topics,
            session_entity_sets,
        )

        self._store.ensure_database(brain_id)
        brain = self._store.get_brain(brain_id)
        session_re = re.compile(
            r"session_(\d+)|(?:\b|_)b(\d+)_t(\d+)\b",
            re.IGNORECASE,
        )

        def sessions_from_text(text: str) -> list[str]:
            found: list[str] = []
            seen: set[str] = set()
            for match in session_re.finditer(text or ""):
                if match.group(1):
                    sid = f"session_{match.group(1)}"
                else:
                    sid = f"session_b{match.group(2)}_t{match.group(3)}"
                if sid not in seen:
                    seen.add(sid)
                    found.append(sid)
            return found

        chunk_ids: list[str] = []
        event_chunks: dict[str, list[str]] = {}
        for node_uuid in brain.graph.nodes:
            if not is_event_labels(brain.labels(node_uuid)):
                continue
            props = brain.node_data(node_uuid)
            ids = [
                str(x)
                for x in (props.get("source_chunk_ids") or [])
                if x
            ]
            single = props.get("source_chunk_id")
            if single:
                ids.append(str(single))
            ids = sorted(set(ids))
            event_chunks[node_uuid] = ids
            chunk_ids.extend(ids)

        chunk_sessions: dict[str, list[str]] = {}
        if chunk_ids:
            try:
                from src.services.data.main import data_adapter

                chunks, _ = data_adapter.get_text_chunks_by_ids(
                    sorted(set(chunk_ids))[:500],
                    False,
                    brain_id,
                )
                for chunk in chunks:
                    cid = str(getattr(chunk, "id", "") or "")
                    body = getattr(chunk, "text", "") or ""
                    if cid:
                        chunk_sessions[cid] = sessions_from_text(body)
            except Exception:
                chunk_sessions = {}

        rows: list[tuple[str, str, str]] = []
        for source, target, _key, edge_data in brain.graph.edges(keys=True, data=True):
            rel_name = str(edge_data.get("rel_type") or edge_data.get("name") or "")
            if is_context_rel(rel_name) or not is_spine_rel(rel_name):
                continue
            source_is_event = is_event_labels(brain.labels(source))
            target_is_event = is_event_labels(brain.labels(target))
            if source_is_event and not target_is_event:
                event_uuid, entity_uuid = source, target
            elif target_is_event and not source_is_event:
                event_uuid, entity_uuid = target, source
            else:
                continue
            entity_name = str(brain.node_data(entity_uuid).get("name") or "")
            sessions: list[str] = []
            for cid in event_chunks.get(event_uuid, []):
                sessions.extend(chunk_sessions.get(cid, []))
            if not sessions:
                props = brain.node_data(event_uuid)
                sessions = sessions_from_text(
                    " ".join(
                        str(props.get(k) or "")
                        for k in ("name", "description", "summary")
                    )
                )
            for sid in sessions:
                rows.append((sid, entity_uuid, entity_name))

        memberships = cluster_sessions_into_topics(session_entity_sets(rows))
        payload = [
            (m.topic_id, m.topic_label, m.session_id, m.weight) for m in memberships
        ]
        return self._store.replace_topic_sessions(brain_id, payload)

    def list_topic_memberships(self, brain_id: str) -> list:
        from src.core.saving.topic_hyperedges import TopicSession

        rows = self._store.list_topic_sessions(brain_id)
        return [
            TopicSession(
                topic_id=tid,
                topic_label=label,
                session_id=sid,
                weight=weight,
            )
            for tid, label, sid, weight in rows
        ]


def _records(data: list | Any) -> Any:
    if hasattr(data, "records"):
        return data
    class _R:
        def __init__(self, records):
            self.records = records
    return _R(data if isinstance(data, list) else [])


_networkx_graph_client: Optional[NetworkXGraphClient] = None


def get_networkx_graph_client() -> NetworkXGraphClient:
    global _networkx_graph_client
    if _networkx_graph_client is None:
        _networkx_graph_client = NetworkXGraphClient()
    return _networkx_graph_client

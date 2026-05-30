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
from src.utils.serialization.data import always_dict


class NetworkXGraphClient(GraphClient):
    """
    Graph client backed by PostgreSQL persistence and NetworkX in-memory traversal.
    """

    def __init__(self):
        self._store = PostgreSQLGraphStore()

    def ensure_database(self, database: str) -> None:
        self._store.ensure_database(database)

    def execute_operation(self, operation: str, brain_id: str) -> Any:
        raise GraphDatabaseError(
            "Ad-hoc Cypher operations are only supported with the Neo4j backend (GRAPH_DB=neo4j). "
            "Use the native PostgreSQL+NetworkX graph API instead."
        )

    @property
    def graphdb_type(self) -> str:
        return "postgresql-networkx"

    @property
    def graphdb_description(self) -> str:
        return (
            "The graph database is PostgreSQL with NetworkX-backed in-memory traversal. "
            "Use the native graph API methods; Cypher is not supported on this backend."
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
            identification_dict: dict[str, Any] = {"name": node.name}
            merged_metadata = {**(metadata or {}), **(node.metadata or {})}
            all_properties: dict[str, Any] = {
                **(node.properties or {}),
                "metadata": merged_metadata or None,
            }
            if identification_params:
                for key, value in identification_params.items():
                    normalized_key = self._clean_property_key(key)
                    if normalized_key != "name":
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
        for n_uuid, m_uuid, b_uuid, r1_data, r2_data, _, _ in self._event_path_records(brain, node_uuids):
            r_direction = "out"
            r2_direction = "out"
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

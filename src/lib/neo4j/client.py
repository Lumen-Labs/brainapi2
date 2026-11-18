"""
File: /client.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday February 2nd 2026 10:02:37 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from datetime import datetime, timezone
import json
import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict
from neo4j import GraphDatabase, Query, READ_ACCESS, WRITE_ACCESS
from neo4j.exceptions import ClientError
from src.adapters.interfaces.embeddings import VectorStoreClient
from src.adapters.interfaces.graph import GraphClient, PredicateWithFlowKey
from src.config import config
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
from src.lib.cypher_safety import (
    MAX_RAW_QUERY_ROWS,
    RAW_QUERY_TIMEOUT_SECONDS,
    clean_label,
    clean_property_key,
    create_database_query,
    optional_node_labels,
    quote_identifier,
    quote_rel_types_pattern,
    relationship_changed_in_window_clause,
    require_database_name,
    require_limit,
    require_path_depth,
    static_property_key,
    validate_raw_cypher,
)
from src.utils.logging import log
from src.utils.serialization.data import always_dict


def _jsonable_neo4j_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable_neo4j_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_neo4j_value(item) for item in value]
    iso = getattr(value, "isoformat", None)
    if callable(iso) and not isinstance(value, str):
        try:
            text = iso()
        except TypeError:
            return value
        if isinstance(text, str):
            return text.replace("+00:00", "Z") if text.endswith("+00:00") else text
    return value


def _is_homogeneous_neo4j_list(items: list) -> bool:
    """True when every item is the same non-null Neo4j primitive type."""
    if not items:
        return True
    kind = type(items[0])
    if kind not in (bool, int, float, str):
        return False
    return all(type(item) is kind for item in items)


class _RawQueryResult:
    def __init__(self, records: list[Any], keys: list[str], truncated: bool):
        self.records = records
        self._keys = keys
        self.truncated = truncated

    def keys(self) -> list[str]:
        return self._keys


class Neo4jClient(GraphClient):
    """
    Neo4j client with support for multiple databases.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            f"bolt://{config.neo4j.host}:{config.neo4j.port}",
            auth=(config.neo4j.username, config.neo4j.password),
            connection_timeout=30,
            max_connection_lifetime=300,
            connection_acquisition_timeout=60,
            warn_notification_severity="OFF",
            notifications_min_severity="OFF",
        )

    def execute_operation(self, operation: str, brain_id: str) -> Any:
        """
        Execute caller-supplied Cypher against the caller's brain database.

        Default is a READ access-mode session with a timeout and result cap so
        writes, DELETE, and admin procedures fail. GRAPH_RAW_WRITE_ENABLED=true
        switches the session to WRITE while still confining the query to this
        brain and rejecting USE / database admin / CALL dbms.*.
        """
        db = require_database_name(brain_id)
        query_text = validate_raw_cypher(operation)
        write_enabled = bool(getattr(config, "graph_raw_write_enabled", False))
        access_mode = WRITE_ACCESS if write_enabled else READ_ACCESS
        records: list[Any] = []
        keys: list[str] = []
        truncated = False
        with self.driver.session(
            database=db,
            default_access_mode=access_mode,
        ) as session:
            result = session.run(
                Query(query_text, timeout=RAW_QUERY_TIMEOUT_SECONDS)
            )
            keys = list(result.keys())
            for index, record in enumerate(result):
                if index >= MAX_RAW_QUERY_ROWS:
                    truncated = True
                    break
                records.append(record)
        return _RawQueryResult(records=records, keys=keys, truncated=truncated)

    def ensure_database(self, database: str) -> None:
        """
        Ensure a database exists.
        """
        database = require_database_name(database)
        if self._verify_database_accessible(database):
            return

        try:
            cypher_query = create_database_query(database)
            self.driver.execute_query(cypher_query, database_="system")
        except Exception as e:
            error_msg = str(e).lower()
            if (
                "already exists" in error_msg
                or "not supported in community edition" in error_msg
                or "unsupported administration command" in error_msg
            ):
                pass
            else:
                raise

        retries = 0
        max_wait_retries = 30
        while retries < max_wait_retries:
            if self._verify_database_accessible(database):
                return
            time.sleep(0.2)
            retries += 1

        if not self._verify_database_accessible(database):
            raise RuntimeError(
                f"Database '{database}' could not be created or is not accessible. "
                "If using Neo4j Community Edition, you can only use the default 'neo4j' database."
            )

    def clear_brain_data(self, brain_id: str) -> None:
        if not self._verify_database_accessible(brain_id):
            return
        self.driver.execute_query(
            "MATCH (node) DETACH DELETE node",
            database_=brain_id,
        )

    def _verify_database_accessible(self, database: str) -> bool:
        """
        Verify that a database is accessible by executing a simple query.
        """
        database = require_database_name(database)
        try:
            self.driver.execute_query("RETURN 1 AS test", database_=database)
            return True
        except ClientError as e:
            error_code = getattr(e, "code", None)
            if error_code == "Neo.ClientError.Database.DatabaseNotFound":
                return False
            raise

    def _execute_query_with_retry(
        self,
        query: str,
        database: str,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        parameters: Optional[dict] = None,
    ):
        """
        Execute a query with retry logic for DatabaseNotFound errors.
        Verifies database accessibility instead of using sleep.
        """
        last_exception = None
        params = parameters or {}
        for attempt in range(max_retries):
            try:
                return self.driver.execute_query(
                    query, parameters_=params, database_=database
                )
            except ClientError as e:
                error_code = getattr(e, "code", None)
                if error_code == "Neo.ClientError.Database.DatabaseNotFound":
                    last_exception = e
                    if attempt < max_retries - 1:
                        self.ensure_database(database)
                        retries = 0
                        while retries < 10 and not self._verify_database_accessible(
                            database
                        ):
                            time.sleep(retry_delay)
                            retries += 1
                        continue
                raise
        if last_exception:
            raise last_exception

    def _run_query(
        self,
        query: str,
        brain_id: str,
        parameters: Optional[dict] = None,
        *,
        retry: bool = False,
        ensure: bool = True,
    ):
        if ensure:
            self.ensure_database(brain_id)
        else:
            require_database_name(brain_id)
        params = parameters or {}
        if retry:
            return self._execute_query_with_retry(query, brain_id, parameters=params)
        return self.driver.execute_query(
            query, parameters_=params, database_=brain_id
        )

    @property
    def graphdb_type(self) -> str:
        """
        Get the type of graph database.
        """
        return "neo4j"

    @property
    def graphdb_description(self) -> str:
        """
        Get the description of the graph database.
        It is used to let the agent know which syntax to use.
        """
        return (
            "The graph database is Neo4j. Cypher is the language used to operate with it. "
            "Raw queries are read-only against the caller's brain database, with a timeout "
            "and result-size cap. Writes, DELETE, USE, CALL dbms.*, and database admin "
            "are rejected. Use structured graph tools to mutate the graph."
        )

    def _clean_labels(self, labels: list[str]) -> list[str]:
        """
        Clean labels to be used in a Cypher query.
        """
        return [clean_label(label) for label in labels]

    def _quoted_labels(self, labels: list[str]) -> str:
        return ":".join(
            quote_identifier(label, kind="label") for label in self._clean_labels(labels)
        )

    def _quoted_rel_type(self, name: str) -> str:
        cleaned = self._clean_labels([name])[0]
        return quote_identifier(cleaned, kind="relationship type")

    def _clean_property_key(self, property_key: str) -> str:
        """
        Clean a property key to be used in a Cypher query.
        """
        return clean_property_key(property_key)

    def _format_property_key(self, property_key: str) -> str:
        """
        Format a property key for use in a Cypher query, including sanitization and quoting.
        """
        return quote_identifier(self._clean_property_key(property_key), kind="property key")

    def _format_value(self, value: Any) -> str:
        """Deprecated: values must be passed as query parameters, not interpolated."""
        raise RuntimeError(
            "Cypher values must be passed as query parameters; do not interpolate them"
        )

    def _bind_property_value(self, value: Any) -> Any:
        """Coerce a value to a Neo4j-storable property (no nested maps)."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, datetime):
            return value
        if isinstance(value, (list, tuple)):
            items = list(value)
            if _is_homogeneous_neo4j_list(items):
                return items
            return json.dumps(items, default=str)
        if isinstance(value, dict):
            return json.dumps(value, default=str)
        try:
            return json.dumps(value, default=str)
        except TypeError:
            return str(value)

    def _neo4j_property_value(self, value: Any) -> Any:
        """Bind a property, storing datetimes as canonical ISO UTC strings."""
        if isinstance(value, datetime):
            from src.core.saving.validity import format_instant

            return format_instant(value)
        return self._bind_property_value(value)

    def _relationship_property_primitives(
        self, properties: Optional[dict]
    ) -> dict[str, Any]:
        from src.core.saving.validity import validity_props_from_mapping

        props = dict(properties or {})
        extra: dict[str, Any] = {}
        extra.update(validity_props_from_mapping(props))
        for key in (
            "valid_at",
            "invalid_at",
            "source_timestamp",
            "source_chunk_ids",
            "v_id",
            "deprecated",
        ):
            if props.get(key) not in (None, "", []) and key not in extra:
                extra[key] = self._neo4j_property_value(props[key])
        for key, value in props.items():
            if key in extra or key == "uuid" or value in (None, "", {}, []):
                continue
            extra[self._clean_property_key(key)] = self._neo4j_property_value(value)
        return extra

    def _python_value(self, value: Any) -> Any:
        """Convert Neo4j driver values (temporals, maps) into Python objects."""
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        to_native = getattr(value, "to_native", None)
        if callable(to_native):
            try:
                value = to_native()
            except Exception:
                return str(value)
        if isinstance(value, datetime):
            return value
        if isinstance(value, dict):
            return {str(key): self._python_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._python_value(item) for item in value]
        items = getattr(value, "items", None)
        if callable(items):
            try:
                return {
                    str(key): self._python_value(item) for key, item in items()
                }
            except Exception:
                return str(value)
        return value

    def _python_properties(self, value: Any) -> dict:
        if not value:
            return {}
        if isinstance(value, str):
            value = always_dict(value)
        coerced = self._python_value(value)
        return coerced if isinstance(coerced, dict) else {}

    def add_nodes(
        self,
        nodes: list[Node],
        brain_id: str,
        identification_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> list[Node] | str:
        """
        Insert multiple nodes into the specified brain's graph and return representations of the created nodes.

        Ensures the target database exists, merges or creates each node by its identification properties (at minimum the node's name), sets node properties and standard attributes (description, happened_at, last_updated, metadata, observations, polarity, and uuid), and retries on transient database-not-found errors.

        Parameters:
            identification_params (dict, optional): Additional property keys and values used to identify (MERGE) nodes besides name; keys will be normalized to property-style keys.
            metadata (dict, optional): Metadata to attach to each node; this metadata is merged with any existing node.metadata for the returned Node objects.

        Returns:
            list[Node]: A list of Node objects representing the added nodes. Each returned Node preserves the original node fields and has its metadata set to the merge of the node's own metadata and the provided `metadata`.
        """
        results = []
        self.ensure_database(brain_id)

        for node in nodes:
            if not node.uuid:
                raise ValueError("Node uuid is required for upsert")

            merged_metadata = {**(metadata or {}), **(node.metadata or {})}
            all_properties = {
                **(node.properties or {}),
                "metadata": merged_metadata or None,
                "name": node.name,
            }

            identification_keys: set[str] = set()
            if identification_params:
                for key, value in identification_params.items():
                    normalized_key = self._clean_property_key(key)
                    if normalized_key not in {"uuid", "name"}:
                        all_properties[normalized_key] = value
                        identification_keys.add(normalized_key)

            attributes = [
                "description",
                "happened_at",
                "last_updated",
                "metadata",
                "observations",
                "polarity",
            ]

            for attr in attributes:
                if attr == "metadata":
                    continue
                if getattr(node, attr, None) is not None:
                    all_properties[attr] = getattr(node, attr)

            regular_props: dict[str, Any] = {}
            description_value = None
            list_props: dict[str, Any] = {}
            for key, value in all_properties.items():
                cleaned_key = (
                    key if key in identification_keys else self._clean_property_key(key)
                )
                if key == "description":
                    description_value = value
                elif key in {"source_chunk_ids", "aliases"}:
                    list_props[static_property_key(key)] = self._bind_property_value(
                        value
                    )
                else:
                    regular_props[cleaned_key] = self._bind_property_value(value)

            params: dict[str, Any] = {"uuid": node.uuid, "props": regular_props}
            assignments = ["n += $props"]
            if description_value is not None:
                params["description"] = description_value
                desc = static_property_key("description")
                assignments.append(
                    "n." + desc + " = CASE "
                    "WHEN n." + desc + " IS NULL OR n." + desc + " = '' THEN $description "
                    "WHEN $description IS NULL OR $description = '' THEN n." + desc + " "
                    "WHEN toLower(n." + desc + ") CONTAINS toLower($description) THEN n." + desc + " "
                    "WHEN toLower($description) CONTAINS toLower(n." + desc + ") THEN $description "
                    "ELSE n." + desc + " + ' | ' + $description END"
                )
            for list_key, list_value in list_props.items():
                params[list_key] = list_value
                assignments.append(
                    "n." + list_key + " = CASE "
                    "WHEN n." + list_key + " IS NULL THEN $" + list_key + " "
                    "WHEN $" + list_key + " IS NULL THEN n." + list_key + " "
                    "ELSE n." + list_key + " + [x IN $" + list_key + " WHERE NOT x IN n." + list_key + "] END"
                )

            labels_expression = self._quoted_labels(node.labels)
            cypher_query = (
                "MERGE (n:" + labels_expression + " {uuid: $uuid})\n"
                "SET " + ", ".join(assignments) + "\n"
                "RETURN n"
            )
            try:
                result = self._run_query(
                    cypher_query, brain_id, params, retry=True, ensure=False
                )
                results.append(result)
            except Exception as e:
                print(f"Error adding node: {e} - {cypher_query}")
                raise

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
        """
        Create or update a relationship of the given predicate type between two existing nodes and set relationship attributes.

        Matches the source node by its labels and name (from `subject`) and the target node by its labels and name (from `to_object`), merges a relationship of type `predicate.name` between them, and sets relationship fields (including `uuid`, `description`, `v_id`, and any provided `properties`, `happened_at`, `flow_key`, `last_updated`) from the supplied `predicate`, `subject`, and `to_object` objects. Ensures the target database exists before executing the query.

        Parameters:
                subject (Node): Source node whose labels and `name` are used to find the relationship start.
                predicate (Predicate): Relationship descriptor whose `name` determines the relationship type and whose fields supply relationship attributes to set.
                to_object (Node): Target node whose labels and `name` are used to find the relationship end.
                brain_id (str): Database name to run the query against.

        Returns:
                str: ``ok`` when the relationship was merged.
        """

        objects = [subject, to_object, predicate]
        attributes = [
            "properties",
            "description",
            "happened_at",
            "flow_key",
            "last_updated",
            "amount",
        ]

        if not subject.uuid or not to_object.uuid or not predicate.uuid:
            raise ValueError("Subject, object, and predicate uuids are required for upsert")

        extra_props: dict[str, Any] = {}
        for obj in objects:
            for attr in attributes:
                value = getattr(obj, attr, None)
                if value is None or value == "":
                    continue
                extra_props[static_property_key(attr)] = self._bind_property_value(
                    value
                )
        extra_props.update(self._relationship_property_primitives(predicate.properties))
        extra_props.pop("uuid", None)

        rel_type = self._quoted_rel_type(predicate.name)
        params = {
            "subject_uuid": subject.uuid,
            "object_uuid": to_object.uuid,
            "predicate_uuid": predicate.uuid,
            "description": predicate.description,
            "v_id": (predicate.properties or {}).get("v_id"),
            "flow_key": predicate.flow_key,
            "extra_props": extra_props,
        }
        extra_set = "SET r += $extra_props" if extra_props else ""
        cypher_query = (
            "MATCH (a) WHERE a['uuid'] = $subject_uuid\n"
            "MATCH (b) WHERE b['uuid'] = $object_uuid\n"
            "MERGE (a)-[r:" + rel_type + " {uuid: $predicate_uuid}]->(b)\n"
            "SET r['description'] = $description,\n"
            "    r['v_id'] = $v_id,\n"
            "    r['flow_key'] = $flow_key\n"
            + extra_set
            + "\nRETURN a, b, r"
        )

        self._run_query(cypher_query, brain_id, params)
        return "ok"

    def search_graph(self, nodes: list[Node], brain_id: str) -> list[Node]:
        """
        Search the graph for nodes and 1 degree relationships.
        """
        if not nodes:
            return []

        payload = [
            {"name": node.name, "labels": self._clean_labels(node.labels)}
            for node in nodes
        ]
        cypher_query = (
            "UNWIND $nodes AS node\n"
            "MATCH (n)\n"
            "WHERE n['name'] = node.name AND all(l IN node.labels WHERE l IN labels(n))\n"
            "OPTIONAL MATCH (n)-[r*1]-(m)\n"
            "RETURN n, r, m"
        )
        result = self._run_query(cypher_query, brain_id, {"nodes": payload})
        return result

    def node_text_search(self, text: str, brain_id: str) -> list[Node]:
        """
        Search the graph for nodes by partial text match into the name of the nodes.
        """
        cypher_query = (
            "MATCH (n)\n"
            "WHERE toLower(n['name']) CONTAINS toLower($text)\n"
            "RETURN n"
        )
        result = self._run_query(cypher_query, brain_id, {"text": text})
        return [
            Node(
                uuid=node.get("uuid", "") or "",
                name=node.get("name", "") or "",
                labels=node.get("labels", []) or [],
                description=node.get("description", "") or "",
                properties=node.get("properties", {}) or {},
            )
            for node in result.records
        ]

    def get_nodes_by_uuid(
        self,
        uuids: list[str],
        brain_id: str,
        with_relationships: Optional[bool] = False,
        relationships_depth: Optional[int] = 1,
        relationships_type: Optional[list[str]] = None,
        preferred_labels: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Get nodes by their UUIDs with optional relationships.
        """
        params: dict[str, Any] = {"uuids": list(uuids)}
        cypher_query = "MATCH (n) WHERE n['uuid'] IN $uuids\n"

        if with_relationships:
            depth = require_path_depth(relationships_depth or 1, name="relationships_depth")
            if relationships_type and len(relationships_type) > 0:
                rel_pattern = quote_rel_types_pattern(relationships_type)
                cypher_query += (
                    "OPTIONAL MATCH (n)-[r:"
                    + rel_pattern
                    + "*1.."
                    + str(depth)
                    + "]-(m)\n"
                )
            else:
                cypher_query += (
                    "OPTIONAL MATCH (n)-[r*1.." + str(depth) + "]-(m)\n"
                )
            if preferred_labels and len(preferred_labels) > 0:
                params["preferred_labels"] = self._clean_labels(preferred_labels)
                cypher_query += (
                    "WHERE any(lbl IN labels(m) WHERE lbl IN $preferred_labels)\n"
                )
            cypher_query += (
                "WITH n, r, m\n"
                "WHERE r IS NOT NULL OR m IS NOT NULL\n"
            )

        cypher_query += (
            "RETURN\n"
            "    n['uuid'] as uuid, n['name'] as name, labels(n) as labels, "
            "n['description'] as description, properties(n) as properties"
        )

        if with_relationships:
            cypher_query += (
                ", r, m['uuid'] as m_uuid, m['name'] as m_name, labels(m) as m_labels, "
                "m['description'] as m_description, properties(m) as m_properties"
            )

        result = self._run_query(cypher_query, brain_id, params)

        if with_relationships:
            return [
                {
                    "node": Node(
                        uuid=record["uuid"],
                        name=record["name"],
                        labels=record["labels"],
                        description=record["description"],
                        properties=record["properties"],
                    ),
                    "relationships": record["r"] if record["r"] else None,
                    "related_nodes": (
                        Node(
                            uuid=record["m_uuid"],
                            name=record["m_name"],
                            labels=record["m_labels"],
                            description=record["m_description"],
                            properties=record["m_properties"],
                        )
                        if record["m_uuid"]
                        else None
                    ),
                }
                for record in result.records
            ]
        else:
            return [
                Node(
                    uuid=record["uuid"],
                    name=record["name"],
                    labels=record["labels"],
                    description=record["description"],
                    properties=record["properties"],
                )
                for record in result.records
            ]

    def get_graph_entities(self, brain_id: str) -> list[str]:
        """
        Get the entities of the graph.
        """
        cypher_query = """
        MATCH (n)
        RETURN DISTINCT labels(n) as labels
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        return [label for record in result.records for label in record["labels"]]

    def get_graph_relationships(self, brain_id: str) -> list[str]:
        """
        Get the relationships of the graph.
        """
        cypher_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN relationshipType
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        return [record["relationshipType"] for record in result.records]

    def get_by_uuid(self, uuid: str, brain_id: str) -> Node:
        """
        Retrieve a node by its UUID.

        Deprecated: use `get_by_uuids` which returns richer node data and supports batching.

        Parameters:
            uuid (str): The UUID of the node to retrieve.
            brain_id (str): Target database/brain identifier.

        Returns:
            Node or None: The matching node, or `None` if no node with the given UUID exists.
        """
        cypher_query = """
        MATCH (n) WHERE n['uuid'] = $uuid
        RETURN n['uuid'] as uuid, n['name'] as name, labels(n) as labels, n['description'] as description,
        properties(n) as properties,
        n['polarity'] as polarity, n['happened_at'] as happened_at, n['last_updated'] as last_updated,
        n['observations'] as observations, n['metadata'] as metadata
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(
            cypher_query, parameters_={"uuid": uuid}, database_=brain_id
        )
        if not result.records or len(result.records) == 0:
            return None
        return Node(
            uuid=result.records[0].get("uuid", ""),
            name=result.records[0].get("name", "") or "",
            labels=result.records[0].get("labels", []) or [],
            description=result.records[0].get("description", "") or "",
            properties=self._python_properties(result.records[0].get("properties", {}) or {}),
        )

    def get_by_uuids(self, uuids: list[str], brain_id: str) -> list[Node]:
        """
        Retrieve nodes that match the given UUIDs from the specified database.

        Parameters:
            uuids (list[str]): Node UUIDs to fetch.
            brain_id (str): Name of the Neo4j database to query.

        Returns:
            list[Node]: Nodes with identifiers, names, labels, descriptions, and properties; includes, when available, polarity, happened_at, last_updated, observations, and metadata.
        """
        cypher_query = (
            "MATCH (n) WHERE n['uuid'] IN $uuids\n"
            "RETURN n['uuid'] as uuid, n['name'] as name, labels(n) as labels, "
            "n['description'] as description,\n"
            "properties(n) as properties,\n"
            "n['polarity'] as polarity, n['happened_at'] as happened_at, "
            "n['last_updated'] as last_updated,\n"
            "n['observations'] as observations, n['metadata'] as metadata"
        )
        result = self._run_query(cypher_query, brain_id, {"uuids": list(uuids)})
        return [
            Node(
                uuid=record.get("uuid", ""),
                name=record.get("name", ""),
                labels=record.get("labels", []),
                description=record.get("description", ""),
                properties=self._python_properties(record.get("properties", {})),
                polarity=record.get("polarity", "neutral"),
                **(
                    {"happened_at": record.get("happened_at", None)}
                    if record.get("happened_at", None) is not None
                    else {}
                ),
                **(
                    {
                        "last_updated": self._python_value(
                            record.get("last_updated", datetime.now(timezone.utc))
                        )
                    }
                    if record.get("last_updated", datetime.now(timezone.utc))
                    is not None
                    else {}
                ),
                **(
                    {"observations": record.get("observations", [])}
                    if record.get("observations", []) is not None
                    else {}
                ),
                **(
                    {
                        "metadata": self._python_properties(
                            record.get("metadata", {})
                        )
                    }
                    if record.get("metadata", {}) is not None
                    else {}
                ),
            )
            for record in result.records
        ]

    def get_by_identification_params(
        self,
        identification_params: IdentificationParams,
        brain_id: str,
        entity_types: Optional[list[str]] = None,
    ) -> Node:
        """
        Finds a node matching the provided identification parameters and optional entity types.

        Parameters:
            identification_params (IdentificationParams): Identification fields used to locate the node (model_dump is used to build WHERE clauses).
            brain_id (str): Target database identifier to query.
            entity_types (list[str], optional): Node labels to restrict the match; when provided, the search matches nodes with any of these labels.

        Returns:
            Node or None: The matching Node populated with uuid, name, labels, description, and properties, or `None` if no node is found.
        """

        params_dict = identification_params.model_dump(
            mode="json", exclude={"entity_types"}
        )

        where_clauses = []
        query_params: dict[str, Any] = {}

        for idx, (key, value) in enumerate(params_dict.items()):
            key_name = "key_" + str(idx)
            value_name = "param_" + str(idx)
            where_clauses.append("n[$" + key_name + "] = $" + value_name)
            query_params[key_name] = self._clean_property_key(key)
            query_params[value_name] = value

        labels_str = ""
        if entity_types:
            query_params["entity_labels"] = self._clean_labels(entity_types)
            where_clauses.append(
                "any(lbl IN labels(n) WHERE lbl IN $entity_labels)"
            )
        where_str = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        cypher_query = (
            "MATCH (n" + labels_str + ") " + where_str + "\n"
            "RETURN n['uuid'] as uuid, n['name'] as name, labels(n) as labels, "
            "n['description'] as description,\n"
            "properties(n) as properties,\n"
            "n['polarity'] as polarity, n['happened_at'] as happened_at, "
            "n['last_updated'] as last_updated,\n"
            "n['observations'] as observations, n['metadata'] as metadata"
        )

        result = self._run_query(cypher_query, brain_id, query_params)
        if not result.records or len(result.records) == 0:
            return None
        return Node(
            uuid=result.records[0].get("uuid", ""),
            name=result.records[0].get("name", "") or "",
            labels=result.records[0].get("labels", []) or [],
            description=result.records[0].get("description", "") or "",
            properties=self._python_properties(result.records[0].get("properties", {}) or {}),
        )

    def get_neighbors(
        self,
        nodes: list[Node | str],
        brain_id: str,
        same_type_only: bool = False,
        limit: int | None = None,
        of_types: Optional[list[str]] = None,
    ) -> Dict[str, List[Tuple[Predicate, Node]]]:
        """
        Retrieve neighboring nodes connected to each given node, grouped by source node UUID.

        Parameters:
            nodes (list[Node | str]): List of Node objects or node UUID strings to find neighbors for.
            brain_id (str): Database identifier to query.
            same_type_only (bool): If True, include only neighbors whose labels share at least one label with the source node.
            limit (int | None): Optional maximum number of neighbor records to return (applies to the entire result set).
            of_types (Optional[list[str]]): Optional list of node label names to filter neighbors by those labels.

        Returns:
            Dict[str, List[Tuple[Predicate, Node]]]: Mapping from each source node UUID to a list of tuples (Predicate, Node),
                where Predicate describes the relationship (including direction, properties, flow key, and UUID) between the source
                and the neighbor, and Node represents the neighboring node (including its UUID, name, labels, description, and properties).
        """
        if len(nodes) == 0:
            return {}

        if isinstance(nodes[0], str):
            node_uuids = nodes
        else:
            node_uuids = [node.uuid for node in nodes]

        params: dict[str, Any] = {"uuids": list(node_uuids)}
        cypher_query = (
            "MATCH (n)-[r]-(c)\n"
            "WHERE n['uuid'] IN $uuids\n"
        )

        if same_type_only:
            cypher_query += " AND size([l IN labels(n) WHERE l IN labels(c)]) > 0"

        if of_types:
            params["of_types"] = self._clean_labels(of_types)
            cypher_query += " AND ANY(l IN labels(c) WHERE l IN $of_types)"

        cypher_query += (
            "\nRETURN n['uuid'] as uuid, n['name'] as name, labels(n) as labels, "
            "n['description'] as description,\n"
            "properties(n) as properties,\n"
            "n['polarity'] as polarity, n['happened_at'] as happened_at, "
            "n['last_updated'] as last_updated,\n"
            "n['observations'] as observations, n['metadata'] as metadata, r AS rel,\n"
            "CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END AS direction,\n"
            "type(r) AS rel_type, r['description'] AS rel_description, "
            "properties(r) AS rel_properties, r['flow_key'] as rel_flowkey, "
            "r['uuid'] as rel_uuid,\n"
            "c['uuid'] AS c_uuid, c['name'] AS c_name, labels(c) AS c_labels, "
            "c['description'] AS c_description, properties(c) AS c_properties\n"
        )
        if limit:
            params["limit"] = require_limit(limit)
            cypher_query += " LIMIT $limit"

        result = self._run_query(cypher_query, brain_id, params)

        neighbors_dict: Dict[str, List[Tuple[Predicate, Node]]] = {
            uuid: [] for uuid in node_uuids
        }

        for record in result.records:
            source_uuid = record["uuid"]
            neighbor = (
                Predicate(
                    name=record.get("rel_type", "") or "",
                    description=record.get("rel_description", "") or "",
                    direction=record.get("direction", "neutral"),
                    properties=self._python_properties(record.get("rel_properties", {}) or {}),
                    flow_key=record.get("rel_flowkey", "") or "",
                    uuid=record.get("rel_uuid", "") or "",
                ),
                Node(
                    uuid=record.get("c_uuid", ""),
                    name=record.get("c_name", "") or "",
                    labels=record.get("c_labels", []) or [],
                    description=record.get("c_description", "") or "",
                    properties=self._python_properties(record.get("c_properties", {}) or {}),
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
        """
        Get the neighbor node tuples by their UUIDs.
        """
        cypher_query = (
            "MATCH (n)-[r]-(m)\n"
            "WHERE n['uuid'] = $a_uuid AND m['uuid'] IN $b_uuids\n"
            "RETURN\n"
            "    n['uuid'] as n_uuid, n['name'] as n_name, labels(n) as n_labels,\n"
            "    n['description'] as n_description, properties(n) as n_properties,\n"
            "    m['uuid'] as m_uuid, m['name'] as m_name, labels(m) as m_labels,\n"
            "    m['description'] as m_description, properties(m) as m_properties, r as rel"
        )
        result = self._run_query(
            cypher_query, brain_id, {"a_uuid": a_uuid, "b_uuids": list(b_uuids)}
        )

        if not result.records:
            if os.getenv("DEBUG") == "true":
                raise ValueError(
                    f"No neighbor nodes found for UUID: {a_uuid} and b_uuids: {b_uuids}"
                )
            else:
                return []

        tuples = []
        for record in result.records:
            rel = record["rel"]

            rel_type = getattr(rel, "type", None) or ""
            rel_desc = ""
            try:
                rel_desc = rel.get("description", "")
            except Exception:
                try:
                    rel_desc = rel["description"]
                except Exception:
                    rel_desc = ""

            direction = "out"
            try:
                start_node = rel.nodes[0]
                start_uuid = None
                try:
                    start_uuid = start_node.get("uuid", None)
                except Exception:
                    start_uuid = start_node["uuid"]
                if start_uuid != record["n_uuid"]:
                    direction = "in"
            except Exception:
                direction = "out"

            tuples.append(
                (
                    Node(
                        uuid=record.get("n_uuid", ""),
                        name=record.get("n_name", "") or "",
                        labels=record.get("n_labels", []) or [],
                        description=record.get("n_description", "") or "",
                        properties=self._python_properties(record.get("n_properties", {}) or {}),
                    ),
                    Predicate(
                        name=rel_type,
                        description=rel_desc,
                        direction=direction,
                    ),
                    Node(
                        uuid=record.get("m_uuid", ""),
                        name=record.get("m_name", "") or "",
                        labels=record.get("m_labels", []) or [],
                        description=record.get("m_description", "") or "",
                        properties=self._python_properties(record.get("m_properties", {}) or {}),
                    ),
                )
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
        """
        Get the connected nodes by their UUIDs.
        """
        params: dict[str, Any] = {}
        clauses = []
        if node:
            params["node_uuid"] = node.uuid
            clauses.append("n['uuid'] = $node_uuid")
        elif uuids:
            params["uuids"] = list(uuids)
            clauses.append("n['uuid'] IN $uuids")
        if with_labels:
            params["with_labels"] = list(with_labels)
            clauses.append("ANY(l IN labels(m) WHERE l IN $with_labels)")
        where_str = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cypher_query = (
            "MATCH (n)-[r]-(m)\n"
            + where_str
            + "\nRETURN\n"
            "    m['uuid'] as uuid, m['name'] as name, labels(m) as labels, "
            "m['description'] as description, properties(m) as properties,\n"
            "    r as rel, type(r) as rel_type, r['description'] as rel_description,\n"
            "    n['uuid'] as n_uuid, n['name'] as n_name, labels(n) as n_labels, "
            "n['description'] as n_description, properties(n) as n_properties,\n"
            "    CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END AS direction"
        )
        result = self._run_query(cypher_query, brain_id, params)
        return [
            (
                Node(
                    uuid=record.get("uuid", ""),
                    name=record.get("name", "") or "",
                    labels=record.get("labels", []) or [],
                    description=record.get("description", "") or "",
                    properties=self._python_properties(record.get("properties", {}) or {}),
                ),
                Predicate(
                    name=record.get("rel_type", "") or "",
                    description=record.get("rel_description", "") or "",
                    direction=record.get("direction", "neutral"),
                ),
                Node(
                    uuid=record.get("n_uuid", "") or "",
                    name=record.get("n_name", "") or "",
                    labels=record.get("n_labels", []) or [],
                    description=record.get("n_description", "") or "",
                    properties=self._python_properties(record.get("n_properties", {}) or {}),
                ),
            )
            for record in result.records
        ]

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
        """
        Search the relationships of the graph.
        """
        filters = []
        params: dict[str, Any] = {
            "skip": require_limit(skip, name="skip"),
            "limit": require_limit(limit),
        }
        if relationship_types:
            params["relationship_types"] = list(relationship_types)
            filters.append("type(r) IN $relationship_types")
        if from_node_labels:
            params["from_node_labels"] = self._clean_labels(from_node_labels)
            filters.append("ANY(lbl IN labels(n) WHERE lbl IN $from_node_labels)")
        if to_node_labels:
            params["to_node_labels"] = self._clean_labels(to_node_labels)
            filters.append("ANY(lbl IN labels(m) WHERE lbl IN $to_node_labels)")
        if relationship_uuids:
            params["relationship_uuids"] = list(relationship_uuids)
            filters.append("r['uuid'] IN $relationship_uuids")
        if query_text:
            params["query_text"] = query_text
            if query_search_target == "all":
                filters.append(
                    "(toLower(coalesce(n['name'], n['name'], '')) CONTAINS toLower($query_text) OR "
                    "toLower(coalesce(m['name'], m['name'], '')) CONTAINS toLower($query_text) OR "
                    "toLower(coalesce(r['description'], r['description'], '')) CONTAINS toLower($query_text))"
                )
            elif query_search_target == "node_name":
                filters.append("toLower(n['name']) CONTAINS toLower($query_text)")
            elif query_search_target == "relationship_description":
                filters.append("toLower(r['description']) CONTAINS toLower($query_text)")
            elif query_search_target == "relationship_name":
                filters.append("toLower(r['name']) CONTAINS toLower($query_text)")
        where_str = ("WHERE " + " AND ".join(filters)) if filters else ""
        return_clause = (
            "RETURN n['uuid'] AS n_uuid, n['name'] AS n_name, labels(n) AS n_labels,\n"
            "    n['description'] AS n_description, properties(n) AS n_properties,\n"
            "    r AS rel, type(r) AS rel_type, r['description'] AS rel_description,\n"
            "    m['uuid'] AS m_uuid, m['name'] AS m_name, labels(m) AS m_labels,\n"
            "    m['description'] AS m_description, properties(m) AS m_properties\n"
            "SKIP $skip\n"
            "LIMIT $limit"
        )
        cypher_query = "MATCH (n)-[r]->(m)\n" + where_str + "\n" + return_clause
        cypher_count = (
            "MATCH (n)-[r]->(m)\n" + where_str + "\nRETURN count(r) AS total"
        )
        result = self._run_query(cypher_query, brain_id, params)
        count_result = self._run_query(cypher_count, brain_id, params, ensure=False)
        total = 0
        if count_result and count_result.records:
            total = count_result.records[0].get("total") or 0

        triples: list[Triple] = []
        for record in result.records:
            relationship = record.get("rel")
            if relationship is None:
                continue
            subject_properties = self._python_properties(record.get("n_properties"))
            object_properties = self._python_properties(record.get("m_properties"))
            relationship_properties = self._python_properties(relationship)
            subject_uuid = record.get("n_uuid") or subject_properties.get("uuid")
            object_uuid = record.get("m_uuid") or object_properties.get("uuid")
            subject_name = (
                record.get("n_name")
                or subject_properties.get("name")
                or subject_properties.get("Name")
                or subject_uuid
            )
            object_name = (
                record.get("m_name")
                or object_properties.get("name")
                or object_properties.get("Name")
                or object_uuid
            )
            subject_description = (
                record.get("n_description")
                or subject_properties.get("description")
                or subject_properties.get("Description")
            )
            object_description = (
                record.get("m_description")
                or object_properties.get("description")
                or object_properties.get("Description")
            )
            subject_labels = (
                record.get("n_labels") or subject_properties.get("labels") or []
            )
            object_labels = (
                record.get("m_labels") or object_properties.get("labels") or []
            )
            relationship_type = record.get("rel_type") or relationship.type
            relationship_description = (
                record.get("rel_description")
                or relationship_properties.get("description")
                or relationship_properties.get("Description")
            )
            subject_uuid_value = subject_uuid or str(relationship.start_node.element_id)
            object_uuid_value = object_uuid or str(relationship.end_node.element_id)
            subject_name_value = subject_name or subject_uuid_value
            object_name_value = object_name or object_uuid_value
            start_uuid = (
                relationship.start_node.get("uuid")
                if "uuid" in relationship.start_node
                else str(relationship.start_node.element_id)
            )
            direction = "out"
            if str(subject_uuid_value) != str(start_uuid):
                direction = "in"
            triples.append(
                Triple(
                    subject=Node(
                        uuid=str(subject_uuid_value),
                        name=str(subject_name_value),
                        labels=list(subject_labels),
                        description=subject_description,
                        properties=subject_properties,
                    ),
                    predicate=Predicate(
                        uuid=str(relationship_properties.get("uuid") or ""),
                        name=str(relationship_type),
                        description=relationship_description or "",
                        direction=direction,
                        properties=relationship_properties,
                        observations=None,
                        level=None,
                        deprecated=relationship_properties.get("deprecated", False),
                    ),
                    object=Node(
                        uuid=str(object_uuid_value),
                        name=str(object_name_value),
                        labels=list(object_labels),
                        description=object_description,
                        properties=object_properties,
                    ),
                )
            )
        return SearchRelationshipsResult(results=triples, total=total)

    def list_validity_relationships(
        self,
        brain_id: str,
        skip: int = 0,
        limit: int = 100,
        since: str | None = None,
        until: str | None = None,
        subject_uuids: list[str] | None = None,
    ) -> tuple[list[tuple[Node, Predicate, Node]], int]:
        params: dict[str, Any] = {
            "skip": require_limit(skip, name="skip"),
            "limit": require_limit(limit),
        }
        filters: list[str] = []
        if since and until:
            params["since"] = since
            params["until"] = until
            filters.append(relationship_changed_in_window_clause("r"))
        if subject_uuids:
            params["subject_uuids"] = list(subject_uuids)
            filters.append("n['uuid'] IN $subject_uuids")
        where_str = ("WHERE " + " AND ".join(filters) + "\n") if filters else ""
        cypher_query = (
            "MATCH (n)-[r]->(m)\n"
            + where_str
            + "RETURN n['uuid'] AS n_uuid, n['name'] AS n_name, labels(n) AS n_labels,\n"
            "    n['description'] AS n_description, properties(n) AS n_properties,\n"
            "    type(r) AS rel_type, r['description'] AS rel_description,\n"
            "    properties(r) AS rel_properties, r['uuid'] AS rel_uuid,\n"
            "    m['uuid'] AS m_uuid, m['name'] AS m_name, labels(m) AS m_labels,\n"
            "    m['description'] AS m_description, properties(m) AS m_properties\n"
            "ORDER BY r['uuid'], elementId(r)\n"
            "SKIP $skip LIMIT $limit"
        )
        count_query = (
            "MATCH (n)-[r]->(m)\n" + where_str + "RETURN count(r) AS total"
        )
        result = self._run_query(cypher_query, brain_id, params)
        count_result = self._run_query(count_query, brain_id, params, ensure=False)
        total = 0
        if count_result and count_result.records:
            total = count_result.records[0].get("total") or 0
        triples: list[tuple[Node, Predicate, Node]] = []
        for record in result.records:
            rel_props = self._python_properties(record.get("rel_properties"))
            triples.append(
                (
                    Node(
                        uuid=str(record.get("n_uuid") or ""),
                        name=str(record.get("n_name") or ""),
                        labels=list(record.get("n_labels") or []),
                        description=record.get("n_description") or "",
                        properties=self._python_properties(record.get("n_properties")),
                    ),
                    Predicate(
                        uuid=str(record.get("rel_uuid") or rel_props.get("uuid") or ""),
                        name=str(record.get("rel_type") or ""),
                        description=record.get("rel_description") or "",
                        properties=rel_props,
                        deprecated=rel_props.get("deprecated", False),
                    ),
                    Node(
                        uuid=str(record.get("m_uuid") or ""),
                        name=str(record.get("m_name") or ""),
                        labels=list(record.get("m_labels") or []),
                        description=record.get("m_description") or "",
                        properties=self._python_properties(record.get("m_properties")),
                    ),
                )
            )
        return triples, int(total)

    def search_entities(
        self,
        brain_id: str,
        limit: int = 10,
        skip: int = 0,
        node_labels: Optional[list[str]] = None,
        node_uuids: Optional[list[str]] = None,
        query_text: Optional[str] = None,
    ) -> SearchEntitiesResult:
        """
        Search the entities of the graph.
        """
        filters = []
        params: dict[str, Any] = {
            "skip": require_limit(skip, name="skip"),
            "limit": require_limit(limit),
        }
        if node_labels:
            params["node_labels"] = self._clean_labels(node_labels)
            filters.append("ANY(lbl IN labels(n) WHERE lbl IN $node_labels)")
        if node_uuids:
            params["node_uuids"] = list(node_uuids)
            filters.append("n['uuid'] IN $node_uuids")
        if query_text:
            params["query_text"] = query_text
            filters.append(
                "(toLower(coalesce(n['name'], n['name'], '')) CONTAINS toLower($query_text))"
            )
        where_str = ("WHERE " + " AND ".join(filters)) if filters else ""
        cypher_query = (
            "MATCH (n)\n"
            + where_str
            + "\nRETURN n['uuid'] as uuid, n['name'] as name, labels(n) as labels, "
            "n['description'] as description,\n"
            "properties(n) as properties,\n"
            "n['polarity'] as polarity, n['happened_at'] as happened_at, "
            "n['last_updated'] as last_updated,\n"
            "n['observations'] as observations, n['metadata'] as metadata\n"
            "SKIP $skip\n"
            "LIMIT $limit"
        )
        cypher_count = "MATCH (n)\n" + where_str + "\nRETURN count(n) AS total"
        result = self._run_query(cypher_query, brain_id, params, retry=True)
        count_result = self._run_query(
            cypher_count, brain_id, params, retry=True, ensure=False
        )
        total = 0
        if count_result and count_result.records:
            total = count_result.records[0].get("total") or 0

        nodes: list[Node] = []
        for record in result.records:
            properties_record = record.get("properties") or {}
            name = (
                record.get("name")
                or properties_record.get("name")
                or properties_record.get("Name")
                or record.get("uuid")
            )
            description = (
                record.get("description")
                or properties_record.get("description")
                or properties_record.get("Description")
            )
            uuid = record.get("uuid") or properties_record.get("uuid") or name
            labels = record.get("labels") or properties_record.get("labels") or []
            properties = _jsonable_neo4j_value(
                self._python_properties(properties_record)
            )
            nodes.append(
                Node(
                    uuid=str(uuid),
                    name=str(name) if name is not None else str(uuid),
                    labels=list(labels),
                    description=description,
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
        """
        Deprecate a relationship from the graph.
        """
        subject_labels = self._quoted_labels(subject.labels)
        object_labels = self._quoted_labels(object.labels)
        rel_type = self._quoted_rel_type(predicate.name)
        cypher_query = (
            "MATCH (a:" + subject_labels + ") WHERE a['name'] = $subject_name\n"
            "MATCH (b:" + object_labels + ") WHERE b['name'] = $object_name\n"
            "MATCH (a)-[r:" + rel_type + "]->(b)\n"
            "SET r['deprecated'] = true,\n"
            "    r['valid_to'] = coalesce(r['valid_to'], $closed_at),\n"
            "    r['invalid_at'] = coalesce(r['invalid_at'], $closed_at)\n"
            "RETURN a['uuid'] as a_uuid, a['name'] as a_name, labels(a) as a_labels, "
            "a['description'] as a_description, properties(a) as a_properties,\n"
            "    b['uuid'] as b_uuid, b['name'] as b_name, labels(b) as b_labels, "
            "b['description'] as b_description, properties(b) as b_properties,\n"
            "    r['uuid'] as r_uuid, type(r) as r_type, r['description'] as r_description, "
            "properties(r) as r_properties"
        )
        from src.core.saving.validity import format_instant, utc_now

        result = self._run_query(
            cypher_query,
            brain_id,
            {
                "subject_name": subject.name,
                "object_name": object.name,
                "closed_at": format_instant(utc_now()),
            },
        )
        if result.records:
            return (
                Node(
                    uuid=result.records[0].get("a_uuid", "") or "",
                    name=result.records[0].get("a_name", "") or "",
                    labels=result.records[0].get("a_labels", []) or [],
                    description=result.records[0].get("a_description", "") or "",
                    properties=self._python_properties(result.records[0].get("a_properties", {}) or {}),
                ),
                Predicate(
                    name=result.records[0].get("r_type", "") or "",
                    description=result.records[0].get("r_description", "") or "",
                    direction=result.records[0].get("r_direction", "neutral"),
                ),
                Node(
                    uuid=result.records[0].get("b_uuid", "") or "",
                    name=result.records[0].get("b_name", "") or "",
                    labels=result.records[0].get("b_labels", []) or [],
                    description=result.records[0].get("b_description", "") or "",
                    properties=self._python_properties(result.records[0].get("b_properties", {}) or {}),
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
        """
        Update properties on a node or relationship identified by UUID.

        Sets and/or removes properties on the matched entity and returns the updated representation.

        Parameters:
            uuid (str): UUID of the node or relationship to update.
            updating (Literal["node", "relationship"]): Whether to update a node or a relationship.
            brain_id (str): Target database identifier.
            new_properties (dict): Mapping of property keys to values to set or update on the entity.
            properties_to_remove (list[str]): List of property keys to remove from the entity.

        Returns:
            Node | Predicate | None: A `Node` when a node was updated, a `Predicate` when a relationship was updated, or `None` if no matching entity was found.
        """

        params: dict[str, Any] = {"uuid": uuid}
        set_parts = []
        for idx, (property_name, value) in enumerate(new_properties.items()):
            key_name = "set_key_" + str(idx)
            value_name = "set_val_" + str(idx)
            params[key_name] = self._clean_property_key(property_name)
            params[value_name] = self._bind_property_value(value)
            set_parts.append("t[$" + key_name + "] = $" + value_name)

        remove_parts = []
        for idx, property_name in enumerate(properties_to_remove):
            key_name = "rm_key_" + str(idx)
            params[key_name] = self._clean_property_key(property_name)
            remove_parts.append("t[$" + key_name + "] = null")

        properties_set_op = ("SET " + ", ".join(set_parts)) if set_parts else ""
        properties_remove_op = ("SET " + ", ".join(remove_parts)) if remove_parts else ""

        if updating == "node":
            cypher_query = (
                "MATCH (t) WHERE t['uuid'] = $uuid\n"
                + properties_set_op
                + "\n"
                + properties_remove_op
                + "\nRETURN t['uuid'] as uuid, t['name'] as name, labels(t) as labels, "
                "t['description'] as description, properties(t) as properties"
            )
        elif updating == "relationship":
            cypher_query = (
                "MATCH ()-[t]->() WHERE t['uuid'] = $uuid\n"
                + properties_set_op
                + "\n"
                + properties_remove_op
                + "\nRETURN t, type(t) AS rel_type, t['description'] AS rel_description, "
                "properties(t) as properties"
            )

        result = self._run_query(cypher_query, brain_id, params)

        if result.records:
            if updating == "node":
                return Node(
                    uuid=result.records[0].get("uuid", "") or "",
                    name=result.records[0].get("name", "") or "",
                    labels=result.records[0].get("labels", []) or [],
                    description=result.records[0].get("description", "") or "",
                    properties=self._python_properties(result.records[0].get("properties", {}) or {}),
                )
            elif updating == "relationship":
                return Predicate(
                    name=result.records[0].get("rel_type", "") or "",
                    description=result.records[0].get("rel_description", "") or "",
                    direction=result.records[0].get("direction", "neutral"),
                )
        return None

    def get_graph_relationship_types(self, brain_id: str) -> list[str]:
        """
        Return all relationship type names present in the specified brain (database).

        Parameters:
            brain_id (str): Target database name to query for relationship types.

        Returns:
            list[str]: A list of relationship type names found in the graph.
        """
        cypher_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN relationshipType
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        return [record["relationshipType"] for record in result.records]

    def get_graph_node_types(self, brain_id: str) -> list[str]:
        """
        Return the set of node label types present in the specified graph database.

        Parameters:
            brain_id (str): The target database/graph identifier to query.

        Returns:
            list[str]: A list of node label strings found in the database.
        """
        cypher_query = """
        CALL db.labels() YIELD label
        RETURN label
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        return [record["label"] for record in result.records]

    def get_graph_node_properties(self, brain_id: str) -> list[str]:
        """
        Return all distinct node property keys present in the database, ordered alphabetically.

        Returns:
            properties (list[str]): Sorted list of distinct property key names used on nodes.
        """
        cypher_query = """
        MATCH (n)
        UNWIND keys(n) AS property
        RETURN DISTINCT property
        ORDER BY property
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        return [record["property"] for record in result.records]

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
        """
        Update the node identified by `uuid` with the provided name, description, labels, and property changes.

        If `new_properties` is provided, only properties that are new or whose values differ from the existing node are set; existing properties not mentioned are left unchanged. If `properties_to_remove` is provided, those properties are removed. If `new_labels` is provided, existing labels are removed (if the node was loaded) and replaced with the cleaned `new_labels`. If no changes are specified, the function returns the existing node (or None if not found) without executing an update.

        Parameters:
            uuid (str): UUID of the node to update.
            brain_id (str): Target database/brain identifier.
            new_name (Optional[str]): New value for the node's `name` property.
            new_description (Optional[str]): New value for the node's `description` property.
            new_labels (Optional[list[str]]): New labels to assign to the node; labels will be cleaned and applied.
            new_properties (Optional[dict]): Properties to add or update; only keys that are new or changed are applied when the existing node can be loaded.
            properties_to_remove (Optional[list[str]]): Property keys to remove from the node.

        Returns:
            Node or None: The updated Node when the update succeeds, the existing Node if no changes were requested, or `None` if the node does not exist or no update occurred.
        """
        operations = []
        params: dict[str, Any] = {"uuid": uuid}

        existing_node = None
        if new_properties:
            existing_node = self.get_by_uuid(uuid, brain_id)
            if not existing_node:
                return None

        if new_name:
            params["new_name"] = new_name
            operations.append("n['name'] = $new_name")

        if new_description:
            params["new_description"] = new_description
            operations.append("n['description'] = $new_description")

        property_items = []
        if new_properties and existing_node:
            existing_properties = existing_node.properties or {}
            for property_key, new_value in new_properties.items():
                existing_value = existing_properties.get(property_key)
                if (
                    property_key not in existing_properties
                    or existing_value != new_value
                ):
                    property_items.append((property_key, new_value))
        elif new_properties:
            property_items = list(new_properties.items())

        for idx, (property_key, new_value) in enumerate(property_items):
            key_name = "prop_key_" + str(idx)
            value_name = "prop_val_" + str(idx)
            params[key_name] = self._clean_property_key(property_key)
            params[value_name] = self._bind_property_value(new_value)
            operations.append("n[$" + key_name + "] = $" + value_name)

        set_clause = ("SET " + ", ".join(operations)) if operations else ""

        remove_clause = ""
        if properties_to_remove:
            remove_parts = []
            for idx, prop in enumerate(properties_to_remove):
                key_name = "rm_key_" + str(idx)
                params[key_name] = self._clean_property_key(prop)
                remove_parts.append("n[$" + key_name + "] = null")
            remove_clause = "SET " + ", ".join(remove_parts)

        labels_clause = ""
        if new_labels:
            cleaned_labels = self._quoted_labels(new_labels)
            node_for_labels = existing_node
            if not node_for_labels:
                node_for_labels = self.get_by_uuid(uuid, brain_id)
            if node_for_labels:
                current_labels = node_for_labels.labels or []
                if current_labels:
                    labels_clause = "REMOVE n:" + self._quoted_labels(current_labels) + "\n"
            labels_clause += "SET n:" + cleaned_labels

        if not set_clause and not remove_clause and not labels_clause:
            return existing_node if existing_node else self.get_by_uuid(uuid, brain_id)

        cypher_query = (
            "MATCH (n)\n"
            "WHERE n['uuid'] = $uuid\n"
            + set_clause
            + "\n"
            + remove_clause
            + "\n"
            + labels_clause
            + "\nRETURN n['uuid'] as uuid, n['name'] as name, labels(n) as labels,\n"
            "    n['description'] as description, properties(n) as properties"
        )

        result = self._run_query(cypher_query, brain_id, params)

        if result.records:
            return Node(
                uuid=result.records[0].get("uuid", "") or "",
                name=result.records[0].get("name", "") or "",
                labels=result.records[0].get("labels", []) or [],
                description=result.records[0].get("description", "") or "",
                properties=self._python_properties(result.records[0].get("properties", {}) or {}),
            )

        return None

    def get_schema(self, brain_id: str) -> dict:
        """
        Retrieve the graph schema for the given brain, including node labels, relationship types, and event names.

        Parameters:
            brain_id (str): Identifier of the Neo4j database/brain to query.

        Returns:
            dict: A dictionary with keys:
                - "labels": list of node label names (str).
                - "relationships": list of relationship type names (str).
                - "event_names": list of node names (str) for nodes labeled "EVENT".
        """
        self.ensure_database(brain_id)

        labels_query = "CALL db.labels() YIELD label RETURN label"
        relationships_query = (
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
        event_names_query = "MATCH (n) WHERE 'EVENT' IN labels(n) RETURN n.name as name"

        labels_result = self.driver.execute_query(labels_query, database_=brain_id)
        relationships_result = self.driver.execute_query(
            relationships_query, database_=brain_id
        )
        event_names_result = self.driver.execute_query(
            event_names_query, database_=brain_id
        )

        labels = [record["label"] for record in labels_result.records]
        relationships = [
            record["relationshipType"] for record in relationships_result.records
        ]
        event_names = [record["name"] for record in event_names_result.records]

        return {
            "labels": labels,
            "relationships": relationships,
            "event_names": event_names,
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
        """
        Determine whether a node with the given UUID, name, and labels exists in the specified database.

        Parameters:
            uuid (str): Node identifier to match.
            name (str): Node name to match.
            labels (list[str]): All labels the node must have.
            brain_id (str): Target database name.

        Returns:
            bool: True if a node matching the UUID, name, and labels exists, False otherwise.
        """
        cypher_query = (
            "MATCH (n:" + self._quoted_labels(labels) + ")\n"
            "WHERE n['name'] = $name\n"
            "AND n['uuid'] = $uuid\n"
            "RETURN n"
        )
        result = self._run_query(
            cypher_query, brain_id, {"uuid": uuid, "name": name}
        )
        return len(result.records) > 0

    def get_neighborhood(
        self, node: Node | str, depth: int, brain_id: str
    ) -> list[dict]:
        """
        Retrieve the neighborhood of a node up to a specified depth as a nested structure.

        Parameters:
            node (Node | str): The starting node or its UUID.
            depth (int): Maximum number of hops to traverse; must be >= 1.
            brain_id (str): Identifier for the brain/graph to scope the query.

        Returns:
            list[dict]: A list of neighbor dictionaries. Each dictionary contains the keys
            `node` (the neighboring Node), `predicate` (the connecting Predicate),
            `direction` (relationship direction relative to the starting node), and
            `neighbors` (a list of child neighbor dictionaries with the same shape).
        """
        if depth < 1:
            return []
        node_uuid = node.uuid if isinstance(node, Node) else node
        return self._get_neighborhood_recursive(node_uuid, depth, brain_id, set())

    def _get_neighborhood_recursive(
        self, node_uuid: str, depth: int, brain_id: str, path_visited: set[str]
    ) -> list[dict]:
        """
        Recursively collects neighboring nodes and their connecting predicates up to the specified depth.

        Parameters:
            node_uuid (str): UUID of the node to start traversal from.
            depth (int): Maximum number of hops to traverse; values less than 1 result in no neighbors.
            brain_id (str): Database/brain identifier to run the query against.
            path_visited (set[str]): Set of node UUIDs already visited in the current traversal path; used to prevent cycles.

        Returns:
            list[dict]: A list of neighbor entries. Each entry is a dict with keys:
                - "predicate": a Predicate instance describing the relationship from the current node to the neighbor.
                - "node": a Node instance representing the neighboring node.
                - "neighbors": a list of nested neighbor entries (same structure) for further hops.
        """
        if depth < 1 or node_uuid in path_visited:
            return []

        path_visited = path_visited.copy()
        path_visited.add(node_uuid)

        cypher_query = """
        MATCH (n)-[r]-(m)
        WHERE n['uuid'] = $uuid
        RETURN 
            r as rel,
            type(r) as rel_type,
            r['description'] as rel_description,
            properties(r) as rel_properties,
            r['flow_key'] as rel_flowkey,
            r['uuid'] as rel_uuid,
            r['last_updated'] as rel_last_updated,
            r['observations'] as rel_observations,
            r['amount'] as rel_amount,
            CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END AS direction,
            m['uuid'] as m_uuid,
            m['name'] as m_name,
            labels(m) as m_labels,
            m['description'] as m_description,
            properties(m) as m_properties,
            m['polarity'] as m_polarity,
            m['metadata'] as m_metadata,
            m['happened_at'] as m_happened_at,
            m['last_updated'] as m_last_updated,
            m['observations'] as m_observations
        """

        self.ensure_database(brain_id)
        result = self.driver.execute_query(
            cypher_query,
            parameters_={"uuid": node_uuid},
            database_=brain_id,
        )

        neighbors = []
        for record in result.records:
            neighbor_node_uuid = record.get("m_uuid", "")
            if neighbor_node_uuid in path_visited:
                continue

            neighbor_node = Node(
                uuid=neighbor_node_uuid,
                name=record.get("m_name", "") or "",
                labels=record.get("m_labels", []) or [],
                description=record.get("m_description", "") or "",
                properties=self._python_properties(record.get("m_properties", {})),
                polarity=record.get("m_polarity", "neutral"),
                metadata=self._python_properties(record.get("m_metadata", {})),
                happened_at=record.get("m_happened_at", "") or "",
                last_updated=record.get("m_last_updated", "") or "",
                observations=record.get("m_observations", []) or [],
            )

            predicate = Predicate(
                name=record.get("rel_type", "") or "",
                description=record.get("rel_description", "") or "",
                direction=record.get("direction", "neutral"),
                properties=self._python_properties(record.get("rel_properties", {})),
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
        """
        Retrieve the event-centric neighbors of a node.
        """
        if len(nodes) == 0:
            return []
        if isinstance(nodes[0], str):
            node_uuids = nodes
        else:
            node_uuids = [node.uuid for node in nodes]
        cypher_query = (
            "MATCH (n)-[r]-(m)-[r2]-(b)\n"
            "WHERE n['uuid'] IN $node_uuids\n"
            "AND r2['flow_key'] = r['flow_key']\n"
            "AND type(r) <> 'HUB_BRIDGE'\n"
            "AND type(r2) <> 'HUB_BRIDGE'\n"
            "RETURN\n"
            "    n['uuid'] as n_uuid, n['name'] as n_name, labels(n) as n_labels, n['description'] as n_description, properties(n) as n_properties, n['polarity'] as n_polarity, n['metadata'] as n_metadata, n['happened_at'] as n_happened_at, n['last_updated'] as n_last_updated, n['observations'] as n_observations,\n"
            "    r['uuid'] as r_uuid, type(r) as r_type, r['description'] as r_description, properties(r) as r_properties, r['flow_key'] as r_flow_key, r['last_updated'] as r_last_updated, r['observations'] as r_observations, r['amount'] as r_amount,\n"
            "    CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END AS r_direction,\n"
            "    m['uuid'] as m_uuid, m['name'] as m_name, labels(m) as m_labels, m['description'] as m_description, properties(m) as m_properties, m['polarity'] as m_polarity, m['metadata'] as m_metadata, m['happened_at'] as m_happened_at, m['last_updated'] as m_last_updated, m['observations'] as m_observations,\n"
            "    r2['uuid'] as r2_uuid, type(r2) as r2_type, r2['description'] as r2_description, properties(r2) as r2_properties, r2['flow_key'] as r2_flow_key, r2['last_updated'] as r2_last_updated, r2['observations'] as r2_observations, r2['amount'] as r2_amount,\n"
            "    CASE WHEN startNode(r2) = m THEN 'out' ELSE 'in' END AS r2_direction,\n"
            "    b['uuid'] as b_uuid, b['name'] as b_name, labels(b) as b_labels, b['description'] as b_description, properties(b) as b_properties, b['polarity'] as b_polarity, b['metadata'] as b_metadata, b['happened_at'] as b_happened_at, b['last_updated'] as b_last_updated, b['observations'] as b_observations"
        )
        result = self._run_query(
            cypher_query, brain_id, {"node_uuids": list(node_uuids)}
        )
        return [
            (
                Node(
                    uuid=record.get("n_uuid", "") or "",
                    name=record.get("n_name", "") or "",
                    labels=record.get("n_labels", []) or [],
                    description=record.get("n_description", "") or "",
                    properties=self._python_properties(record.get("n_properties", {})),
                    polarity=record.get("n_polarity", "neutral"),
                    metadata=self._python_properties(record.get("n_metadata", {})),
                    happened_at=record.get("n_happened_at", "") or "",
                    last_updated=record.get("n_last_updated", "") or "",
                    observations=record.get("n_observations", []) or [],
                ),
                Predicate(
                    uuid=record.get("r_uuid", "") or "",
                    name=record.get("r_type", "") or "",
                    description=record.get("r_description", "") or "",
                    direction=record.get("r_direction", "neutral"),
                    properties=self._python_properties(record.get("r_properties", {})),
                    flow_key=record.get("r_flow_key", "") or "",
                    last_updated=record.get("r_last_updated", "") or "",
                    observations=record.get("r_observations", []) or [],
                    amount=record.get("r_amount"),
                ),
                Node(
                    uuid=record.get("m_uuid", "") or "",
                    name=record.get("m_name", "") or "",
                    labels=record.get("m_labels", []) or [],
                    description=record.get("m_description", "") or "",
                    properties=self._python_properties(record.get("m_properties", {})),
                    polarity=record.get("m_polarity", "neutral"),
                    metadata=self._python_properties(record.get("m_metadata", {})),
                    happened_at=record.get("m_happened_at", "") or "",
                    last_updated=record.get("m_last_updated", "") or "",
                    observations=record.get("m_observations", []) or [],
                ),
                Predicate(
                    uuid=record.get("r2_uuid", "") or "",
                    name=record.get("r2_type", "") or "",
                    description=record.get("r2_description", "") or "",
                    direction=record.get("r2_direction", "neutral"),
                    properties=self._python_properties(record.get("r2_properties", {})),
                    flow_key=record.get("r2_flow_key", "") or "",
                    last_updated=record.get("r2_last_updated", "") or "",
                    observations=record.get("r2_observations", []) or [],
                    amount=record.get("r2_amount"),
                ),
                Node(
                    uuid=record.get("b_uuid", "") or "",
                    name=record.get("b_name", "") or "",
                    labels=record.get("b_labels", []) or [],
                    description=record.get("b_description", "") or "",
                    properties=self._python_properties(record.get("b_properties", {})),
                    polarity=record.get("b_polarity", "neutral"),
                    metadata=self._python_properties(record.get("b_metadata", {})),
                    happened_at=record.get("b_happened_at", "") or "",
                    last_updated=record.get("b_last_updated", "") or "",
                    observations=record.get("b_observations", []) or [],
                ),
            )
            for record in result.records
        ]

    def get_nexts_by_flow_key(
        self, predicates: list[PredicateWithFlowKey], brain_id: str
    ) -> List[Tuple[Node, Predicate, Node]]:
        """
        Retrieve the next connected node tuple(s) for a relationship identified by a flow key, grouped by the predicate UUID.

        Parameters:
            predicates (list[PredicateWithFlowKey]): A list of predicates with their flow keys.
            brain_id (str): Database name (brain) to execute the query against.

        Returns:
            Dict[str, List[Tuple[Node, Predicate, Node]]]: A dictionary mapping predicate UUIDs to lists of (subject node, predicate, object node) tuples that are the next nodes matching the provided flow key; empty dictionary if none are found for any predicate UUID.
        """
        res = {}
        for predicate in predicates:
            cypher_query = """
            MATCH ()-[r]-(m)-[r2]-(b)
            WHERE r['uuid'] = $predicate_uuid
            AND r2['flow_key'] = $flow_key
            RETURN
                m['uuid'] as m_uuid, m['name'] as m_name, labels(m) as m_labels, m['description'] as m_description, properties(m) as m_properties, m['polarity'] as m_polarity, m['metadata'] as m_metadata, m['happened_at'] as m_happened_at, m['last_updated'] as m_last_updated, m['observations'] as m_observations,
                r2['uuid'] as r2_uuid, type(r2) as r2_type, r2['description'] as r2_description, properties(r2) as r2_properties, r2['flow_key'] as r2_flow_key, r2['last_updated'] as r2_last_updated, r2['observations'] as r2_observations, r2['amount'] as r2_amount,
                CASE WHEN startNode(r2) = m THEN 'out' ELSE 'in' END AS r2_direction,
                b['uuid'] as b_uuid, b['name'] as b_name, labels(b) as b_labels, b['description'] as b_description, properties(b) as b_properties, b['polarity'] as b_polarity, b['metadata'] as b_metadata, b['happened_at'] as b_happened_at, b['last_updated'] as b_last_updated, b['observations'] as b_observations
            """
            self.ensure_database(brain_id)
            result = self.driver.execute_query(
                cypher_query,
                parameters_={
                    "predicate_uuid": predicate["predicate_uuid"],
                    "flow_key": predicate["flow_key"],
                },
                database_=brain_id,
            )
            res[predicate["predicate_uuid"]] = [
                (
                    Node(
                        uuid=record.get("m_uuid", "") or "",
                        name=record.get("m_name", "") or "",
                        labels=record.get("m_labels", []) or [],
                        description=record.get("m_description", "") or "",
                        properties=self._python_properties(record.get("m_properties", {})),
                        polarity=record.get("m_polarity", "neutral"),
                        metadata=self._python_properties(record.get("m_metadata", {})),
                        happened_at=record.get("m_happened_at", "") or "",
                        last_updated=record.get("m_last_updated", "") or "",
                        observations=record.get("m_observations", []) or [],
                    ),
                    Predicate(
                        uuid=record.get("r2_uuid", "") or "",
                        name=record.get("r2_type", "") or "",
                        description=record.get("r2_description", "") or "",
                        direction=record.get("r2_direction", "neutral"),
                        properties=self._python_properties(record.get("r2_properties", {})),
                        flow_key=record.get("r2_flow_key", "") or "",
                        last_updated=record.get("r2_last_updated", "") or "",
                        observations=record.get("r2_observations", []) or [],
                        amount=record.get("r2_amount"),
                    ),
                    Node(
                        uuid=record.get("b_uuid", "") or "",
                        name=record.get("b_name", "") or "",
                        labels=record.get("b_labels", []) or [],
                        description=record.get("b_description", "") or "",
                        properties=self._python_properties(record.get("b_properties", {})),
                        polarity=record.get("b_polarity", "neutral"),
                        metadata=self._python_properties(record.get("b_metadata", {})),
                        happened_at=record.get("b_happened_at", "") or "",
                        last_updated=record.get("b_last_updated", "") or "",
                        observations=record.get("b_observations", []) or [],
                    ),
                )
                for record in result.records
            ]
        return res

    def get_triples_by_uuid(
        self, uuids: list[str], brain_id: str
    ) -> List[Tuple[Node, Predicate, Node]]:
        """
        Get triples by its UUID.
        """
        cypher_query = (
            "MATCH (n)-[r]-(m) WHERE r['uuid'] IN $uuids\n"
            "RETURN\n"
            "    n['uuid'] as n_uuid, n['name'] as n_name, labels(n) as n_labels, n['description'] as n_description, properties(n) as n_properties,\n"
            "    r['uuid'] as r_uuid, type(r) as r_type, r['description'] as r_description, properties(r) as r_properties,\n"
            "    m['uuid'] as m_uuid, m['name'] as m_name, labels(m) as m_labels, m['description'] as m_description, properties(m) as m_properties"
        )
        result = self._run_query(cypher_query, brain_id, {"uuids": list(uuids)})
        return [
            (
                Node(
                    uuid=record.get("n_uuid", "") or "",
                    name=record.get("n_name", "") or "",
                    labels=record.get("n_labels", []) or [],
                    description=record.get("n_description", "") or "",
                    properties=self._python_properties(record.get("n_properties", {}) or {}),
                ),
                Predicate(
                    name=record.get("r_type", "") or "",
                    description=record.get("r_description", "") or "",
                    direction=record.get("r_direction", "neutral"),
                    properties=self._python_properties(record.get("r_properties", {}) or {}),
                ),
                Node(
                    uuid=record.get("m_uuid", "") or "",
                    name=record.get("m_name", "") or "",
                    labels=record.get("m_labels", []) or [],
                    description=record.get("m_description", "") or "",
                    properties=self._python_properties(record.get("m_properties", {}) or {}),
                ),
            )
            for record in result.records
        ]

    def remove_nodes(self, uuids: list[str], brain_id: str) -> list[Node]:
        """
        Remove nodes from the graph.
        """
        cypher_query = (
            "MATCH (n) WHERE n['uuid'] IN $uuids\n"
            "WITH n, {\n"
            "uuid: n['uuid'],\n"
            "name: n['name'],\n"
            "labels: labels(n),\n"
            "description: n['description'],\n"
            "properties: properties(n),\n"
            "polarity: n['polarity'],\n"
            "metadata: n['metadata'],\n"
            "happened_at: n['happened_at'],\n"
            "last_updated: n['last_updated'],\n"
            "observations: n['observations']\n"
            "} AS node\n"
            "DELETE n\n"
            "RETURN node"
        )
        result = self._run_query(cypher_query, brain_id, {"uuids": list(uuids)})
        return [Node(**record.get("node", {})) for record in result.records]

    def remove_relationships(
        self,
        relationships: list[Tuple[NodeDict, PredicateDict, NodeDict]],
        brain_id: str,
    ) -> list[Tuple[Node, Predicate, Node]]:
        """
        Remove relationships from the graph. Matches each relationship either by
        predicate uuid (r.uuid) or by tip/tail node identifiers (node uuid or
        name + labels). Returns the list of deleted triples (tail, predicate, head).
        """
        self.ensure_database(brain_id)
        deleted: list[Tuple[Node, Predicate, Node]] = []
        by_rel_uuid: list[str] = []
        by_tip_tail: list[Tuple[NodeDict, PredicateDict, NodeDict]] = []
        for tail, pred, head in relationships:
            if pred.get("uuid"):
                by_rel_uuid.append(pred["uuid"])
            else:
                by_tip_tail.append((tail, pred, head))
        with_fields = """
            n['uuid'] AS n_uuid, n['name'] AS n_name, labels(n) AS n_labels, n['description'] AS n_description, properties(n) AS n_properties,
            r['uuid'] AS r_uuid, type(r) AS r_type, r['description'] AS r_description, properties(r) AS r_properties,
            m['uuid'] AS m_uuid, m['name'] AS m_name, labels(m) AS m_labels, m['description'] AS m_description, properties(m) AS m_properties
        """
        return_list = "n_uuid, n_name, n_labels, n_description, n_properties, r_uuid, r_type, r_description, r_properties, m_uuid, m_name, m_labels, m_description, m_properties"
        if by_rel_uuid:
            cypher = (
                "MATCH (n)-[r]-(m) WHERE r['uuid'] IN $uuids\n"
                "WITH n, r, m, " + with_fields.strip() + "\n"
                "DELETE r\n"
                "RETURN " + return_list
            )
            result = self._run_query(
                cypher, brain_id, {"uuids": list(by_rel_uuid)}, ensure=False
            )
            for record in result.records:
                deleted.append(self._record_to_triple(record))
        for index, (tail, pred, head) in enumerate(by_tip_tail):
            params: dict[str, Any] = {}
            tail_match = self._node_match_cypher("n", tail, params, "tail" + str(index))
            head_match = self._node_match_cypher("m", head, params, "head" + str(index))
            rel_type = pred.get("name")
            if rel_type:
                rel_part = "[r:" + self._quoted_rel_type(rel_type) + "]"
            else:
                rel_part = "[r]"
            cypher = (
                "MATCH " + tail_match + "\n"
                "MATCH " + head_match + "\n"
                "MATCH (n)-" + rel_part + "-(m)\n"
                "WITH n, r, m, " + with_fields.strip() + "\n"
                "DELETE r\n"
                "RETURN " + return_list
            )
            result = self._run_query(cypher, brain_id, params, ensure=False)
            for record in result.records:
                deleted.append(self._record_to_triple(record))
        return deleted

    def _record_to_triple(self, record: Any) -> Tuple[Node, Predicate, Node]:
        return (
            Node(
                uuid=record.get("n_uuid", "") or "",
                name=record.get("n_name", "") or "",
                labels=record.get("n_labels", []) or [],
                description=record.get("n_description", "") or "",
                properties=self._python_properties(record.get("n_properties", {}) or {}),
            ),
            Predicate(
                uuid=record.get("r_uuid", "") or "",
                name=record.get("r_type", "") or "",
                description=record.get("r_description", "") or "",
                direction="neutral",
                properties=self._python_properties(record.get("r_properties", {}) or {}),
            ),
            Node(
                uuid=record.get("m_uuid", "") or "",
                name=record.get("m_name", "") or "",
                labels=record.get("m_labels", []) or [],
                description=record.get("m_description", "") or "",
                properties=self._python_properties(record.get("m_properties", {}) or {}),
            ),
        )

    def _node_match_cypher(
        self, alias: str, node: NodeDict, params: dict[str, Any], token: str
    ) -> str:
        if not node:
            return "(" + alias + ")"
        if node.get("uuid"):
            pname = token + "_uuid"
            params[pname] = node["uuid"]
            return "(" + alias + ") WHERE " + alias + "['uuid'] = $" + pname
        labels = node.get("labels") or []
        name = node.get("name") or ""
        pname = token + "_name"
        params[pname] = name
        if labels:
            labels_part = self._quoted_labels(labels)
            return (
                "(" + alias + ":" + labels_part + ") WHERE "
                + alias + "['name'] = $" + pname
            )
        if name:
            return "(" + alias + ") WHERE " + alias + "['name'] = $" + pname
        return "(" + alias + ")"

    def list_relationships(
        self, subject: str, object: str, brain_id: str
    ) -> list[Tuple[Node, Predicate, Node]]:
        """
        List the relationships between the subject and object.
        """
        cypher_query = (
            "MATCH (n)-[r]-(m) WHERE n['uuid'] = $subject AND m['uuid'] = $object\n"
            "RETURN n['uuid'] as n_uuid, n['name'] as n_name, labels(n) as n_labels, n['description'] as n_description, properties(n) as n_properties, n['polarity'] as n_polarity, n['metadata'] as n_metadata, n['happened_at'] as n_happened_at, n['last_updated'] as n_last_updated, n['observations'] as n_observations,\n"
            "r['uuid'] as r_uuid, type(r) as r_type, r['description'] as r_description, properties(r) as r_properties, r['flow_key'] as r_flow_key, r['last_updated'] as r_last_updated, r['observations'] as r_observations, r['amount'] as r_amount,\n"
            "m['uuid'] as m_uuid, m['name'] as m_name, labels(m) as m_labels, m['description'] as m_description, properties(m) as m_properties, m['polarity'] as m_polarity, m['metadata'] as m_metadata, m['happened_at'] as m_happened_at, m['last_updated'] as m_last_updated, m['observations'] as m_observations"
        )
        result = self._run_query(
            cypher_query, brain_id, {"subject": subject, "object": object}
        )
        return [
            (
                Node(
                    uuid=record.get("n_uuid", "") or "",
                    name=record.get("n_name", "") or "",
                    labels=record.get("n_labels", []) or [],
                    description=record.get("n_description", "") or "",
                    properties=self._python_properties(record.get("n_properties", {}) or {}),
                    polarity=record.get("n_polarity", "neutral"),
                    metadata=record.get("n_metadata", {}) or {},
                    happened_at=record.get("n_happened_at", "") or "",
                    last_updated=record.get("n_last_updated", "") or "",
                    observations=record.get("n_observations", []) or [],
                ),
                Predicate(
                    uuid=record.get("r_uuid", "") or "",
                    name=record.get("r_type", "") or "",
                    description=record.get("r_description", "") or "",
                    direction="neutral",
                    properties=self._python_properties(record.get("r_properties", {}) or {}),
                    flow_key=record.get("r_flow_key", "") or "",
                    last_updated=record.get("r_last_updated", "") or "",
                    observations=record.get("r_observations", []) or [],
                    amount=record.get("r_amount"),
                ),
                Node(
                    uuid=record.get("m_uuid", "") or "",
                    name=record.get("m_name", "") or "",
                    labels=record.get("m_labels", []) or [],
                    description=record.get("m_description", "") or "",
                    properties=self._python_properties(record.get("m_properties", {}) or {}),
                    polarity=record.get("m_polarity", "neutral"),
                    metadata=record.get("m_metadata", {}) or {},
                    happened_at=record.get("m_happened_at", "") or "",
                    last_updated=record.get("m_last_updated", "") or "",
                    observations=record.get("m_observations", []) or [],
                ),
            )
            for record in result.records
        ]


    def get_event_hub_facts(
        self,
        event_uuids: list[str],
        brain_id: str,
    ) -> List[Tuple[Node, Predicate, Node, Predicate, Node]]:
        unique = sorted({str(u) for u in event_uuids if u})
        if not unique:
            return []
        cypher_query = (
            "MATCH (n)-[r]-(m)-[r2]-(b)\n"
            "WHERE m['uuid'] IN $event_uuids\n"
            "AND r2['flow_key'] = r['flow_key']\n"
            "AND type(r) <> 'HUB_BRIDGE'\n"
            "AND type(r2) <> 'HUB_BRIDGE'\n"
            "AND 'EVENT' IN labels(m)\n"
            "RETURN\n"
            "    n['uuid'] as n_uuid, n['name'] as n_name, labels(n) as n_labels, n['description'] as n_description, properties(n) as n_properties, n['polarity'] as n_polarity, n['metadata'] as n_metadata, n['happened_at'] as n_happened_at, n['last_updated'] as n_last_updated, n['observations'] as n_observations,\n"
            "    r['uuid'] as r_uuid, type(r) as r_type, r['description'] as r_description, properties(r) as r_properties, r['flow_key'] as r_flow_key, r['last_updated'] as r_last_updated, r['observations'] as r_observations, r['amount'] as r_amount,\n"
            "    CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END AS r_direction,\n"
            "    m['uuid'] as m_uuid, m['name'] as m_name, labels(m) as m_labels, m['description'] as m_description, properties(m) as m_properties, m['polarity'] as m_polarity, m['metadata'] as m_metadata, m['happened_at'] as m_happened_at, m['last_updated'] as m_last_updated, m['observations'] as m_observations,\n"
            "    r2['uuid'] as r2_uuid, type(r2) as r2_type, r2['description'] as r2_description, properties(r2) as r2_properties, r2['flow_key'] as r2_flow_key, r2['last_updated'] as r2_last_updated, r2['observations'] as r2_observations, r2['amount'] as r2_amount,\n"
            "    CASE WHEN startNode(r2) = m THEN 'out' ELSE 'in' END AS r2_direction,\n"
            "    b['uuid'] as b_uuid, b['name'] as b_name, labels(b) as b_labels, b['description'] as b_description, properties(b) as b_properties, b['polarity'] as b_polarity, b['metadata'] as b_metadata, b['happened_at'] as b_happened_at, b['last_updated'] as b_last_updated, b['observations'] as b_observations\n"
            "ORDER BY m_uuid, r_uuid, r2_uuid"
        )
        result = self._run_query(
            cypher_query, brain_id, {"event_uuids": unique}
        )
        return [
            (
                Node(
                    uuid=record.get("n_uuid", "") or "",
                    name=record.get("n_name", "") or "",
                    labels=record.get("n_labels", []) or [],
                    description=record.get("n_description", "") or "",
                    properties=self._python_properties(record.get("n_properties", {})),
                    polarity=record.get("n_polarity", "neutral"),
                    metadata=self._python_properties(record.get("n_metadata", {})),
                    happened_at=record.get("n_happened_at", "") or "",
                    last_updated=record.get("n_last_updated", "") or "",
                    observations=record.get("n_observations", []) or [],
                ),
                Predicate(
                    uuid=record.get("r_uuid", "") or "",
                    name=record.get("r_type", "") or "",
                    description=record.get("r_description", "") or "",
                    direction=record.get("r_direction", "neutral"),
                    properties=self._python_properties(record.get("r_properties", {})),
                    flow_key=record.get("r_flow_key", "") or "",
                    last_updated=record.get("r_last_updated", "") or "",
                    observations=record.get("r_observations", []) or [],
                    amount=record.get("r_amount"),
                ),
                Node(
                    uuid=record.get("m_uuid", "") or "",
                    name=record.get("m_name", "") or "",
                    labels=record.get("m_labels", []) or [],
                    description=record.get("m_description", "") or "",
                    properties=self._python_properties(record.get("m_properties", {})),
                    polarity=record.get("m_polarity", "neutral"),
                    metadata=self._python_properties(record.get("m_metadata", {})),
                    happened_at=record.get("m_happened_at", "") or "",
                    last_updated=record.get("m_last_updated", "") or "",
                    observations=record.get("m_observations", []) or [],
                ),
                Predicate(
                    uuid=record.get("r2_uuid", "") or "",
                    name=record.get("r2_type", "") or "",
                    description=record.get("r2_description", "") or "",
                    direction=record.get("r2_direction", "neutral"),
                    properties=self._python_properties(record.get("r2_properties", {})),
                    flow_key=record.get("r2_flow_key", "") or "",
                    last_updated=record.get("r2_last_updated", "") or "",
                    observations=record.get("r2_observations", []) or [],
                    amount=record.get("r2_amount"),
                ),
                Node(
                    uuid=record.get("b_uuid", "") or "",
                    name=record.get("b_name", "") or "",
                    labels=record.get("b_labels", []) or [],
                    description=record.get("b_description", "") or "",
                    properties=self._python_properties(record.get("b_properties", {})),
                    polarity=record.get("b_polarity", "neutral"),
                    metadata=self._python_properties(record.get("b_metadata", {})),
                    happened_at=record.get("b_happened_at", "") or "",
                    last_updated=record.get("b_last_updated", "") or "",
                    observations=record.get("b_observations", []) or [],
                ),
            )
            for record in result.records
        ]

    def rebuild_hub_bridge_index(self, brain_id: str) -> int:
        from src.core.saving.hub_bridges import (
            bridges_from_memberships,
            entity_event_memberships,
        )

        self.ensure_database(brain_id)
        cypher_rows = """
        MATCH (event)-[r]-(entity)
        WHERE 'EVENT' IN labels(event)
          AND NOT 'EVENT' IN labels(entity)
          AND type(r) <> 'HUB_BRIDGE'
        RETURN event['uuid'] AS event_uuid,
               entity['uuid'] AS entity_uuid,
               entity['name'] AS entity_name,
               type(r) AS rel_name
        """
        result = self.driver.execute_query(cypher_rows, database_=brain_id)
        rows = [
            (
                str(record.get("event_uuid") or ""),
                str(record.get("entity_uuid") or ""),
                str(record.get("entity_name") or ""),
                str(record.get("rel_name") or ""),
            )
            for record in result.records
        ]
        bridges = bridges_from_memberships(entity_event_memberships(rows))
        self.driver.execute_query(
            "MATCH ()-[r:HUB_BRIDGE]->() DELETE r",
            database_=brain_id,
        )
        for bridge in bridges:
            self.driver.execute_query(
                """
                MATCH (a {uuid: $event_a}), (b {uuid: $event_b})
                WHERE 'EVENT' IN labels(a) AND 'EVENT' IN labels(b)
                MERGE (a)-[r:HUB_BRIDGE {
                    shared_entity: $shared_entity
                }]->(b)
                SET r.shared_entity_name = $shared_entity_name,
                    r.weight = $weight,
                    r.uuid = $bridge_uuid
                """,
                parameters_={
                    "event_a": bridge.event_a,
                    "event_b": bridge.event_b,
                    "shared_entity": bridge.shared_entity,
                    "shared_entity_name": bridge.shared_entity_name,
                    "weight": float(bridge.weight),
                    "bridge_uuid": (
                        f"hub-bridge:{bridge.event_a}:{bridge.event_b}:"
                        f"{bridge.shared_entity}"
                    ),
                },
                database_=brain_id,
            )
        return len(bridges)

    def refresh_hub_bridges_for_entities(
        self, brain_id: str, entity_uuids: list[str]
    ) -> int:
        from src.core.saving.hub_bridges import (
            bridges_from_memberships,
            entity_event_memberships,
        )

        touched = sorted({str(u) for u in entity_uuids if u})
        if not touched:
            return self.rebuild_hub_bridge_index(brain_id)

        self.ensure_database(brain_id)
        cypher_rows = """
        MATCH (event)-[r]-(entity)
        WHERE 'EVENT' IN labels(event)
          AND NOT 'EVENT' IN labels(entity)
          AND type(r) <> 'HUB_BRIDGE'
          AND entity['uuid'] IN $entity_uuids
        RETURN event['uuid'] AS event_uuid,
               entity['uuid'] AS entity_uuid,
               entity['name'] AS entity_name,
               type(r) AS rel_name
        """
        result = self.driver.execute_query(
            cypher_rows,
            parameters_={"entity_uuids": touched},
            database_=brain_id,
        )
        rows = [
            (
                str(record.get("event_uuid") or ""),
                str(record.get("entity_uuid") or ""),
                str(record.get("entity_name") or ""),
                str(record.get("rel_name") or ""),
            )
            for record in result.records
        ]
        bridges = bridges_from_memberships(entity_event_memberships(rows))
        self.driver.execute_query(
            """
            MATCH ()-[r:HUB_BRIDGE]->()
            WHERE r.shared_entity IN $entity_uuids
            DELETE r
            """,
            parameters_={"entity_uuids": touched},
            database_=brain_id,
        )
        for bridge in bridges:
            self.driver.execute_query(
                """
                MATCH (a {uuid: $event_a}), (b {uuid: $event_b})
                WHERE 'EVENT' IN labels(a) AND 'EVENT' IN labels(b)
                MERGE (a)-[r:HUB_BRIDGE {
                    shared_entity: $shared_entity
                }]->(b)
                SET r.shared_entity_name = $shared_entity_name,
                    r.weight = $weight,
                    r.uuid = $bridge_uuid
                """,
                parameters_={
                    "event_a": bridge.event_a,
                    "event_b": bridge.event_b,
                    "shared_entity": bridge.shared_entity,
                    "shared_entity_name": bridge.shared_entity_name,
                    "weight": float(bridge.weight),
                    "bridge_uuid": (
                        f"hub-bridge:{bridge.event_a}:{bridge.event_b}:"
                        f"{bridge.shared_entity}"
                    ),
                },
                database_=brain_id,
            )
        return len(bridges)

    def get_hub_bridges(
        self,
        event_uuids: list[str],
        brain_id: str,
    ) -> list:
        from src.core.saving.hub_bridges import HubBridge

        unique = sorted({str(u) for u in event_uuids if u})
        if not unique:
            return []
        self.ensure_database(brain_id)
        result = self.driver.execute_query(
            """
            MATCH (a)-[r:HUB_BRIDGE]-(b)
            WHERE a['uuid'] IN $uuids OR b['uuid'] IN $uuids
            RETURN a['uuid'] AS event_a,
                   b['uuid'] AS event_b,
                   r.shared_entity AS shared_entity,
                   r.shared_entity_name AS shared_entity_name,
                   r.weight AS weight
            ORDER BY a['uuid'], b['uuid'], r.shared_entity
            """,
            parameters_={"uuids": unique},
            database_=brain_id,
        )
        bridges: list[HubBridge] = []
        seen: set[tuple[str, str, str]] = set()
        for record in result.records:
            a = str(record.get("event_a") or "")
            b = str(record.get("event_b") or "")
            entity = str(record.get("shared_entity") or "")
            if not a or not b or a == b:
                continue
            lo, hi = (a, b) if a < b else (b, a)
            key = (lo, hi, entity)
            if key in seen:
                continue
            seen.add(key)
            bridges.append(
                HubBridge(
                    event_a=lo,
                    event_b=hi,
                    shared_entity=entity,
                    shared_entity_name=str(record.get("shared_entity_name") or ""),
                    weight=float(record.get("weight") or 1.0),
                )
            )
            return bridges

    def get_fact_record(self, fact_id: str, brain_id: str) -> Optional[dict]:
        if not fact_id:
            return None
        params = {"uuid": fact_id}
        rel_query = (
            "MATCH (a)-[r]->(b)\n"
            "WHERE r['uuid'] = $uuid\n"
            "RETURN 'relationship' AS kind,\n"
            "       r['uuid'] AS uuid,\n"
            "       type(r) AS name,\n"
            "       coalesce(r['description'], '') AS description,\n"
            "       properties(r) AS properties,\n"
            "       a['uuid'] AS subject_uuid,\n"
            "       a['name'] AS subject_name,\n"
            "       labels(a) AS subject_labels,\n"
            "       b['uuid'] AS object_uuid,\n"
            "       b['name'] AS object_name,\n"
            "       labels(b) AS object_labels,\n"
            "       [] AS labels"
        )
        result = self._run_query(rel_query, brain_id, params)
        if result.records:
            return self._fact_row_from_record(result.records[0])
        node_query = (
            "MATCH (n)\n"
            "WHERE n['uuid'] = $uuid\n"
            "RETURN 'node' AS kind,\n"
            "       n['uuid'] AS uuid,\n"
            "       n['name'] AS name,\n"
            "       coalesce(n['description'], '') AS description,\n"
            "       properties(n) AS properties,\n"
            "       null AS subject_uuid,\n"
            "       null AS subject_name,\n"
            "       [] AS subject_labels,\n"
            "       null AS object_uuid,\n"
            "       null AS object_name,\n"
            "       [] AS object_labels,\n"
            "       labels(n) AS labels"
        )
        result = self._run_query(node_query, brain_id, params, ensure=False)
        if result.records:
            return self._fact_row_from_record(result.records[0])
        return None

    def _fact_row_from_record(self, record) -> dict:
        row = dict(record)
        row["properties"] = self._python_properties(row.get("properties"))
        return row

    def set_fact_verification(
        self,
        fact_id: str,
        brain_id: str,
        *,
        kind: Literal["relationship", "node"],
        properties: dict,
    ) -> Optional[dict]:
        params = {
            "uuid": fact_id,
            "verification_state": properties.get("verification_state"),
            "verified_by": properties.get("verified_by"),
            "verified_at": properties.get("verified_at"),
            "verification_source": properties.get("verification_source"),
            "verification_note": properties.get("verification_note"),
            "verification_history": self._bind_property_value(
                properties.get("verification_history") or []
            ),
        }
        set_clause = (
            "SET t['verification_state'] = $verification_state,\n"
            "    t['verified_by'] = $verified_by,\n"
            "    t['verified_at'] = $verified_at,\n"
            "    t['verification_source'] = $verification_source,\n"
            "    t['verification_note'] = $verification_note,\n"
            "    t['verification_history'] = $verification_history"
        )
        if kind == "relationship":
            cypher_query = (
                "MATCH ()-[t]->() WHERE t['uuid'] = $uuid\n"
                + set_clause
                + "\nRETURN t['uuid'] AS uuid"
            )
        else:
            cypher_query = (
                "MATCH (t) WHERE t['uuid'] = $uuid\n"
                + set_clause
                + "\nRETURN t['uuid'] AS uuid"
            )
        result = self._run_query(cypher_query, brain_id, params)
        if not result.records:
            return None
        return self.get_fact_record(fact_id, brain_id)

    def list_facts_by_verification(
        self,
        brain_id: str,
        *,
        state: Optional[str] = None,
        include_rejected: bool = False,
        limit: int = 50,
        skip: int = 0,
    ) -> tuple[list[dict], int]:
        params = {
            "state": state,
            "include_rejected": bool(include_rejected),
            "skip": require_limit(skip, name="skip"),
            "limit": require_limit(limit),
        }
        inner = (
            "MATCH (a)-[r]->(b)\n"
            "WHERE type(r) <> 'HUB_BRIDGE'\n"
            "AND ($state IS NULL OR coalesce(r['verification_state'], 'unverified') = $state)\n"
            "AND ($include_rejected OR coalesce(r['verification_state'], 'unverified') <> 'rejected')\n"
            "RETURN 'relationship' AS kind,\n"
            "       r['uuid'] AS uuid,\n"
            "       type(r) AS name,\n"
            "       coalesce(r['description'], '') AS description,\n"
            "       properties(r) AS properties,\n"
            "       a['uuid'] AS subject_uuid,\n"
            "       a['name'] AS subject_name,\n"
            "       labels(a) AS subject_labels,\n"
            "       b['uuid'] AS object_uuid,\n"
            "       b['name'] AS object_name,\n"
            "       labels(b) AS object_labels,\n"
            "       [] AS labels\n"
            "UNION ALL\n"
            "MATCH (n)\n"
            "WHERE 'EVENT' IN labels(n)\n"
            "AND ($state IS NULL OR coalesce(n['verification_state'], 'unverified') = $state)\n"
            "AND ($include_rejected OR coalesce(n['verification_state'], 'unverified') <> 'rejected')\n"
            "RETURN 'node' AS kind,\n"
            "       n['uuid'] AS uuid,\n"
            "       n['name'] AS name,\n"
            "       coalesce(n['description'], '') AS description,\n"
            "       properties(n) AS properties,\n"
            "       null AS subject_uuid,\n"
            "       null AS subject_name,\n"
            "       [] AS subject_labels,\n"
            "       null AS object_uuid,\n"
            "       null AS object_name,\n"
            "       [] AS object_labels,\n"
            "       labels(n) AS labels"
        )
        list_query = (
            "CALL {\n"
            + inner
            + "\n}\n"
            "RETURN kind, uuid, name, description, properties, "
            "subject_uuid, subject_name, subject_labels, "
            "object_uuid, object_name, object_labels, labels\n"
            "ORDER BY uuid\n"
            "SKIP $skip\n"
            "LIMIT $limit"
        )
        count_query = "CALL {\n" + inner + "\n}\nRETURN count(*) AS total"
        result = self._run_query(list_query, brain_id, params)
        count_result = self._run_query(count_query, brain_id, params, ensure=False)
        total = 0
        if count_result and count_result.records:
            total = count_result.records[0].get("total") or 0
        rows = []
        for record in result.records:
            rows.append(self._fact_row_from_record(record))
        return rows, int(total)


_neo4j_client = Neo4jClient()

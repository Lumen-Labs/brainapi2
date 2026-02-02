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
import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict
from neo4j import GraphDatabase
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
from src.utils.logging import log


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

    def execute_operation(self, operation: str, brain_id: str) -> str:
        """
        Execute a Neo4j operation with database override.
        """
        db = brain_id
        return self.driver.execute_query(operation, database_=db)

    def ensure_database(self, database: str) -> None:
        """
        Ensure a database exists.
        """
        if self._verify_database_accessible(database):
            return

        try:
            cypher_query = f"CREATE DATABASE {database}"
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

    def _verify_database_accessible(self, database: str) -> bool:
        """
        Verify that a database is accessible by executing a simple query.
        """
        try:
            self.driver.execute_query("RETURN 1 AS test", database_=database)
            return True
        except ClientError as e:
            error_code = getattr(e, "code", None)
            if error_code == "Neo.ClientError.Database.DatabaseNotFound":
                return False
            raise

    def _execute_query_with_retry(
        self, query: str, database: str, max_retries: int = 3, retry_delay: float = 0.1
    ):
        """
        Execute a query with retry logic for DatabaseNotFound errors.
        Verifies database accessibility instead of using sleep.
        """
        last_exception = None
        for attempt in range(max_retries):
            try:
                return self.driver.execute_query(query, database_=database)
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
        return "The graph database is Neo4j. Cyphter is the language used to operate with it."

    def _clean_labels(self, labels: list[str]) -> list[str]:
        """
        Clean a label to be used in a Cypher query.
        """
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

    def _clean_property_key(self, property_key: str) -> str:
        """
        Clean a property key to be used in a Cypher query.
        """
        words = property_key.split()
        return words[0].lower() + "".join(word.capitalize() for word in words[1:])

    def _format_property_key(self, property_key: str) -> str:
        """
        Format a property key for use in a Cypher query, including sanitization and quoting.
        """
        sanitized_key = property_key.rstrip("`").lstrip("`")
        needs_quoting = any(
            char in sanitized_key
            for char in ["-", " ", ".", "+", "*", "/", "%", ":", "@", "#", "$"]
        )
        cleaned_key = self._clean_property_key(sanitized_key)
        return f"`{cleaned_key}`" if needs_quoting else cleaned_key

    def _format_value(self, value: Any) -> str:
        """
        Format a value for use in a Cypher query, including sanitization and quoting.
        """
        if isinstance(value, str):
            v = value.replace("'", "\\'")
            return f"'{v}'"
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif value is None:
            return "null"
        else:
            v = str(value).replace("'", "\\'")
            return f"'{v}'"

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
            identification_dict = {"name": node.name}

            merged_metadata = {**(metadata or {}), **(node.metadata or {})}
            all_properties = {
                **(node.properties or {}),
                "metadata": merged_metadata or None,
            }

            if identification_params:
                for key, value in identification_params.items():
                    normalized_key = self._clean_property_key(key)
                    if normalized_key != "name":
                        identification_dict[normalized_key] = value

            identification_items = []
            for key, value in identification_dict.items():
                cypher_key = self._format_property_key(key)
                escaped_value = self._format_value(value)
                identification_items.append(f"{cypher_key}: {escaped_value}")

            identification_set_str = f"{{{', '.join(identification_items)}}}"

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

            property_assignments = []
            for key, value in all_properties.items():
                cypher_key = self._format_property_key(key)
                escaped_value = self._format_value(value)
                property_assignments.append(f"n.{cypher_key} = {escaped_value}")

            property_assignments.append(f"n['uuid'] = '{node.uuid}'")

            properties_set = f"{', '.join(property_assignments)}"

            labels_expression = ":".join(self._clean_labels(node.labels))
            cypher_query = f"""
    MERGE (n:{labels_expression} {identification_set_str})
    SET {properties_set}
    RETURN n
            """
            try:
                result = self._execute_query_with_retry(cypher_query, brain_id)
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
                query_result: The raw result returned by the Neo4j driver for the executed Cypher query.
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

        extra_ops = ""
        for obj in objects:
            for attr in attributes:
                value = getattr(obj, attr, None)
                if value:
                    extra_ops += f"""
            SET r['{attr}'] = {self._format_value(value)}
            """

        cypher_query = f"""
        MATCH (a:{":".join(self._clean_labels(subject.labels))}) WHERE a['name'] = {self._format_value(subject.name)}
        MATCH (b:{":".join(self._clean_labels(to_object.labels))}) WHERE b['name'] = {self._format_value(to_object.name)}
        MERGE (a)-[r:{":".join(self._clean_labels([predicate.name]))}]->(b)
        ON CREATE 
        SET r['description'] = {self._format_value(predicate.description)}, 
        r['uuid'] = {self._format_value(predicate.uuid)}, 
        r['v_id'] = {self._format_value(predicate.properties.get("v_id"))},
        r['flow_key'] = {self._format_value(predicate.flow_key)}
        {extra_ops}
        RETURN a, b
        """

        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        return result

    def search_graph(self, nodes: list[Node], brain_id: str) -> list[Node]:
        """
        Search the graph for nodes and 1 degree relationships.
        """
        if not nodes:
            return []

        queries = []
        for node in nodes:
            query = f"""MATCH (n:{":".join(self._clean_labels(node.labels))}) WHERE n['name'] = {self._format_value(node.name)}
    OPTIONAL MATCH (n)-[r*1]-(m)
    RETURN n, r, m"""
            queries.append(query)

        cypher_query = " UNION ".join(queries)
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        return result

    def node_text_search(self, text: str, brain_id: str) -> list[Node]:
        """
        Search the graph for nodes by partial text match into the name of the nodes.
        """
        cypher_query = f"""
        MATCH (n) 
        WHERE toLower(n['name']) CONTAINS toLower({self._format_value(text)})
        RETURN n
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
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
        cypher_query = f"""
        MATCH (n) WHERE n['uuid'] IN ["{'","'.join(uuids)}"]
        """

        if with_relationships:
            if relationships_type and len(relationships_type) > 0:
                rel_pattern = "|".join([f"`{rt}`" for rt in relationships_type])
                cypher_query += f"""
                            OPTIONAL MATCH (n)-[r:{rel_pattern}*1..{relationships_depth}]-(m)
                            """
            else:
                cypher_query += f"""
                OPTIONAL MATCH (n)-[r*1..{relationships_depth}]-(m)
                """
            if preferred_labels and len(preferred_labels) > 0:
                cypher_query += f"""
                WHERE any(lbl IN labels(m) WHERE lbl IN ["{'","'.join(self._clean_labels(preferred_labels))}"])
                """
            cypher_query += """
            WITH n, r, m
            WHERE r IS NOT NULL OR m IS NOT NULL
            """

        cypher_query += """
        RETURN 
            n['uuid'] as uuid, n['name'] as name, labels(n) as labels, n['description'] as description, properties(n) as properties
        """

        if with_relationships:
            cypher_query += ", r, m['uuid'] as m_uuid, m['name'] as m_name, labels(m) as m_labels, m['description'] as m_description, properties(m) as m_properties"

        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)

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
            properties=result.records[0].get("properties", {}) or {},
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
        cypher_query = f"""
        MATCH (n) WHERE n['uuid'] IN ["{'","'.join(uuids)}"]
        RETURN n['uuid'] as uuid, n['name'] as name, labels(n) as labels, n['description'] as description,
        properties(n) as properties,
        n['polarity'] as polarity, n['happened_at'] as happened_at, n['last_updated'] as last_updated,
        n['observations'] as observations, n['metadata'] as metadata
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        return [
            Node(
                uuid=record.get("uuid", ""),
                name=record.get("name", ""),
                labels=record.get("labels", []),
                description=record.get("description", ""),
                properties=record.get("properties", {}),
                polarity=record.get("polarity", "neutral"),
                **(
                    {"happened_at": record.get("happened_at", None)}
                    if record.get("happened_at", None) is not None
                    else {}
                ),
                **(
                    {
                        "last_updated": record.get(
                            "last_updated", datetime.now(timezone.utc)
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
                    {"metadata": record.get("metadata", {})}
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
        query_params = {}

        for idx, (key, value) in enumerate(params_dict.items()):
            param_name = f"param_{idx}"
            where_clauses.append(f"n.{self._format_property_key(key)} = ${param_name}")
            query_params[param_name] = value

        labels_str = (
            (":" + ":".join(self._clean_labels(entity_types))) if entity_types else ""
        )
        where_str = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        cypher_query = f"""
        MATCH (n{labels_str}) {where_str}
        RETURN n['uuid'] as uuid, n['name'] as name, labels(n) as labels, n['description'] as description,
        properties(n) as properties,
        n['polarity'] as polarity, n['happened_at'] as happened_at, n['last_updated'] as last_updated,
        n['observations'] as observations, n['metadata'] as metadata
        """

        self.ensure_database(brain_id)
        result = self.driver.execute_query(
            cypher_query, parameters_=query_params, database_=brain_id
        )
        if not result.records or len(result.records) == 0:
            return None
        return Node(
            uuid=result.records[0].get("uuid", ""),
            name=result.records[0].get("name", "") or "",
            labels=result.records[0].get("labels", []) or [],
            description=result.records[0].get("description", "") or "",
            properties=result.records[0].get("properties", {}) or {},
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

        uuids_list = '", "'.join(node_uuids)
        cypher_query = f"""
        MATCH (n)-[r]-(c)
        WHERE n['uuid'] IN ["{uuids_list}"]
        """

        if same_type_only:
            cypher_query += " AND size([l IN labels(n) WHERE l IN labels(c)]) > 0"

        if of_types:
            cypher_query += (
                " AND ANY(l IN labels(c) WHERE l IN ["
                + ",".join(f"'{t}'" for t in self._clean_labels(of_types))
                + "])"
            )

        cypher_query += """
        RETURN n['uuid'] as uuid, n['name'] as name, labels(n) as labels, n['description'] as description,
        properties(n) as properties,
        n['polarity'] as polarity, n['happened_at'] as happened_at, n['last_updated'] as last_updated,
        n['observations'] as observations, n['metadata'] as metadata, r AS rel,
        CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END AS direction,
        type(r) AS rel_type, r['description'] AS rel_description, properties(r) AS rel_properties, r['flow_key'] as rel_flowkey, r['uuid'] as rel_uuid,
        c['uuid'] AS c_uuid, c['name'] AS c_name, labels(c) AS c_labels, c['description'] AS c_description, properties(c) AS c_properties
        """
        if limit:
            cypher_query += f" LIMIT {limit}"

        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)

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
        """
        Get the neighbor node tuples by their UUIDs.
        """
        b_uuids_str = ",".join([f'"{nid}"' for nid in b_uuids])
        cypher_query = f"""
        MATCH (n)-[r]-(m)
        WHERE n['uuid'] = '{a_uuid}' AND m['uuid'] IN [{b_uuids_str}]
        RETURN
            n['uuid'] as n_uuid, n['name'] as n_name, labels(n) as n_labels,
            n['description'] as n_description, properties(n) as n_properties,
            m['uuid'] as m_uuid, m['name'] as m_name, labels(m) as m_labels,
            m['description'] as m_description, properties(m) as m_properties, r as rel
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)

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
                        properties=record.get("n_properties", {}) or {},
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
                        properties=record.get("m_properties", {}) or {},
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
        labels_str = ""
        if with_labels:
            labels_list = ",".join([f"'{label}'" for label in with_labels])
            labels_str = f"AND ANY(l IN labels(m) WHERE l IN [{labels_list}])"
        identification_str = ""
        if node:
            identification_str = f"WHERE n['uuid'] = '{node.uuid}'"
        elif uuids:
            uuids_str = ",".join([f'"{uuid}"' for uuid in uuids])
            identification_str = f"WHERE n['uuid'] IN [{uuids_str}]"
        cypher_query = f"""
        MATCH (n)-[r]-(m)
        {identification_str}
        {labels_str}
        RETURN
            m['uuid'] as uuid, m['name'] as name, labels(m) as labels, m['description'] as description, properties(m) as properties,
            r as rel, type(r) as rel_type, r['description'] as rel_description,
            n['uuid'] as n_uuid, n['name'] as n_name, labels(n) as n_labels, n['description'] as n_description, properties(n) as n_properties,
            CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END AS direction
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        return [
            (
                Node(
                    uuid=record.get("uuid", ""),
                    name=record.get("name", "") or "",
                    labels=record.get("labels", []) or [],
                    description=record.get("description", "") or "",
                    properties=record.get("properties", {}) or {},
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
                    properties=record.get("n_properties", {}) or {},
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
        if relationship_types:
            filters.append(
                "type(r) IN [" + ",".join(f"'{t}'" for t in relationship_types) + "]"
            )
        if from_node_labels:
            filters.append(
                "ANY(lbl IN labels(n) WHERE lbl IN ["
                + ",".join(
                    f"'{label}'" for label in self._clean_labels(from_node_labels)
                )
                + "])"
            )
        if to_node_labels:
            filters.append(
                "ANY(lbl IN labels(m) WHERE lbl IN ["
                + ",".join(f"'{label}'" for label in self._clean_labels(to_node_labels))
                + "])"
            )
        if relationship_uuids:
            filters.append(
                "r['uuid'] IN [" + ",".join(f"'{u}'" for u in relationship_uuids) + "]"
            )
        if query_text:
            if query_search_target == "all":
                filters.append(
                    f"(toLower(coalesce(n['name'], n['name'], '')) CONTAINS toLower('{query_text}') OR "
                    f"toLower(coalesce(m['name'], m['name'], '')) CONTAINS toLower('{query_text}') OR "
                    f"toLower(coalesce(r['description'], r['description'], '')) CONTAINS toLower('{query_text}'))"
                )
            elif query_search_target == "node_name":
                filters.append(f"toLower(n['name']) CONTAINS toLower('{query_text}')")
            elif query_search_target == "relationship_description":
                filters.append(
                    f"toLower(r['description']) CONTAINS toLower('{query_text}')"
                )
            elif query_search_target == "relationship_name":
                filters.append(f"toLower(r['name']) CONTAINS toLower('{query_text}')")
        cypher_query = f"""
        MATCH (n)-[r]->(m)
        {"WHERE " + " AND ".join(filters) if filters else ""}
        RETURN n['uuid'] AS n_uuid, n['name'] AS n_name, labels(n) AS n_labels,
            n['description'] AS n_description, properties(n) AS n_properties,
            r AS rel, type(r) AS rel_type, r['description'] AS rel_description,
            m['uuid'] AS m_uuid, m['name'] AS m_name, labels(m) AS m_labels,
            m['description'] AS m_description, properties(m) AS m_properties
        SKIP {skip}
        LIMIT {limit}
        """
        cypher_count = f"""
        MATCH (n)-[r]-(m)
        {"WHERE " + " AND ".join(filters) if filters else ""}
        RETURN count(r) AS total
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        count_result = self.driver.execute_query(cypher_count, database_=brain_id)
        total = 0
        if count_result and count_result.records:
            total = count_result.records[0].get("total") or 0

        triples: list[Triple] = []
        for record in result.records:
            relationship = record.get("rel")
            if relationship is None:
                continue
            subject_properties = dict(record.get("n_properties") or {})
            object_properties = dict(record.get("m_properties") or {})
            relationship_properties = dict(relationship)
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
                        name=str(relationship_type),
                        description=relationship_description or "",
                        direction=direction,
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
        if node_labels:
            filters.append(
                "ANY(lbl IN labels(n) WHERE lbl IN ["
                + ",".join(f"'{label}'" for label in self._clean_labels(node_labels))
                + "])"
            )
        if node_uuids:
            filters.append(
                f"n['uuid'] IN [{','.join(self._format_value(u) for u in node_uuids)}]"
            )
        if query_text:
            filters.append(
                f"(toLower(coalesce(n['name'], n['name'], '')) CONTAINS toLower({self._format_value(query_text)}))"
            )
        cypher_query = f"""
        MATCH (n)
        {"WHERE " + " AND ".join(filters) if filters else ""}
        RETURN n['uuid'] as uuid, n['name'] as name, labels(n) as labels, n['description'] as description,
        properties(n) as properties,
        n['polarity'] as polarity, n['happened_at'] as happened_at, n['last_updated'] as last_updated,
        n['observations'] as observations, n['metadata'] as metadata
        SKIP {skip}
        LIMIT {limit}
        """
        cypher_count = f"""
        MATCH (n)
        {"WHERE " + " AND ".join(filters) if filters else ""}
        RETURN count(n) AS total
        """
        self.ensure_database(brain_id)
        result = self._execute_query_with_retry(cypher_query, brain_id)
        count_result = self._execute_query_with_retry(cypher_count, brain_id)
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
            properties = dict(properties_record)
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
        cypher_query = f"""
        MATCH (a:{":".join(self._clean_labels(subject.labels))}) WHERE a['name'] = '{subject.name}'
        MATCH (b:{":".join(self._clean_labels(object.labels))}) WHERE b['name'] = '{object.name}'
        MATCH (a)-[r:{":".join(self._clean_labels([predicate.name]))}]->(b)
        SET r['deprecated'] = true
        RETURN a['uuid'] as a_uuid, a['name'] as a_name, labels(a) as a_labels, a['description'] as a_description, properties(a) as a_properties,
            b['uuid'] as b_uuid, b['name'] as b_name, labels(b) as b_labels, b['description'] as b_description, properties(b) as b_properties,
            r['uuid'] as r_uuid, type(r) as r_type, r['description'] as r_description, properties(r) as r_properties
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        if result.records:
            return (
                Node(
                    uuid=result.records[0].get("a_uuid", "") or "",
                    name=result.records[0].get("a_name", "") or "",
                    labels=result.records[0].get("a_labels", []) or [],
                    description=result.records[0].get("a_description", "") or "",
                    properties=result.records[0].get("a_properties", {}) or {},
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
                    properties=result.records[0].get("b_properties", {}) or {},
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

        property_set_operations = []
        for property, value in new_properties.items():
            cypher_key = self._format_property_key(property)
            key_str = cypher_key.strip("`")
            formatted_value = self._format_value(value)
            property_set_operations.append(f"t['{key_str}'] = {formatted_value}")

        property_remove_operations = []
        for property in properties_to_remove:
            cypher_key = self._format_property_key(property)
            key_str = cypher_key.strip("`")
            property_remove_operations.append(f"t['{key_str}']")

        properties_set_op = (
            f"SET {', '.join(property_set_operations)}"
            if property_set_operations
            else ""
        )
        properties_remove_op = (
            f"REMOVE {', '.join(property_remove_operations)}"
            if property_remove_operations
            else ""
        )

        if updating == "node":
            cypher_query = f"""
            MATCH (t) WHERE t['uuid'] = $uuid
            {properties_set_op}
            {properties_remove_op}
            RETURN t['uuid'] as uuid, t['name'] as name, labels(t) as labels, t['description'] as description, properties(t) as properties
            """
        elif updating == "relationship":
            cypher_query = f"""
            MATCH ()-[t]->() WHERE t['uuid'] = $uuid
            {properties_set_op}
            {properties_remove_op}
            RETURN t, type(t) AS rel_type, t['description'] AS rel_description, properties(t) as properties
            """

        self.ensure_database(brain_id)
        result = self.driver.execute_query(
            cypher_query, parameters_={"uuid": uuid}, database_=brain_id
        )

        if result.records:
            if updating == "node":
                return Node(
                    uuid=result.records[0].get("uuid", "") or "",
                    name=result.records[0].get("name", "") or "",
                    labels=result.records[0].get("labels", []) or [],
                    description=result.records[0].get("description", "") or "",
                    properties=result.records[0].get("properties", {}) or {},
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

        existing_node = None
        if new_properties:
            existing_node = self.get_by_uuid(uuid, brain_id)
            if not existing_node:
                return None

        if new_name:
            operations.append(f"n['name'] = {self._format_value(new_name)}")

        if new_description:
            operations.append(
                f"n['description'] = {self._format_value(new_description)}"
            )

        if new_properties and existing_node:
            existing_properties = existing_node.properties or {}

            for property_key, new_value in new_properties.items():
                existing_value = existing_properties.get(property_key)

                if (
                    property_key not in existing_properties
                    or existing_value != new_value
                ):
                    key_str = self._clean_property_key(property_key)
                    formatted_value = self._format_value(new_value)
                    operations.append(f"n['{key_str}'] = {formatted_value}")
        elif new_properties:
            for property_key, value in new_properties.items():
                key_str = self._clean_property_key(property_key)
                formatted_value = self._format_value(value)
                operations.append(f"n['{key_str}'] = {formatted_value}")

        set_clause = f"SET {', '.join(operations)}" if operations else ""

        remove_clause = ""
        if properties_to_remove:
            remove_operations = [
                f"n['{self._clean_property_key(prop)}']"
                for prop in properties_to_remove
            ]
            remove_clause = f"REMOVE {', '.join(remove_operations)}"

        labels_clause = ""
        if new_labels:
            cleaned_labels = self._clean_labels(new_labels)
            node_for_labels = existing_node
            if not node_for_labels:
                node_for_labels = self.get_by_uuid(uuid, brain_id)
            if node_for_labels:
                current_labels = node_for_labels.labels or []
                if current_labels:
                    cleaned_current = self._clean_labels(current_labels)
                    remove_labels = ":".join(cleaned_current)
                    labels_clause = f"REMOVE n:{remove_labels}\n"
            labels_clause += f"SET n:{':'.join(cleaned_labels)}"

        if not set_clause and not remove_clause and not labels_clause:
            return existing_node if existing_node else self.get_by_uuid(uuid, brain_id)

        cypher_query = f"""
        MATCH (n)
        WHERE n['uuid'] = $uuid
        {set_clause}
        {remove_clause}
        {labels_clause}
        RETURN n['uuid'] as uuid, n['name'] as name, labels(n) as labels,
            n['description'] as description, properties(n) as properties
        """

        self.ensure_database(brain_id)
        result = self.driver.execute_query(
            cypher_query, parameters_={"uuid": uuid}, database_=brain_id
        )

        if result.records:
            return Node(
                uuid=result.records[0].get("uuid", "") or "",
                name=result.records[0].get("name", "") or "",
                labels=result.records[0].get("labels", []) or [],
                description=result.records[0].get("description", "") or "",
                properties=result.records[0].get("properties", {}) or {},
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
        cypher_query = f"""
        MATCH (n:{":".join(self._clean_labels(labels))})
        WHERE n['name'] = $name
        AND n['uuid'] = $uuid
        RETURN n
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(
            cypher_query, parameters_={"uuid": uuid, "name": name}, database_=brain_id
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
            CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END AS direction,
            m['uuid'] as m_uuid,
            m['name'] as m_name,
            labels(m) as m_labels,
            m['description'] as m_description,
            properties(m) as m_properties
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
                properties=record.get("m_properties", {}) or {},
            )

            predicate = Predicate(
                name=record.get("rel_type", "") or "",
                description=record.get("rel_description", "") or "",
                direction=record.get("direction", "neutral"),
                properties=record.get("rel_properties", {}) or {},
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
            cypher_query = f"""
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
                        properties=record.get("m_properties", {}) or {},
                        polarity=record.get("m_polarity", "neutral"),
                        metadata=record.get("m_metadata", {}) or {},
                        happened_at=record.get("m_happened_at", "") or "",
                        last_updated=record.get("m_last_updated", "") or "",
                        observations=record.get("m_observations", []) or [],
                    ),
                    Predicate(
                        uuid=record.get("r2_uuid", "") or "",
                        name=record.get("r2_type", "") or "",
                        description=record.get("r2_description", "") or "",
                        direction=record.get("r2_direction", "neutral"),
                        properties=record.get("r2_properties", {}) or {},
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
                        properties=record.get("b_properties", {}) or {},
                        polarity=record.get("b_polarity", "neutral"),
                        metadata=record.get("b_metadata", {}) or {},
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
        cypher_query = f"""
        MATCH (n)-[r]-(m) WHERE r['uuid'] IN [{",".join([f'"{uuid}"' for uuid in uuids])}]
        RETURN 
            n['uuid'] as n_uuid, n['name'] as n_name, labels(n) as n_labels, n['description'] as n_description, properties(n) as n_properties,
            r['uuid'] as r_uuid, type(r) as r_type, r['description'] as r_description, properties(r) as r_properties,
            m['uuid'] as m_uuid, m['name'] as m_name, labels(m) as m_labels, m['description'] as m_description, properties(m) as m_properties
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(
            cypher_query,
            database_=brain_id,
        )
        return [
            (
                Node(
                    uuid=record.get("n_uuid", "") or "",
                    name=record.get("n_name", "") or "",
                    labels=record.get("n_labels", []) or [],
                    description=record.get("n_description", "") or "",
                    properties=record.get("n_properties", {}) or {},
                ),
                Predicate(
                    name=record.get("r_type", "") or "",
                    description=record.get("r_description", "") or "",
                    direction=record.get("r_direction", "neutral"),
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
            for record in result.records
        ]

    def remove_nodes(self, uuids: list[str], brain_id: str) -> list[Node]:
        """
        Remove nodes from the graph.
        """
        cypher_query = f"""
        MATCH (n) WHERE n['uuid'] IN [{",".join([f'"{uuid}"' for uuid in uuids])}]
        WITH n, {{
        uuid: n['uuid'],
        name: n['name'],
        labels: labels(n),
        description: n['description'],
        properties: properties(n),
        polarity: n['polarity'],
        metadata: n['metadata'],
        happened_at: n['happened_at'],
        last_updated: n['last_updated'],
        observations: n['observations']
        }} AS node
        DELETE n
        RETURN node
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(
            cypher_query,
            database_=brain_id,
        )
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
            cypher = f"""
            MATCH (n)-[r]-(m) WHERE r['uuid'] IN [{",".join([self._format_value(u) for u in by_rel_uuid])}]
            WITH n, r, m, {with_fields.strip()}
            DELETE r
            RETURN {return_list}
            """
            result = self.driver.execute_query(cypher, database_=brain_id)
            for record in result.records:
                deleted.append(self._record_to_triple(record))
        for tail, pred, head in by_tip_tail:
            tail_match = self._node_match_cypher("n", tail)
            head_match = self._node_match_cypher("m", head)
            rel_type = pred.get("name")
            if rel_type:
                rel_part = f"[r:{':'.join(self._clean_labels([rel_type]))}]"
            else:
                rel_part = "[r]"
            cypher = f"""
            MATCH {tail_match}
            MATCH {head_match}
            MATCH (n){rel_part}-(m)
            WITH n, r, m, {with_fields.strip()}
            DELETE r
            RETURN {return_list}
            """
            result = self.driver.execute_query(cypher, database_=brain_id)
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

    def _node_match_cypher(self, alias: str, node: NodeDict) -> str:
        if not node:
            return f"({alias})"
        if node.get("uuid"):
            return (
                f"({alias}) WHERE {alias}['uuid'] = {self._format_value(node['uuid'])}"
            )
        labels = node.get("labels") or []
        name = node.get("name") or ""
        if labels:
            labels_part = ":".join(self._clean_labels(labels))
            return f"({alias}:{labels_part}) WHERE {alias}['name'] = {self._format_value(name)}"
        if name:
            return f"({alias}) WHERE {alias}['name'] = {self._format_value(name)}"
        return f"({alias})"

    def list_relationships(
        self, subject: str, object: str, brain_id: str
    ) -> list[Tuple[Node, Predicate, Node]]:
        """
        List the relationships between the subject and object.
        """
        cypher_query = f"""
        MATCH (n)-[r]-(m) WHERE n['uuid'] = {self._format_value(subject)} AND m['uuid'] = {self._format_value(object)}
        RETURN n['uuid'] as n_uuid, n['name'] as n_name, labels(n) as n_labels, n['description'] as n_description, properties(n) as n_properties, n['polarity'] as n_polarity, n['metadata'] as n_metadata, n['happened_at'] as n_happened_at, n['last_updated'] as n_last_updated, n['observations'] as n_observations,
        r['uuid'] as r_uuid, type(r) as r_type, r['description'] as r_description, properties(r) as r_properties, r['flow_key'] as r_flow_key, r['last_updated'] as r_last_updated, r['observations'] as r_observations, r['amount'] as r_amount,
        m['uuid'] as m_uuid, m['name'] as m_name, labels(m) as m_labels, m['description'] as m_description, properties(m) as m_properties, m['polarity'] as m_polarity, m['metadata'] as m_metadata, m['happened_at'] as m_happened_at, m['last_updated'] as m_last_updated, m['observations'] as m_observations
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        return [
            (
                Node(
                    uuid=record.get("n_uuid", "") or "",
                    name=record.get("n_name", "") or "",
                    labels=record.get("n_labels", []) or [],
                    description=record.get("n_description", "") or "",
                    properties=record.get("n_properties", {}) or {},
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
                    properties=record.get("r_properties", {}) or {},
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
                    properties=record.get("m_properties", {}) or {},
                    polarity=record.get("m_polarity", "neutral"),
                    metadata=record.get("m_metadata", {}) or {},
                    happened_at=record.get("m_happened_at", "") or "",
                    last_updated=record.get("m_last_updated", "") or "",
                    observations=record.get("m_observations", []) or [],
                ),
            )
            for record in result.records
        ]


_neo4j_client = Neo4jClient()

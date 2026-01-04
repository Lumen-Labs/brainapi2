"""
File: /client.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 12:58:06 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
from src.adapters.interfaces.embeddings import VectorStoreClient
from src.adapters.interfaces.graph import GraphClient
from src.config import config
from src.constants.kg import (
    IdentificationParams,
    Node,
    Predicate,
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
        try:
            cypher_query = f"CREATE DATABASE {database}"
            self.driver.execute_query(cypher_query, database_="system")
        except Exception as e:
            error_msg = str(e).lower()
            if (
                "already exists" in error_msg
                or "not supported in community edition" in error_msg
                or "unsupported administration command" in error_msg
                or "database not found" in error_msg
                or "graph not found" in error_msg
            ):
                pass
            else:
                raise

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
            label.replace(" ", "_").upper().replace("-", "_").replace(".", "_")
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
        Add nodes to the graph.
        """
        results = []
        self.ensure_database(brain_id)

        for node in nodes:
            identification_dict = {"name": node.name}

            all_properties = {
                **(node.properties or {}),
                **(metadata or {}),
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

            if node.description is not None:
                all_properties["description"] = node.description

            property_assignments = []
            for key, value in all_properties.items():
                cypher_key = self._format_property_key(key)
                escaped_value = self._format_value(value)
                property_assignments.append(f"n.{cypher_key} = {escaped_value}")

            property_assignments.append(f"n.uuid = '{node.uuid}'")

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
                log("Nodes added: ", results)
            except Exception as e:
                print(f"Error adding node: {e} - {cypher_query}")
                raise

        return [
            Node(
                uuid=node.uuid,
                labels=node.labels,
                name=node.name,
                description=node.description,
                properties={**node.properties, **(metadata or {})},
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
        Add a relationship between two nodes to the graph.
        """
        cypher_query = f"""
        MATCH (a:{":".join(self._clean_labels(subject.labels))}) WHERE a.name = {self._format_value(subject.name)}
        MATCH (b:{":".join(self._clean_labels(to_object.labels))}) WHERE b.name = {self._format_value(to_object.name)}
        MERGE (a)-[r:{":".join(self._clean_labels([predicate.name]))}]->(b)
        ON CREATE SET r.description = {self._format_value(predicate.description)}, r.uuid = {self._format_value(predicate.uuid)}
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
            query = f"""MATCH (n:{":".join(self._clean_labels(node.labels))}) WHERE n.name = {self._format_value(node.name)}
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
        WHERE toLower(n.name) CONTAINS toLower({self._format_value(text)})
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
        MATCH (n) WHERE n.uuid IN ["{'","'.join(uuids)}"]
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
            n.uuid as uuid, n.name as name, labels(n) as labels, n.description as description, properties(n) as properties
        """

        if with_relationships:
            cypher_query += ", r, m.uuid as m_uuid, m.name as m_name, labels(m) as m_labels, m.description as m_description, properties(m) as m_properties"

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
        Get a node by its UUID.
        """
        cypher_query = f"""
        MATCH (n) WHERE n.uuid = '{uuid}'
        RETURN n.uuid as uuid, n.name as name, labels(n) as labels, n.description as description, properties(n) as properties
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
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
        Get nodes by their UUIDs.
        """
        cypher_query = f"""
        MATCH (n) WHERE n.uuid IN ["{'","'.join(uuids)}"]
        RETURN n.uuid as uuid, n.name as name, labels(n) as labels, n.description as description, properties(n) as properties
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
        Get a node by its identification params and entity types.
        """

        params_dict = identification_params.model_dump(
            mode="json", exclude={"entity_types"}
        )

        where_clauses = []

        for key, value in params_dict.items():
            where_clauses.append(
                f"n.{self._format_property_key(key)} = {self._format_value(value)}"
            )

        cypher_query = f"""
        MATCH (n{(":" + ":".join(self._clean_labels(entity_types))) if entity_types else ""}) {("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""}
        RETURN n.uuid as uuid, n.name as name, labels(n) as labels, n.description as description, properties(n) as properties
        """

        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        if not result.records or len(result.records) == 0:
            return None
        return Node(
            uuid=result.records[0].get("uuid", ""),
            name=result.records[0].get("name", "") or "",
            labels=result.records[0].get("labels", []) or [],
            description=result.records[0].get("description", "") or "",
            properties=result.records[0].get("properties", {}) or {},
        )

    def get_graph_property_keys(self, brain_id: str) -> list[str]:
        """
        Get the property keys of the graph.
        """
        cypher_query = """
        CALL db.propertyKeys() YIELD propertyKey
        RETURN propertyKey
        """
        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)
        return [record["propertyKey"] for record in result.records]

    def get_neighbors(
        self,
        nodes: list[Node | str],
        brain_id: str,
        same_type_only: bool = False,
        limit: int | None = None,
        of_types: Optional[list[str]] = None,
    ) -> Dict[str, List[Tuple[Predicate, Node]]]:
        """
        Get the neighbors of a node.
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
        WHERE n.uuid IN ["{uuids_list}"]
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
        RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, n.description AS description, properties(n) AS properties, r AS rel,
               CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END AS direction,
               type(r) AS rel_type, r.description AS rel_description,
               c.uuid AS c_uuid, c.name AS c_name, labels(c) AS c_labels, c.description AS c_description, properties(c) AS c_properties
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
        WHERE n.uuid = '{a_uuid}' AND m.uuid IN [{b_uuids_str}]
        RETURN
            n.uuid as n_uuid, n.name as n_name, labels(n) as n_labels,
            n.description as n_description, properties(n) as n_properties,
            m.uuid as m_uuid, m.name as m_name, labels(m) as m_labels,
            m.description as m_description, properties(m) as m_properties, r as rel
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
            identification_str = f"WHERE n.uuid = '{node.uuid}'"
        elif uuids:
            uuids_str = ",".join([f'"{uuid}"' for uuid in uuids])
            identification_str = f"WHERE n.uuid IN [{uuids_str}]"
        cypher_query = f"""
        MATCH (n)-[r]-(m)
        {identification_str}
        {labels_str}
        RETURN
            m.uuid as uuid, m.name as name, labels(m) as labels, m.description as description, properties(m) as properties,
            r as rel, type(r) as rel_type, r.description as rel_description,
            n.uuid as n_uuid, n.name as n_name, labels(n) as n_labels, n.description as n_description, properties(n) as n_properties,
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
                "r.uuid IN [" + ",".join(f"'{u}'" for u in relationship_uuids) + "]"
            )
        if query_text:
            if query_search_target == "all":
                filters.append(
                    f"(toLower(coalesce(n.name, n.name, '')) CONTAINS toLower('{query_text}') OR "
                    f"toLower(coalesce(m.name, m.name, '')) CONTAINS toLower('{query_text}') OR "
                    f"toLower(coalesce(r.description, r.description, '')) CONTAINS toLower('{query_text}'))"
                )
            elif query_search_target == "node_name":
                filters.append(f"toLower(n.name) CONTAINS toLower('{query_text}')")
            elif query_search_target == "relationship_description":
                filters.append(
                    f"toLower(r.description) CONTAINS toLower('{query_text}')"
                )
            elif query_search_target == "relationship_name":
                filters.append(f"toLower(r.name) CONTAINS toLower('{query_text}')")
        cypher_query = f"""
        MATCH (n)-[r]->(m)
        {"WHERE " + " AND ".join(filters) if filters else ""}
        RETURN n.uuid AS n_uuid, n.name AS n_name, labels(n) AS n_labels,
            n.description AS n_description, properties(n) AS n_properties,
            r AS rel, type(r) AS rel_type, r.description AS rel_description,
            m.uuid AS m_uuid, m.name AS m_name, labels(m) AS m_labels,
            m.description AS m_description, properties(m) AS m_properties
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
            filters.append(f"n.uuid IN [{','.join(node_uuids)}]")
        if query_text:
            filters.append(
                f"(toLower(coalesce(n.name, n.name, '')) CONTAINS toLower({self._format_value(query_text)}))"
            )
        cypher_query = f"""
        MATCH (n)
        {"WHERE " + " AND ".join(filters) if filters else ""}
        RETURN n.uuid as uuid, n.name as name, labels(n) as labels, n.description as description, properties(n) as properties
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
        MATCH (a:{":".join(self._clean_labels(subject.labels))}) WHERE a.name = '{subject.name}'
        MATCH (b:{":".join(self._clean_labels(object.labels))}) WHERE b.name = '{object.name}'
        MATCH (a)-[r:{":".join(self._clean_labels([predicate.name]))}]->(b)
        SET r.deprecated = true
        RETURN a.uuid as a_uuid, a.name as a_name, labels(a) as a_labels, a.description as a_description, properties(a) as a_properties,
            b.uuid as b_uuid, b.name as b_name, labels(b) as b_labels, b.description as b_description, properties(b) as b_properties,
            r.uuid as r_uuid, type(r) as r_type, r.description as r_description, properties(r) as r_properties
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
        Update the properties of a node or relationship in the graph.
        """

        property_set_operations = []
        for property, value in new_properties.items():
            cypher_key = self._format_property_key(property)
            formatted_value = self._format_value(value)
            property_set_operations.append(f"t.{cypher_key} = {formatted_value}")

        property_remove_operations = []
        for property in properties_to_remove:
            cypher_key = self._format_property_key(property)
            property_remove_operations.append(f"t.{cypher_key}")

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
            MATCH (t) WHERE t.uuid = '{uuid}'
            {properties_set_op}
            {properties_remove_op}
            RETURN t.uuid as uuid, t.name as name, labels(t) as labels, t.description as description, properties(t) as properties
            """
        elif updating == "relationship":
            cypher_query = f"""
            MATCH ()-[t]->() WHERE t.uuid = '{uuid}'
            {properties_set_op}
            {properties_remove_op}
            RETURN t, type(t) AS rel_type, t.description AS rel_description, properties(t) as properties
            """

        self.ensure_database(brain_id)
        result = self.driver.execute_query(cypher_query, database_=brain_id)

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

    def get_schema(self, brain_id: str) -> dict:
        """
        Get the schema/ontology of the graph.
        Returns all node labels and relationship types in JSON format.
        """
        self.ensure_database(brain_id)

        labels_query = "CALL db.labels() YIELD label RETURN label"
        relationships_query = (
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )

        labels_result = self.driver.execute_query(labels_query, database_=brain_id)
        relationships_result = self.driver.execute_query(
            relationships_query, database_=brain_id
        )

        labels = [record["label"] for record in labels_result.records]
        relationships = [
            record["relationshipType"] for record in relationships_result.records
        ]

        return {"labels": labels, "relationships": relationships}

    def get_2nd_degree_hops(
        self,
        from_: List[str],
        flattened: bool,
        vector_store_adapter: VectorStoreClient,
        brain_id: str,
    ) -> Dict[str, List[Tuple[Predicate, Node, List[Tuple[Predicate, Node]]]]]:
        return super().get_2nd_degree_hops(
            from_, flattened, vector_store_adapter, brain_id
        )


_neo4j_client = Neo4jClient()

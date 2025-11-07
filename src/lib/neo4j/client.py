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
from typing import Any, Optional, Tuple
from neo4j import GraphDatabase
from src.adapters.interfaces.graph import GraphClient
from src.config import config
from src.constants.kg import IdentificationParams, Node, Predicate, Triple
from src.utils.logging import log


class Neo4jClient(GraphClient):
    """
    Neo4j client with support for multiple databases.
    """

    def __init__(self, database: Optional[str] = None):
        self.driver = GraphDatabase.driver(
            f"bolt://{config.neo4j.host}:{config.neo4j.port}",
            auth=(config.neo4j.username, config.neo4j.password),
        )
        self.database = database or "neo4j"

    def execute_operation(self, operation: str, database: Optional[str] = None) -> str:
        """
        Execute a Neo4j operation with database override.
        """
        db = database or self.database
        return self.driver.execute_query(operation, database_=db)

    def ensure_database(self, database: str) -> None:
        """
        Ensure a database exists.
        """
        try:
            cypher_query = f"CREATE DATABASE {database}"
            self.execute_operation(cypher_query, database="neo4j")
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
        return "".join(word.capitalize() for word in words)

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
        identification_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
        database: Optional[str] = None,
    ) -> list[Node] | str:
        """
        Add nodes to the graph.
        """
        results = []
        for node in nodes:
            identification_dict = {"Name": node.name}

            all_properties = {
                **(node.properties or {}),
                **(metadata or {}),
            }

            if identification_params:
                for key, value in identification_params.items():
                    identification_dict[key] = value

            identification_items = []
            for key, value in identification_dict.items():
                cypher_key = self._format_property_key(key)

                if isinstance(value, str):
                    escaped_value = self._format_value(value)
                    identification_items.append(f"{cypher_key}: {escaped_value}")
                elif isinstance(value, (int, float, bool)):
                    identification_items.append(
                        f"{cypher_key}: {self._format_value(value)}"
                    )
                elif value is None:
                    identification_items.append(f"{cypher_key}: null")
                else:
                    escaped_value = self._format_value(value)
                    identification_items.append(f"{cypher_key}: {escaped_value}")

            identification_set_str = f"{{{', '.join(identification_items)}}}"

            if node.description is not None:
                all_properties["description"] = node.description

            property_assignments = []
            for key, value in all_properties.items():
                cypher_key = self._format_property_key(key)

                if isinstance(value, str):
                    escaped_value = self._format_value(value)
                    property_assignments.append(f"n.{cypher_key} = {escaped_value}")
                elif isinstance(value, (int, float, bool)):
                    property_assignments.append(
                        f"n.{cypher_key} = {self._format_value(value)}"
                    )
                elif value is None:
                    property_assignments.append(f"n.{cypher_key} = null")
                else:
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
            self.ensure_database(database)
            try:
                result = self.driver.execute_query(cypher_query, database_=database)
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
    ) -> str:
        """
        Add a relationship between two nodes to the graph.
        """
        safe_desc = predicate.description.replace("'", "\\'")

        cypher_query = f"""
        MATCH (a:{":".join(self._clean_labels(subject.labels))}) WHERE a.Name = '{subject.name}'
        MATCH (b:{":".join(self._clean_labels(to_object.labels))}) WHERE b.Name = '{to_object.name}'
        CREATE (a)-[:{":".join(self._clean_labels([predicate.name]))} {{ Description: '{safe_desc}' }}]->(b)
        RETURN a, b
        """
        result = self.driver.execute_query(cypher_query)
        return result

    def search_graph(self, nodes: list[Node]) -> list[Node]:
        """
        Search the graph for nodes and 1 degree relationships.
        """
        if not nodes:
            return []

        queries = []
        for node in nodes:
            query = f"""MATCH (n:{":".join(self._clean_labels(node.labels))}) WHERE n.Name = '{node.name}'
    OPTIONAL MATCH (n)-[r*1]-(m)
    RETURN n, r, m"""
            queries.append(query)

        cypher_query = " UNION ".join(queries)
        result = self.driver.execute_query(cypher_query)
        return result

    def node_text_search(self, text: str) -> list[Node]:
        """
        Search the graph for nodes by partial text match into the name of the nodes.
        """
        cypher_query = f"""
        MATCH (n) 
        WHERE toLower(n.Name) CONTAINS toLower('{text}')
        RETURN n
        """
        result = self.driver.execute_query(cypher_query)
        return [
            Node(
                uuid=node["uuid"],
                name=node["name"],
                labels=node["labels"],
                description=node["description"],
                properties=node["properties"],
            )
            for node in result.records
        ]

    def get_nodes_by_uuid(
        self,
        uuids: list[str],
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

        result = self.driver.execute_query(cypher_query)

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

    def get_graph_entities(self) -> list[str]:
        """
        Get the entities of the graph.
        """
        cypher_query = """
        MATCH (n)
        RETURN DISTINCT labels(n) as labels
        """
        result = self.driver.execute_query(cypher_query)
        return [label for record in result.records for label in record["labels"]]

    def get_graph_relationships(self) -> list[str]:
        """
        Get the relationships of the graph.
        """
        cypher_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN relationshipType
        """
        result = self.driver.execute_query(cypher_query)
        return [record["relationshipType"] for record in result.records]

    def get_by_uuid(self, uuid: str) -> Node:
        """
        Get a node by its UUID.
        """
        cypher_query = f"""
        MATCH (n) WHERE n.uuid = '{uuid}'
        RETURN n.uuid as uuid, n.name as name, labels(n) as labels, n.description as description, properties(n) as properties
        """
        result = self.driver.execute_query(cypher_query)
        return (
            Node(
                uuid=result.records[0]["uuid"],
                name=result.records[0]["name"],
                labels=result.records[0]["labels"],
                description=result.records[0]["description"],
                properties=result.records[0]["properties"],
            )
            if result.records
            else None
        )

    def get_by_uuids(self, uuids: list[str]) -> list[Node]:
        """
        Get nodes by their UUIDs.
        """
        cypher_query = f"""
        MATCH (n) WHERE n.uuid IN ["{'","'.join(uuids)}"]
        RETURN n.uuid as uuid, n.name as name, labels(n) as labels, n.description as description, properties(n) as properties
        """
        result = self.driver.execute_query(cypher_query)
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

    def get_by_identification_params(
        self,
        identification_params: IdentificationParams,
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
            where_clauses.append(f"n.{key} = '{value}'")

        cypher_query = f"""
        MATCH (n{(":" + ":".join(self._clean_labels(entity_types))) if entity_types else ""}) {("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""}
        RETURN n.uuid as uuid, n.name as name, labels(n) as labels, n.description as description, properties(n) as properties
        """
        result = self.driver.execute_query(cypher_query)
        return (
            Node(
                uuid=result.records[0]["uuid"],
                name=result.records[0]["name"],
                labels=result.records[0]["labels"],
                description=result.records[0]["description"],
                properties=result.records[0]["properties"],
            )
            if result.records
            else None
        )

    def get_graph_property_keys(self) -> list[str]:
        """
        Get the property keys of the graph.
        """
        cypher_query = """
        CALL db.propertyKeys() YIELD propertyKey
        RETURN propertyKey
        """
        result = self.driver.execute_query(cypher_query)
        return [record["propertyKey"] for record in result.records]

    def get_neighbors(
        self, node: Node, limit: int
    ) -> list[Tuple[Node, Predicate, Node]]:
        """
        Get the neighbors of a node.
        """

        cypher_query = f"""
        MATCH (n {{uuid: '{node.uuid}'}})-[r]-(c)-[r2]-(m)
        WHERE size([l IN labels(n) WHERE l IN labels(m)]) > 0
        RETURN m.uuid AS uuid, m.name AS name, labels(m) AS labels, m.description AS description, properties(m) AS properties, r2 AS rel,
               CASE WHEN startNode(r2) = c THEN 'out' ELSE 'in' END AS direction,
               c.uuid AS c_uuid, c.name AS c_name, labels(c) AS c_labels, c.description AS c_description, properties(c) AS c_properties
        """
        result = self.driver.execute_query(cypher_query)

        neighbors = []
        for record in result.records:
            rel = record["rel"]
            rel_type = getattr(rel, "type", None)
            rel_desc = ""
            try:
                rel_desc = rel.get("description", "")
            except Exception:
                try:
                    rel_desc = rel["description"]
                except Exception:
                    rel_desc = ""
            direction = record.get("direction", "out")

            neighbors.append(
                (
                    Node(
                        uuid=record["uuid"],
                        name=record["name"],
                        labels=record["labels"],
                        description=record["description"],
                        properties=record["properties"],
                    ),
                    Predicate(
                        name=rel_type or "",
                        description=rel_desc or "",
                        direction=direction,
                    ),
                    Node(
                        uuid=record["c_uuid"],
                        name=record["c_name"],
                        labels=record["c_labels"],
                        description=record["c_description"],
                        properties=record["c_properties"],
                    ),
                )
            )
        return [(n, p, c) for n, p, c in neighbors]

    def get_node_with_rel_by_uuid(
        self, rel_ids_with_node_ids: list[tuple[str, str]]
    ) -> list[dict]:
        """
        Get the node with the relationships by their UUIDs.
        """

    def get_neighbor_node_tuples(
        self, a_uuid: str, b_uuids: list[str]
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
        result = self.driver.execute_query(cypher_query)

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
                        uuid=record["n_uuid"],
                        name=record["n_name"],
                        labels=record["n_labels"],
                        description=record["n_description"],
                        properties=record["n_properties"],
                    ),
                    Predicate(
                        name=rel_type,
                        description=rel_desc,
                        direction=direction,
                    ),
                    Node(
                        uuid=record["m_uuid"],
                        name=record["m_name"],
                        labels=record["m_labels"],
                        description=record["m_description"],
                        properties=record["m_properties"],
                    ),
                )
            )

        return tuples

    def get_connected_nodes(
        self,
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
        result = self.driver.execute_query(cypher_query)
        return [
            (
                Node(
                    uuid=record["uuid"],
                    name=record["name"],
                    labels=record["labels"],
                    description=record["description"],
                    properties=record["properties"],
                ),
                Predicate(
                    name=record["rel_type"] or "",
                    description=record["rel_description"] or "",
                    direction=record.get("direction", "neutral"),
                ),
                Node(
                    uuid=record["n_uuid"],
                    name=record["n_name"],
                    labels=record["n_labels"],
                    description=record["n_description"],
                    properties=record["n_properties"],
                ),
            )
            for record in result.records
        ]

    def search_relationships(self, limit: int = 10, skip: int = 0) -> list[Triple]:
        """
        Search the relationships of the graph.
        """
        cypher_query = f"""
        MATCH (n)-[r]-(m)
        RETURN n.uuid as n_uuid, n.name as n_name, labels(n) as n_labels, n.description as n_description, properties(n) as n_properties,
               r as rel, type(r) as rel_type, r.description as rel_description,
               m.uuid as m_uuid, m.name as m_name, labels(m) as m_labels, m.description as m_description, properties(m) as m_properties
        SKIP {skip}
        LIMIT {limit}
        """
        result = self.driver.execute_query(cypher_query)
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
        return triples

    def search_entities(self, limit: int = 10, skip: int = 0) -> list[Node]:
        """
        Search the entities of the graph.
        """
        cypher_query = f"""
        MATCH (n)
        RETURN n.uuid as uuid, n.name as name, labels(n) as labels, n.description as description, properties(n) as properties
        SKIP {skip}
        LIMIT {limit}
        """
        result = self.driver.execute_query(cypher_query)
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
        return nodes


_neo4j_client = Neo4jClient()

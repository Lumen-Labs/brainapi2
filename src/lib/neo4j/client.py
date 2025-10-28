"""
File: /client.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 12:58:06 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import Optional
from neo4j import GraphDatabase
from src.adapters.interfaces.graph import GraphClient
from src.config import config
from src.constants.kg import Node, Predicate


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

    def add_nodes(
        self,
        nodes: list[Node],
        identification_params: dict,
        metadata: dict,
        database: Optional[str] = None,
    ) -> list[Node] | str:
        """
        Add nodes to the graph.
        """
        results = []
        for node in nodes:
            identification_set = f"{{name: '{node.name}'}}"

            all_properties = {
                **(node.properties or {}),
                **(metadata or {}),
            }

            if node.description is not None:
                all_properties["description"] = node.description

            property_assignments = []
            for key, value in all_properties.items():
                sanitized_key = key.rstrip("`").lstrip("`")
                needs_quoting = any(
                    char in sanitized_key
                    for char in ["-", " ", ".", "+", "*", "/", "%", ":", "@", "#", "$"]
                )
                cypher_key = f"`{sanitized_key}`" if needs_quoting else sanitized_key

                if isinstance(value, str):
                    escaped_value = value.replace("'", "\\'")
                    property_assignments.append(f"n.{cypher_key} = '{escaped_value}'")
                elif isinstance(value, (int, float, bool)):
                    property_assignments.append(f"n.{cypher_key} = {value}")
                elif value is None:
                    property_assignments.append(f"n.{cypher_key} = null")
                else:
                    escaped_value = str(value).replace("'", "\\'")
                    property_assignments.append(f"n.{cypher_key} = '{escaped_value}'")

            property_assignments.append(f"n.uuid = '{node.uuid}'")

            properties_set = f"{', '.join(property_assignments)}"

            labels_expression = ":".join(
                label.replace(" ", "_") for label in node.labels
            )
            cypher_query = f"""
    MERGE (n:{labels_expression} {identification_set})
    SET {properties_set}
    RETURN n
            """
            self.ensure_database(database)
            result = self.driver.execute_query(cypher_query, database_=database)
            results.append(result)

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
        MATCH (a:{":".join([lb.replace(" ", "") for lb in subject.labels])}) WHERE a.name = '{subject.name}'
        MATCH (b:{":".join([lb.replace(" ", "") for lb in to_object.labels])}) WHERE b.name = '{to_object.name}'
        CREATE (a)-[:{predicate.name.replace(" ", "_").replace("-", "_").upper()} {{ description: '{safe_desc}' }}]->(b)
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
            query = f"""MATCH (n:{":".join(node.labels)}) WHERE n.name = '{node.name}'
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
        WHERE toLower(n.name) CONTAINS toLower('{text}')
        RETURN n.uuid as uuid, n.name as name, labels(n) as labels, n.description as description, properties(n) as properties
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
                WHERE any(lbl IN labels(m) WHERE lbl IN ["{'","'.join(preferred_labels)}"])
                """
            cypher_query += """
            WITH n, r, m
            WHERE r IS NOT NULL OR m IS NOT NULL
            """

        cypher_query += """
        RETURN n.uuid as uuid, n.name as name, labels(n) as labels, n.description as description, properties(n) as properties
        """

        if with_relationships:
            cypher_query += ", r, m"

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
                    "related_nodes": record["m"] if record["m"] else None,
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


_neo4j_client = Neo4jClient()

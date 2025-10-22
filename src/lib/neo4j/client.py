"""
File: /client.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 12:58:06 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from neo4j import GraphDatabase
from src.adapters.interfaces.graph import GraphClient
from src.config import config
from src.constants.kg import Node


class Neo4jClient(GraphClient):
    """
    Neo4j client.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            f"bolt://{config.neo4j.host}:{config.neo4j.port}",
            auth=(config.neo4j.username, config.neo4j.password),
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
        Get the description of the graph database. It is used to let the agent know which syntax to use.
        """
        return "The graph database is Neo4j. Cyphter is the language used to operate with it."

    def execute_operation(self, operation: str) -> str:
        """
        Execute a Neo4j operation.
        """
        return self.driver.execute_query(operation)

    def add_nodes(
        self, nodes: list[Node], identification_params: dict, metadata: dict
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
                if isinstance(value, str):
                    escaped_value = value.replace("'", "\\'")
                    property_assignments.append(f"n.{key} = '{escaped_value}'")
                elif isinstance(value, (int, float, bool)):
                    property_assignments.append(f"n.{key} = {value}")
                elif value is None:
                    property_assignments.append(f"n.{key} = null")
                else:
                    escaped_value = str(value).replace("'", "\\'")
                    property_assignments.append(f"n.{key} = '{escaped_value}'")

            property_assignments.append(f"n.uuid = '{node.uuid}'")

            properties_set = f"{', '.join(property_assignments)}"

            cypher_query = f"""
    MERGE (n:{node.label.replace(" ", "_")} {identification_set})
    SET {properties_set}
    RETURN n
            """
            result = self.driver.execute_query(cypher_query)
            results.append(result)

        return [
            Node(
                uuid=node.uuid,
                label=node.label,
                name=node.name,
                description=node.description,
                properties={**node.properties, **(metadata or {})},
            )
            for node in nodes
        ]


_neo4j_client = Neo4jClient()

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

    def execute_operation(self, operation: str) -> str:
        """
        Execute a Neo4j operation.
        """
        return self.driver.execute_query(operation)


_neo4j_client = Neo4jClient()

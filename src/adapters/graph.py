"""
File: /graph.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 8:59:28 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from src.adapters.interfaces.graph import GraphClient


class GraphAdapter:
    """
    Adapter for the graph client.
    """

    def __init__(self):
        self.graph = None

    @property
    def graphdb_type(self) -> str:
        """
        This is the type of the graph database.
        It is used to let the agent know which syntax to use.
        """
        return self.graph.graphdb_type

    def add_client(self, client: GraphClient) -> None:
        """
        Add a graph client to the adapter.
        """
        self.graph = client

    def execute_operation(self, operation: str) -> str:
        """
        Execute a graph operation.
        """
        return self.graph.execute_operation(operation)


_graph_adapter = GraphAdapter()

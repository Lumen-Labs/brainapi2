"""
File: /graph.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 8:59:28 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import Optional
from src.adapters.interfaces.graph import GraphClient
from src.constants.kg import Node


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

    @property
    def graphdb_description(self) -> str:
        """
        This is the description of the graph database.
        It is used to let the agent know which syntax to use.
        """
        return self.graph.graphdb_description

    def add_client(self, client: GraphClient) -> None:
        """
        Add a graph client to the adapter.
        """
        self.graph = client

    def execute_operation(self, operation: str) -> str:
        """
        Execute a generic graph operation.
        """
        try:
            return self.graph.execute_operation(operation)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error executing graph operation: {e} - {operation}")
            return f"Error executing graph operation: {e}"

    def add_nodes(
        self,
        nodes: list[Node],
        identification_params: Optional[dict],
        metadata: Optional[dict],
    ) -> list[Node] | str:
        """
        Add nodes to the graph.
        """
        return self.graph.add_nodes(nodes, identification_params, metadata)

    def add_relationship(
        self,
        subject: Node,
        predicate: str,
        to_object: Node,
    ) -> str:
        """
        Add a relationship between two nodes to the graph.
        """
        return self.graph.add_relationship(subject, predicate, to_object)

    def search_graph(
        self,
        nodes: list[Node],
    ) -> list[Node]:
        """
        Search the graph for nodes and 1 degree relationships.
        """
        return self.graph.search_graph(nodes)


_graph_adapter = GraphAdapter()

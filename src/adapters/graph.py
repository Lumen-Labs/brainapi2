"""
File: /graph.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 8:59:28 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import Optional, Tuple
from src.adapters.interfaces.graph import GraphClient
from src.constants.kg import IdentificationParams, Node, Predicate, Triple


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
        database: Optional[str] = None,
    ) -> list[Node] | str:
        """
        Add nodes to the graph.
        """
        return self.graph.add_nodes(nodes, identification_params, metadata, database)

    def add_relationship(
        self,
        subject: Node,
        predicate: Predicate,
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

    def node_text_search(self, text: str) -> list[Node]:
        """
        Search the graph for nodes by partial text match into the name of the nodes.
        """
        return self.graph.node_text_search(text)

    def get_nodes_by_uuid(
        self,
        uuids: list[str],
        with_relationships: Optional[bool] = False,
        relationships_depth: Optional[int] = 1,
        relationships_type: Optional[list[str]] = None,
        preferred_labels: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Get nodes by their UUIDs with optional relationships and preferred labels.
        """
        return self.graph.get_nodes_by_uuid(
            uuids,
            with_relationships,
            relationships_depth,
            relationships_type,
            preferred_labels,
        )

    def get_graph_entities(self) -> list[str]:
        """
        Get the entities of the graph.
        """
        return self.graph.get_graph_entities()

    def get_graph_relationships(self) -> list[str]:
        """
        Get the relationships of the graph.
        """
        return self.graph.get_graph_relationships()

    def get_graph_property_keys(self) -> list[str]:
        """
        Get the property keys of the graph.
        """
        return self.graph.get_graph_property_keys()

    def get_by_uuid(self, uuid: str) -> Node:
        """
        Get a node by its UUID.
        """
        return self.graph.get_by_uuid(uuid)

    def get_by_uuids(self, uuids: list[str]) -> list[Node]:
        """
        Get nodes by their UUIDs.
        """
        return self.graph.get_by_uuids(uuids)

    def get_by_identification_params(
        self,
        identification_params: IdentificationParams,
        entity_types: Optional[list[str]] = None,
    ) -> Node:
        """
        Get a node by its identification params and entity types.
        """
        return self.graph.get_by_identification_params(
            identification_params, entity_types
        )

    def get_neighbors(
        self, node: Node, limit: int
    ) -> list[Tuple[Node, Predicate, Node]]:
        """
        Get the neighbors of a node.
        """
        return self.graph.get_neighbors(node, limit)

    def get_node_with_rel_by_uuid(
        self, rel_ids_with_node_ids: list[tuple[str, str]]
    ) -> list[dict]:
        """
        Get the node with the relationships by their UUIDs.
        """
        return self.graph.get_node_with_rel_by_uuid(rel_ids_with_node_ids)

    def get_neighbor_node_tuples(
        self, a_uuid: str, b_uuids: list[str]
    ) -> list[Tuple[Node, Predicate, Node]]:
        """
        Get the neighbor node tuples by their UUIDs.
        """
        return self.graph.get_neighbor_node_tuples(a_uuid, b_uuids)

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
        return self.graph.get_connected_nodes(
            node=node, uuids=uuids, limit=limit, with_labels=with_labels
        )


_graph_adapter = GraphAdapter()

"""
File: /graph.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 27th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import Literal, Optional, Tuple
from src.adapters.interfaces.graph import GraphClient
from src.constants.kg import (
    IdentificationParams,
    Node,
    Predicate,
    SearchEntitiesResult,
    SearchRelationshipsResult,
)


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

    def execute_operation(self, operation: str, brain_id: str = "default") -> str:
        """
        Execute a generic graph operation.
        """
        try:
            return self.graph.execute_operation(operation, brain_id)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error executing graph operation: {e} - {operation}")
            return f"Error executing graph operation: {e}"

    def add_nodes(
        self,
        nodes: list[Node],
        brain_id: str = "default",
        identification_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> list[Node] | str:
        """
        Add nodes to the graph.
        """
        return self.graph.add_nodes(nodes, brain_id, identification_params, metadata)

    def add_relationship(
        self,
        subject: Node,
        predicate: Predicate,
        to_object: Node,
        brain_id: str = "default",
    ) -> str:
        """
        Add a relationship between two nodes to the graph.
        """
        return self.graph.add_relationship(subject, predicate, to_object, brain_id)

    def search_graph(
        self,
        nodes: list[Node],
        brain_id: str = "default",
    ) -> list[Node]:
        """
        Search the graph for nodes and 1 degree relationships.
        """
        return self.graph.search_graph(nodes, brain_id)

    def node_text_search(self, text: str, brain_id: str = "default") -> list[Node]:
        """
        Search the graph for nodes by partial text match into the name of the nodes.
        """
        return self.graph.node_text_search(text, brain_id)

    def get_nodes_by_uuid(
        self,
        uuids: list[str],
        brain_id: str = "default",
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
            brain_id,
            with_relationships,
            relationships_depth,
            relationships_type,
            preferred_labels,
        )

    def get_graph_entities(self, brain_id: str = "default") -> list[str]:
        """
        Get the entities of the graph.
        """
        return self.graph.get_graph_entities(brain_id)

    def get_graph_relationships(self, brain_id: str = "default") -> list[str]:
        """
        Get the relationships of the graph.
        """
        return self.graph.get_graph_relationships(brain_id)

    def get_by_uuid(self, uuid: str, brain_id: str = "default") -> Node:
        """
        Get a node by its UUID.
        """
        return self.graph.get_by_uuid(uuid, brain_id)

    def get_by_uuids(self, uuids: list[str], brain_id: str = "default") -> list[Node]:
        """
        Get nodes by their UUIDs.
        """
        return self.graph.get_by_uuids(uuids, brain_id)

    def get_by_identification_params(
        self,
        identification_params: IdentificationParams,
        brain_id: str = "default",
        entity_types: Optional[list[str]] = None,
    ) -> Node:
        """
        Get a node by its identification params and entity types.
        """
        return self.graph.get_by_identification_params(
            identification_params, brain_id, entity_types
        )

    def get_neighbors(
        self, node: Node, limit: int, brain_id: str = "default"
    ) -> list[Tuple[Node, Predicate, Node]]:
        """
        Get the neighbors of a node.
        """
        return self.graph.get_neighbors(node, limit, brain_id)

    def get_node_with_rel_by_uuid(
        self, rel_ids_with_node_ids: list[tuple[str, str]], brain_id: str = "default"
    ) -> list[dict]:
        """
        Get the node with the relationships by their UUIDs.
        """
        return self.graph.get_node_with_rel_by_uuid(rel_ids_with_node_ids, brain_id)

    def get_neighbor_node_tuples(
        self, a_uuid: str, b_uuids: list[str], brain_id: str = "default"
    ) -> list[Tuple[Node, Predicate, Node]]:
        """
        Get the neighbor node tuples by their UUIDs.
        """
        return self.graph.get_neighbor_node_tuples(a_uuid, b_uuids, brain_id)

    def get_connected_nodes(
        self,
        brain_id: str = "default",
        node: Optional[Node] = None,
        uuids: Optional[list[str]] = None,
        limit: Optional[int] = 10,
        with_labels: Optional[list[str]] = None,
    ) -> list[Tuple[Node, Predicate, Node]]:
        """
        Get the connected nodes by their UUIDs.
        """
        return self.graph.get_connected_nodes(
            brain_id, node=node, uuids=uuids, limit=limit, with_labels=with_labels
        )

    def search_relationships(
        self,
        brain_id: str = "default",
        limit: int = 10,
        skip: int = 0,
        relationship_types: Optional[list[str]] = None,
        from_node_labels: Optional[list[str]] = None,
        to_node_labels: Optional[list[str]] = None,
        query_text: Optional[str] = None,
        query_search_target: Optional[str] = "all",
    ) -> SearchRelationshipsResult:
        """
        Search the relationships of the graph.
        """
        relationship_uuids = []
        # TODO: semantic search + src/core/agents/tools/kg_agent/KGAgentAddTripletsTool.py:165
        return self.graph.search_relationships(
            brain_id,
            limit,
            skip,
            relationship_types,
            from_node_labels,
            to_node_labels,
            relationship_uuids,
            query_text,
            query_search_target,
        )

    def search_entities(
        self,
        brain_id: str = "default",
        limit: int = 10,
        skip: int = 0,
        node_labels: Optional[list[str]] = None,
        query_text: Optional[str] = None,
    ) -> SearchEntitiesResult:
        """
        Search the entities of the graph.
        """
        node_uuids = []
        # TODO: semantic search
        return self.graph.search_entities(
            brain_id, limit, skip, node_labels, node_uuids, query_text
        )

    def deprecate_relationship(
        self,
        subject: Node,
        predicate: Predicate,
        object: Node,
        brain_id: str = "default",
    ) -> Tuple[Node, Predicate, Node] | None:
        """
        Deprecate a relationship from the graph.
        """
        return self.graph.deprecate_relationship(subject, predicate, object, brain_id)

    def update_properties(
        self,
        uuid: str,
        updating: Literal["node", "relationship"],
        brain_id: str = "default",
        new_properties: dict = {},
        properties_to_remove: list[str] = [],
    ) -> Node | Predicate | None:
        """
        Update the properties of a node or relationship in the graph.
        """
        return self.graph.update_properties(
            uuid, updating, brain_id, new_properties, properties_to_remove
        )
    
    def get_graph_relationship_types(self, brain_id: str = "default") -> list[str]:
        """
        Get all unique relationship types from the graph.
        """
        return self.graph.get_graph_relationship_types(brain_id)

    def get_graph_node_types(self, brain_id: str = "default") -> list[str]:
        """
        Get all unique node types from the graph.
        """
        return self.graph.get_graph_node_types(brain_id)
    
    def get_graph_node_properties(self, brain_id: str = "default") -> list[str]:
        """
        Get all unique property keys from nodes in the graph.
        """
        return self.graph.get_graph_node_properties(brain_id)
    
    def update_node(
        self,
        uuid: str,
        brain_id: str = "default",
        new_name: Optional[str] = None,
        new_description: Optional[str] = None,
        new_labels: Optional[list[str]] = None,
        new_properties: Optional[dict] = None,
        properties_to_remove: Optional[list[str]] = None,
    ) -> Node | None:
        """
        Update an entity (node) in the graph.
        """
        return self.graph.update_node(
            uuid,
            brain_id,
            new_name,
            new_description,
            new_labels,
            new_properties,
            properties_to_remove,
        )

_graph_adapter = GraphAdapter()

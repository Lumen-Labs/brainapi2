"""
File: /graph.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 27th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple

from src.constants.kg import (
    IdentificationParams,
    Node,
    Predicate,
    SearchEntitiesResult,
    SearchRelationshipsResult,
)


class GraphClient(ABC):
    """
    Abstract base class for graph clients.
    """

    @property
    @abstractmethod
    def graphdb_description(self) -> str:
        """
        Get the description of the graph database.
        """
        raise NotImplementedError("graphdb_description method not implemented")

    @property
    @abstractmethod
    def graphdb_type(self) -> str:
        """
        Get the type of graph database.
        """
        raise NotImplementedError("graphdb_type method not implemented")

    @abstractmethod
    def execute_operation(self, operation: str, brain_id: str) -> str:
        """
        Execute a generic graph operation.
        """
        raise NotImplementedError("execute_operation method not implemented")

    @abstractmethod
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
        raise NotImplementedError("add_nodes method not implemented")

    @abstractmethod
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
        raise NotImplementedError("add_relationship method not implemented")

    @abstractmethod
    def search_graph(
        self,
        nodes: list[Node],
        brain_id: str,
    ) -> list[Node]:
        """
        Search the graph for nodes and 1 degree relationships.
        """
        raise NotImplementedError("search_graph method not implemented")

    @abstractmethod
    def node_text_search(self, text: str, brain_id: str) -> list[Node]:
        """
        Search the graph for nodes by partial text match into the name of the nodes.
        """
        raise NotImplementedError("node_text_search method not implemented")

    @abstractmethod
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
        Get nodes by their UUIDs with optional relationships and preferred labels.
        """
        raise NotImplementedError("get_nodes_by_uuid method not implemented")

    @abstractmethod
    def get_graph_entities(self, brain_id: str) -> list[str]:
        """
        Get the entities of the graph.
        """
        raise NotImplementedError("get_graph_entities method not implemented")

    @abstractmethod
    def get_graph_relationships(self, brain_id: str) -> list[str]:
        """
        Get the relationships of the graph.
        """
        raise NotImplementedError("get_graph_relationships method not implemented")

    @abstractmethod
    def get_by_uuid(
        self,
        uuid: str,
        brain_id: str,
    ) -> Node:
        """
        Get a node by its UUID.
        """
        raise NotImplementedError("get_by_uuid method not implemented")

    @abstractmethod
    def get_by_uuids(
        self,
        uuids: list[str],
        brain_id: str,
    ) -> list[Node]:
        """
        Get nodes by their UUIDs.
        """
        raise NotImplementedError("get_by_uuids method not implemented")

    @abstractmethod
    def get_by_identification_params(
        self,
        identification_params: IdentificationParams,
        brain_id: str,
        entity_types: Optional[list[str]] = None,
    ) -> Node:
        """
        Get a node by its identification params.
        """
        raise NotImplementedError("get_by_identification_params method not implemented")

    @abstractmethod
    def get_neighbors(
        self,
        node: Node,
        limit: int,
        brain_id: str,
    ) -> list[Tuple[Node, Predicate, Node]]:
        """
        Get the neighbors of a node.
        """
        raise NotImplementedError("get_neighbors method not implemented")

    @abstractmethod
    def get_node_with_rel_by_uuid(
        self,
        rel_ids_with_node_ids: list[tuple[str, str]],
        brain_id: str,
    ) -> list[dict]:
        """
        Get the node with the relationships by their UUIDs.
        """
        raise NotImplementedError("get_node_with_rel_by_uuid method not implemented")

    @abstractmethod
    def get_neighbor_node_tuples(
        self,
        a_uuid: str,
        b_uuids: list[str],
        brain_id: str,
    ) -> list[Tuple[Node, Predicate, Node]]:
        """
        Get the neighbor node tuples by their UUIDs.
        """
        raise NotImplementedError("get_neighbor_node_tuples method not implemented")

    @abstractmethod
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
        raise NotImplementedError("get_connected_nodes method not implemented")

    @abstractmethod
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
        query_search_target: Optional[
            str
        ] = "all",  # Search into the relationship desc or node names or relationship desc
    ) -> SearchRelationshipsResult:
        """
        Search the relationships of the graph.
        """
        raise NotImplementedError("search_relationships method not implemented")

    @abstractmethod
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
        raise NotImplementedError("search_entities method not implemented")

    @abstractmethod
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
        raise NotImplementedError("deprecate_relationship method not implemented")

    @abstractmethod
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
        raise NotImplementedError("update_properties method not implemented")

    @abstractmethod
    def get_graph_relationship_types(self, brain_id: str) -> list[str]:
        """
        Get all unique relationship types from the graph.
        """
        raise NotImplementedError("get_graph_relationship_types method not implemented")

    @abstractmethod
    def get_graph_node_types(self, brain_id: str) -> list[str]:
        """
        Get all unique node types from the graph.
        """
        raise NotImplementedError("get_graph_node_types method not implemented")

    @abstractmethod
    def get_graph_node_properties(self, brain_id: str) -> list[str]:
        """
        Get all unique property keys from nodes in the graph.
        """
        raise NotImplementedError("get_graph_node_properties method not implemented")
    
    @abstractmethod
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
        Update an entity (node) in the graph.
        """
        raise NotImplementedError("update_entity method not implemented")
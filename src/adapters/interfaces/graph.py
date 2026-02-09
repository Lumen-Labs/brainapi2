"""
File: /graph.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday February 2nd 2026 10:02:37 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

from src.constants.kg import (
    IdentificationParams,
    Node,
    NodeDict,
    Predicate,
    PredicateDict,
    SearchEntitiesResult,
    SearchRelationshipsResult,
)
from src.adapters.interfaces.embeddings import VectorStoreClient


class PredicateWithFlowKey(TypedDict):
    predicate_uuid: str
    flow_key: str


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
        Retrieve the relationship types present in the specified graph.

        Parameters:
            brain_id (str): Identifier of the brain/graph to query.

        Returns:
            list[str]: Relationship type names present in the graph.
        """
        raise NotImplementedError("get_graph_relationships method not implemented")

    @abstractmethod
    def get_by_uuid(
        self,
        uuid: str,
        brain_id: str,
    ) -> Node:
        """
        Retrieve a node identified by its UUID from the specified brain.

        Parameters:
            uuid (str): The node's UUID.
            brain_id (str): Identifier of the brain/graph to query.

        Returns:
            Node: The node matching the given UUID.
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
        nodes: list[Node | str],
        brain_id: str,
        same_type_only: bool = False,
        limit: int | None = None,
        of_types: Optional[list[str]] = None,
    ) -> Dict[str, List[Tuple[Predicate, Node]]]:
        """
        Get the neighbors of a node with their relationships.
        """
        raise NotImplementedError("get_neighbors method not implemented")

    @abstractmethod
    def get_event_centric_neighbors(
        self,
        nodes: list[Node | str],
        brain_id: str,
    ) -> List[Tuple[Node, Predicate, Node, Predicate, Node]]:
        """
        Get the event-centric neighbors of a node.
        """
        raise NotImplementedError("get_event_centric_neighbors method not implemented")

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
        Update properties on a node or relationship in the graph.

        Parameters:
            uuid (str): UUID of the target node or relationship.
            updating (Literal["node", "relationship"]): Which entity type to update.
            brain_id (str): Identifier of the graph/brain containing the entity.
            new_properties (dict): Properties to set or overwrite on the entity.
            properties_to_remove (list[str]): Property keys to remove from the entity.

        Returns:
            Node | Predicate | None: The updated node or relationship if changes were applied, `None` if the entity was not found or no update occurred.
        """
        raise NotImplementedError("update_properties method not implemented")

    @abstractmethod
    def get_graph_relationship_types(self, brain_id: str) -> list[str]:
        """
        List unique relationship types present in the specified graph.

        Parameters:
            brain_id (str): Identifier of the graph/brain to query.

        Returns:
            relationship_types (list[str]): Relationship type names present in the graph.
        """
        raise NotImplementedError("get_graph_relationship_types method not implemented")

    @abstractmethod
    def get_graph_node_types(self, brain_id: str) -> list[str]:
        """
        Return all unique node types (labels) present in the graph for the given brain.

        Parameters:
            brain_id (str): Identifier of the brain/graph to query.

        Returns:
            list[str]: A list of node type names (labels) present in the specified graph.
        """
        raise NotImplementedError("get_graph_node_types method not implemented")

    @abstractmethod
    def get_graph_node_properties(self, brain_id: str) -> list[str]:
        """
        Retrieve all unique node property keys present in the graph for the given brain.

        Parameters:
            brain_id (str): Identifier of the graph/brain to query.

        Returns:
            list[str]: Unique property key names found on nodes in the specified graph.
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
        Update a node's identifying fields, labels, and properties.

        Parameters:
            uuid (str): UUID of the node to update.
            brain_id (str): Identifier of the graph or brain containing the node.
            new_name (Optional[str]): New name for the node.
            new_description (Optional[str]): New description for the node.
            new_labels (Optional[list[str]]): New labels to replace existing labels.
            new_properties (Optional[dict]): Properties to add or update on the node.
            properties_to_remove (Optional[list[str]]): Property keys to remove from the node.

        Returns:
            Node | None: The updated node, or `None` if no matching node exists.
        """
        raise NotImplementedError("update_node method not implemented")

    def get_schema(self, brain_id: str) -> dict:
        """
        Get the schema/ontology of the graph.
        """
        raise NotImplementedError("get_schema method not implemented")

    @abstractmethod
    def get_2nd_degree_hops(
        self,
        from_: List[str],
        flattened: bool,
        vector_store_adapter: VectorStoreClient,
        brain_id: str,
    ) -> Dict[str, List[Tuple[Predicate, Node, List[Tuple[Predicate, Node]]]]]:
        """
        Return second-degree hops for each starting node.

        Parameters:
                from_ (List[str]): List of starting node UUIDs or identifiers to expand.
                flattened (bool): If true, return lightweight results with non-essential metadata removed.
                vector_store_adapter (VectorStoreClient): Adapter used to resolve or enrich nodes via the vector store.
                brain_id (str): Identifier of the brain/graph to query.

        Returns:
                hops_by_start (Dict[str, List[Tuple[Predicate, Node, List[Tuple[Predicate, Node]]]]]):
                        Mapping from each starting node identifier in `from_` to a list of first-degree entries.
                        Each first-degree entry is a tuple of:
                          - `Predicate`: relationship from the starting node to the first-hop node,
                          - `Node`: the first-hop node,
                          - `List[Tuple[Predicate, Node]]`: list of second-degree hops where each item is a `(Predicate, Node)` tuple representing the relationship and the second-hop node.
        """
        raise NotImplementedError("get_2nd_degree_hops not implemented")

    @abstractmethod
    def check_node_existence(
        self,
        uuid: str,
        name: str,
        labels: list[str],
        brain_id: str,
    ) -> bool:
        """
        Determine whether a node with the given identity exists in the specified brain.

        Parameters:
            uuid (str): UUID of the node to check.
            name (str): Name of the node to match.
            labels (list[str]): List of labels/types the node must have.
            brain_id (str): Identifier of the brain (graph) to search within.

        Returns:
            true if a node matching the provided uuid, name, and labels exists in the brain, false otherwise.
        """
        raise NotImplementedError("check_node_existence method not implemented")

    @abstractmethod
    def get_neighborhood(
        self, node: Node | str, depth: int, brain_id: str
    ) -> list[dict]:
        """
        Retrieve the neighborhood of a node up to a specified depth.

        Parameters:
            node (Node | str): The starting node or its UUID.
            depth (int): Maximum distance from the starting node to include (1 = immediate neighbors).
            brain_id (str): Identifier of the brain/graph to query.

        Returns:
            list[dict]: A nested list of dictionaries representing the neighborhood; each dictionary represents a node and contains its neighbors as nested structures up to the requested depth.
        """
        raise NotImplementedError("get_neighborhood method not implemented")

    @abstractmethod
    def get_nexts_by_flow_key(
        self, predicates: list[PredicateWithFlowKey], brain_id: str
    ) -> Dict[str, List[Tuple[Node, Predicate, Node]]]:
        """
        Retrieve the next connected node tuple(s) for a relationship identified by a flow key, grouped by the predicate UUID.

        Parameters:
            predicates: list[PredicateWithFlowKey]: A list of predicates with their flow keys.
            brain_id (str): Identifier of the brain/graph to query.

        Returns:
            Dict[str, List[Tuple[Node, Predicate, Node]]]: A dictionary mapping predicate UUIDs to lists of (subject node, predicate, object node) tuples that are the next nodes matching the provided flow key; empty dictionary if none are found for any predicate UUID.
        """
        raise NotImplementedError("get_nexts_by_flow_key method not implemented")

    @abstractmethod
    def get_triples_by_uuid(
        self, uuids: list[str], brain_id: str
    ) -> List[Tuple[Node, Predicate, Node]]:
        """
        Get triples by its UUIDs.
        """
        raise NotImplementedError("get_triples_by_uuid method not implemented")

    @abstractmethod
    def remove_nodes(self, uuids: list[str], brain_id: str) -> list[Node]:
        """
        Remove nodes from the graph.
        """
        raise NotImplementedError("remove_nodes method not implemented")

    @abstractmethod
    def remove_relationships(
        self,
        relationships: list[Tuple[NodeDict, PredicateDict, NodeDict]],
        brain_id: str,
    ) -> list[Tuple[Node, Predicate, Node]]:
        """
        Remove relationships from the graph.
        """
        raise NotImplementedError("remove_relationships method not implemented")

    @abstractmethod
    def list_relationships(
        self,
        brain_id: str,
        subject: str,
        object: str,
    ) -> list[Tuple[Node, Predicate, Node]]:
        """
        List the relationships between the subject and object.
        """
        raise NotImplementedError("list_relationships method not implemented")

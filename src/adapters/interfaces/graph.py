"""
File: /graph.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 19th 2025 9:01:05 am
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.constants.kg import Node, Predicate


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
    def execute_operation(self, operation: str) -> str:
        """
        Execute a generic graph operation.
        """
        raise NotImplementedError("execute_operation method not implemented")

    @abstractmethod
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
        raise NotImplementedError("add_nodes method not implemented")

    @abstractmethod
    def add_relationship(
        self,
        subject: Node,
        predicate: Predicate,
        to_object: Node,
    ) -> str:
        """
        Add a relationship between two nodes to the graph.
        """
        raise NotImplementedError("add_relationship method not implemented")

    @abstractmethod
    def search_graph(
        self,
        nodes: list[Node],
    ) -> list[Node]:
        """
        Search the graph for nodes and 1 degree relationships.
        """
        raise NotImplementedError("search_graph method not implemented")

    @abstractmethod
    def node_text_search(self, text: str) -> list[Node]:
        """
        Search the graph for nodes by partial text match into the name of the nodes.
        """
        raise NotImplementedError("node_text_search method not implemented")

    @abstractmethod
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
        raise NotImplementedError("get_nodes_by_uuid method not implemented")

    @abstractmethod
    def get_graph_entities(self) -> list[str]:
        """
        Get the entities of the graph.
        """
        raise NotImplementedError("get_graph_entities method not implemented")

    @abstractmethod
    def get_by_uuid(
        self,
        uuid: str,
    ) -> Node:
        """
        Get a node by its UUID.
        """
        raise NotImplementedError("get_by_uuid method not implemented")

    @abstractmethod
    def get_by_uuids(
        self,
        uuids: list[str],
    ) -> list[Node]:
        """
        Get nodes by their UUIDs.
        """
        raise NotImplementedError("get_by_uuids method not implemented")

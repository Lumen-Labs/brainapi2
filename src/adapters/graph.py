"""
File: /graph.py
Created Date: Sunday October 19th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 12th 2026 8:26:26 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import Dict, List, Literal, Optional, Tuple
from src.adapters.interfaces.graph import GraphClient
from src.constants.kg import (
    IdentificationParams,
    Node,
    Predicate,
    SearchEntitiesResult,
    SearchRelationshipsResult,
)
from src.adapters.interfaces.embeddings import VectorStoreClient
from src.utils.normalization.list_reduction import reduce_list


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
        Retrieve the relationship types present in the graph.

        Parameters:
            brain_id (str): Identifier of the graph/brain to query.

        Returns:
            A list of relationship type names.
        """
        return self.graph.get_graph_relationships(brain_id)

    def get_by_uuid(self, uuid: str, brain_id: str = "default") -> Node:
        """
        Retrieve the node with the specified UUID from the graph.

        Returns:
            Node: The node matching the provided UUID.
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
        self,
        nodes: list[Node | str],
        same_type_only: bool = False,
        limit: int | None = None,
        of_types: Optional[list[str]] = None,
        brain_id: str = "default",
    ) -> Dict[str, List[Tuple[Predicate, Node]]]:
        """
        Get the neighbors of a node.
        """
        return self.graph.get_neighbors(
            nodes,
            brain_id=brain_id,
            same_type_only=same_type_only,
            limit=limit,
            of_types=of_types,
        )

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
        Update properties on a graph node or relationship.

        Parameters:
                uuid (str): UUID of the node or relationship to update.
                updating (Literal["node", "relationship"]): Target entity type to update.
                brain_id (str): Identifier of the graph (defaults to "default").
                new_properties (dict): Properties to add or replace on the entity.
                properties_to_remove (list[str]): Names of properties to remove from the entity.

        Returns:
                Node | Predicate | None: The updated entity, or `None` if the entity was not found.
        """
        return self.graph.update_properties(
            uuid, updating, brain_id, new_properties, properties_to_remove
        )

    def get_graph_relationship_types(self, brain_id: str = "default") -> list[str]:
        """
        Retrieve all relationship type names present in the graph.

        Parameters:
            brain_id (str): Identifier of the graph (brain) to query. Defaults to "default".

        Returns:
            list[str]: List of unique relationship type names found in the graph.
        """
        return self.graph.get_graph_relationship_types(brain_id)

    def get_graph_node_types(self, brain_id: str = "default") -> list[str]:
        """
        Get all unique node types from the graph.

        @returns:
            A list of node type names available in the graph.
        """
        return self.graph.get_graph_node_types(brain_id)

    def get_graph_node_properties(self, brain_id: str = "default") -> list[str]:
        """
        Retrieve all unique node property keys present in the graph.

        @returns
            list[str]: A list of unique property key names present on nodes for the specified brain.
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
        Update an existing node's identifying fields, labels, and properties in the graph.

        Parameters:
            uuid (str): UUID of the node to update.
            brain_id (str): Identifier of the brain/graph where the node resides.
            new_name (Optional[str]): New name for the node; leave None to keep the current name.
            new_description (Optional[str]): New description for the node; leave None to keep the current description.
            new_labels (Optional[list[str]]): New set of labels for the node; provide to replace the node's labels.
            new_properties (Optional[dict]): Properties to add or update on the node; keys are property names and values are their new values.
            properties_to_remove (Optional[list[str]]): List of property names to remove from the node.

        Returns:
            Node | None: The updated node if the update succeeded, or `None` if the node was not found.
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

    def get_schema(self, brain_id: str = "default") -> dict:
        """
        Get the schema/ontology of the graph.
        """
        return self.graph.get_schema(brain_id)

    def get_2nd_degree_hops(
        self,
        from_uuids: List[str],
        flattened: bool,
        vector_store_adapter: VectorStoreClient,
        brain_id: str = "default",
    ) -> List[Tuple[Node, List[Tuple[Predicate, Node, List[Tuple[Predicate, Node]]]]]]:

        def flatten_node(n):
            return (
                {"uuid": n.uuid, "labels": n.labels, "name": n.name} if flattened else n
            )

        def flatten_pred(p):
            return (
                {"uuid": p.uuid, "name": p.name, "direction": p.direction}
                if flattened
                else p
            )

        nodes = self.get_by_uuids(from_uuids, brain_id)
        nodes_by_uuid = {n.uuid: n for n in nodes}

        vs = vector_store_adapter.get_by_ids(
            [n.properties["v_id"] for n in nodes], brain_id=brain_id, store="nodes"
        )
        averaged_vector = [
            sum(v.embeddings[i] for v in vs) / len(vs)
            for i in range(len(vs[0].embeddings))
        ]

        all_fd_nodes = self.get_neighbors(list(nodes_by_uuid.keys()), brain_id=brain_id)

        all_fd_v_ids = [
            fd[1].properties["v_id"] for fds in all_fd_nodes.values() for fd in fds
        ]
        all_fd_vs = vector_store_adapter.get_by_ids(
            all_fd_v_ids, brain_id=brain_id, store="nodes"
        )
        fd_vs_by_uuid = {v.metadata["uuid"]: v for v in all_fd_vs}

        all_filtered_fd_uuids = []
        filtered_fd_by_origin = {}

        for node_uuid, fd_list in all_fd_nodes.items():
            fd_nodes_by_uuid = {fd[1].uuid: fd[1] for fd in fd_list}
            fd_vs_with_desc = [
                {
                    "embeddings": fd_vs_by_uuid[fd[1].uuid].embeddings,
                    "metadata": fd_vs_by_uuid[fd[1].uuid].metadata,
                    "description": (
                        fd_nodes_by_uuid.get(fd[1].uuid, {}).description
                        if fd_nodes_by_uuid.get(fd[1].uuid)
                        else None
                    ),
                }
                for fd in fd_list
                if fd[1].uuid in fd_vs_by_uuid
            ]
            from_node = nodes_by_uuid[node_uuid]
            filtered = reduce_list(
                fd_vs_with_desc,
                access_key="embeddings",
                similarity_threshold=0.5,
                by_vector=averaged_vector,
                rerank={
                    "local": "description",
                    "with_": from_node.description,
                },
            )
            filtered_uuids = {v["metadata"]["uuid"] for v in filtered}
            filtered_fd_by_origin[node_uuid] = [
                fd for fd in fd_list if fd[1].uuid in filtered_uuids
            ]
            all_filtered_fd_uuids.extend(filtered_uuids)

        all_sd_nodes = self.get_neighbors(all_filtered_fd_uuids, brain_id=brain_id)

        all_sd_v_ids = [
            sd[1].properties["v_id"] for sds in all_sd_nodes.values() for sd in sds
        ]
        all_sd_vs = vector_store_adapter.get_by_ids(
            all_sd_v_ids, brain_id=brain_id, store="nodes"
        )
        sd_vs_by_uuid = {v.metadata["uuid"]: v for v in all_sd_vs}

        hops = []
        exclude_set = set(from_uuids)

        for from_uuid in from_uuids:
            if from_uuid not in nodes_by_uuid:
                continue
            from_node = nodes_by_uuid[from_uuid]
            node_hops = []

            for fd_pred, fd_node in filtered_fd_by_origin.get(from_uuid, []):
                sd_list = all_sd_nodes.get(fd_node.uuid, [])
                sd_nodes_by_uuid = {sd[1].uuid: sd[1] for sd in sd_list}

                sd_vs_with_desc = [
                    {
                        "embeddings": sd_vs_by_uuid[sd[1].uuid].embeddings,
                        "metadata": sd_vs_by_uuid[sd[1].uuid].metadata,
                        "description": (
                            sd_nodes_by_uuid.get(sd[1].uuid, {}).description
                            if sd_nodes_by_uuid.get(sd[1].uuid)
                            else None
                        ),
                    }
                    for sd in sd_list
                    if sd[1].uuid in sd_vs_by_uuid
                ]

                reduced = reduce_list(
                    sd_vs_with_desc,
                    access_key="embeddings",
                    similarity_threshold=0.5,
                    by_vector=averaged_vector,
                    rerank={
                        "local": "description",
                        "with_": from_node.description,
                    },
                )
                reduced_uuids = {v["metadata"]["uuid"] for v in reduced}

                second_degree = [
                    (flatten_pred(sd[0]), flatten_node(sd[1]))
                    for sd in sd_list
                    if sd[1].uuid in reduced_uuids
                    and sd[1].uuid not in exclude_set
                    and sd[1].uuid != from_uuid
                ]

                node_hops.append(
                    (flatten_pred(fd_pred), flatten_node(fd_node), second_degree)
                )

            hops.append((flatten_node(from_node), node_hops))

        return hops

    def check_node_existence(
        self,
        uuid: str,
        name: str,
        labels: list[str],
        brain_id: str = "default",
    ) -> bool:
        """
        Check if a node exists in the graph.
        """
        return self.graph.check_node_existence(uuid, name, labels, brain_id)


_graph_adapter = GraphAdapter()

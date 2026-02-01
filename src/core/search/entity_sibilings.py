"""
File: /entity_sibilings.py
Created Date: Monday January 12th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 12th 2026 9:48:46 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from dataclasses import dataclass
from typing import List, Literal, Tuple

from src.constants.kg import EntitySynergy, Node
from src.services.kg_agent.main import (
    embeddings_adapter,
    graph_adapter,
    vector_store_adapter,
)


class EntitySinergyRetriever:
    def __init__(self, brain_id: str = "default"):
        """
        Initialize the retriever with the identifier of the brain (knowledge graph) to operate against.
        
        Parameters:
            brain_id (str): Identifier of the brain/knowledge graph to use for adapter calls. Defaults to "default".
        """
        self.brain_id = brain_id

    def retrieve_sibilings(
        self, target: str, polarity: Literal["same", "opposite"]
    ) -> Tuple[Node, List[EntitySynergy]]:
        """
        Find related entities ("siblings") for a given target text by locating the target node and assembling EntitySynergy connections from its neighbors and embedding-based similarity.
        
        This returns the resolved target graph node (or None if no matching node is found) and a list of EntitySynergy objects that represent related nodes discovered via neighbor relationships and vector-similarity matches. The method may return early with an empty list when required intermediate data (target node, similarity seeds, or similar nodes) is missing.
        
        Parameters:
            target: The text used to locate the target entity in the node store.
            polarity: A directive indicating the type of relation to retrieve; expected values are "same" or "opposite".
        
        Returns:
            (target_node, connections): 
                - target_node (Node or None): the graph node matching the provided target text, or `None` if not found.
                - connections (List[EntitySynergy]): a list of EntitySynergy entries representing related entities; may be empty.
        """
        target_embedding = embeddings_adapter.embed_text(target)
        target_node_vs = vector_store_adapter.search_vectors(
            target_embedding.embeddings, store="nodes", brain_id=self.brain_id
        )
        if not target_node_vs:
            return None, []
        target_node_id = target_node_vs[0].metadata.get("uuid")
        target_node = graph_adapter.get_by_uuid(target_node_id, brain_id=self.brain_id)
        if not target_node:
            return None, []

        _neighbors = graph_adapter.get_neighbors(
            [target_node_id], brain_id=self.brain_id
        )
        neighbors = _neighbors[target_node_id]

        positivep_connections: List[EntitySynergy] = []
        seen_set = set([target_node.uuid])

        similarity_seed_nodes = []
        neighbor_map = {}

        for neighbor in neighbors:
            seed_nodes_for_neighbor = []

            if "EVENT" in neighbor[1].labels:
                flow_key = neighbor[0].flow_key or neighbor[0].properties.get(
                    "flow_key"
                )
                if not flow_key:
                    continue
                next_rels = graph_adapter.get_next_by_flow_key(
                    neighbor[0].uuid, flow_key, brain_id=self.brain_id
                )
                seed_nodes_for_neighbor = [t[2] for t in next_rels]
            else:
                seed_nodes_for_neighbor = [neighbor[1]]

            for seed_node in seed_nodes_for_neighbor:
                similarity_seed_nodes.append(seed_node)
                if seed_node.uuid not in neighbor_map:
                    neighbor_map[seed_node.uuid] = []
                neighbor_map[seed_node.uuid].append(neighbor[1])

        if not similarity_seed_nodes:
            return target_node, positivep_connections

        for similarity_seed_node in similarity_seed_nodes:
            if any(
                label in similarity_seed_node.labels for label in target_node.labels
            ):
                for connected_by_node in neighbor_map[similarity_seed_node.uuid]:
                    positivep_connections.append(
                        EntitySynergy(
                            node=similarity_seed_node,
                            connected_by=(connected_by_node, connected_by_node),
                        )
                    )
                    seen_set.add(similarity_seed_node.uuid)

        v_ids_to_search = [
            seed_node.properties.get("v_id")
            for seed_node in similarity_seed_nodes
            if seed_node.properties.get("v_id")
        ]

        if not v_ids_to_search:
            return target_node, positivep_connections

        all_similar_dict = vector_store_adapter.search_similar_by_ids(
            v_ids_to_search,
            brain_id=self.brain_id,
            store="nodes",
            min_similarity=0.5,
            limit=10,
        )

        similar_node_uuids = set()
        v_id_to_seed_node = {
            seed_node.properties.get("v_id"): seed_node
            for seed_node in similarity_seed_nodes
            if seed_node.properties.get("v_id")
        }

        for v_id, similar_vectors in all_similar_dict.items():
            for similar_node_v in similar_vectors:
                similar_uuid = similar_node_v.metadata.get("uuid")
                if similar_uuid:
                    similar_node_uuids.add(similar_uuid)

        if not similar_node_uuids:
            return target_node, positivep_connections

        similar_nodes = graph_adapter.get_by_uuids(
            list(similar_node_uuids), brain_id=self.brain_id
        )
        similar_nodes_by_uuid = {node.uuid: node for node in similar_nodes if node}

        if not similar_nodes_by_uuid:
            return target_node, positivep_connections

        all_neighbors = graph_adapter.get_neighbors(
            list(similar_nodes_by_uuid.keys()),
            brain_id=self.brain_id,
            of_types=list(set(target_node.labels)),
        )

        for v_id, similar_vectors in all_similar_dict.items():
            seed_node = v_id_to_seed_node.get(v_id)
            if not seed_node:
                continue

            for similar_node_v in similar_vectors:
                similar_uuid = similar_node_v.metadata.get("uuid")
                similar_node_1 = similar_nodes_by_uuid.get(similar_uuid)

                if not similar_node_1:
                    continue

                neighbors_of_similar = all_neighbors.get(similar_node_1.uuid, [])

                for final_connection_tuple in neighbors_of_similar:
                    final_node_uuid = final_connection_tuple[1].uuid
                    if final_node_uuid in seen_set:
                        continue

                    seen_set.add(final_node_uuid)

                    positivep_connections.append(
                        EntitySynergy(
                            node=final_connection_tuple[1],
                            connected_by=(seed_node, similar_node_1),
                        )
                    )

        return target_node, positivep_connections
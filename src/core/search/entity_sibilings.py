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
        self.brain_id = brain_id

    def retrieve_sibilings(
        self, target: str, polarity: Literal["same", "opposite"]
    ) -> Tuple[Node, List[EntitySynergy]]:
        """
        Retrieve the siblings of an entity.
        """
        target_embedding = embeddings_adapter.embed_text(target)
        target_node_vs = vector_store_adapter.search_vectors(
            target_embedding.embeddings, store="nodes", brain_id=self.brain_id
        )

        target_node_id = target_node_vs[0].metadata.get("uuid")
        target_node = graph_adapter.get_by_uuid(target_node_id, brain_id=self.brain_id)

        if not target_node:
            return None, []

        _neighbors = graph_adapter.get_neighbors(
            [target_node_id], brain_id=self.brain_id
        )
        neighbors = _neighbors[target_node_id]

        positivep_connections: List[EntitySynergy] = []
        seen_set = set()

        for neighbor in neighbors:
            similarity_seed_nodes = []

            if "EVENT" in neighbor[1].labels:
                flow_key = neighbor[0].flow_key or neighbor[0].properties.get(
                    "flow_key"
                )
                if not flow_key:
                    continue
                next_rels = graph_adapter.get_next_by_flow_key(
                    neighbor[0].uuid, flow_key, brain_id=self.brain_id
                )
                similarity_seed_nodes.extend(t[2] for t in next_rels)
            else:
                similarity_seed_nodes.append(neighbor[1])

            for similarity_seed_node in similarity_seed_nodes:
                if target_node.labels in similarity_seed_node.labels:
                    positivep_connections.append(
                        EntitySynergy(
                            node=similarity_seed_node,
                            connected_by=(neighbor[1], neighbor[1]),
                        )
                    )
                    seen_set.add(similarity_seed_node.uuid)

                similar_dict = vector_store_adapter.search_similar_by_ids(
                    [similarity_seed_node.properties["v_id"]],
                    brain_id=self.brain_id,
                    store="nodes",
                    min_similarity=0.5,
                    limit=10,
                )
                similar = similar_dict.get(similarity_seed_node.properties["v_id"], [])

                for similar_node_v in similar:
                    # if similar_node_v.metadata["uuid"] == neighbor[1].uuid:
                    #     continue
                    # if similar_node_v.metadata["uuid"] == similarity_seed_node.uuid:
                    #     continue

                    similar_node_1 = graph_adapter.get_by_uuid(
                        similar_node_v.metadata["uuid"], brain_id=self.brain_id
                    )

                    if not similar_node_1:
                        continue

                    final_positivep_connection = graph_adapter.get_neighbors(
                        [similar_node_1.uuid],
                        brain_id=self.brain_id,
                        of_types=list(set(target_node.labels)),
                    )

                    for final_positivep_connection_tuple in final_positivep_connection[
                        similar_node_1.uuid
                    ]:
                        if final_positivep_connection_tuple[1].uuid in seen_set:
                            continue

                        seen_set.add(final_positivep_connection_tuple[1].uuid)

                        positivep_connections.append(
                            EntitySynergy(
                                node=final_positivep_connection_tuple[1],
                                connected_by=(similarity_seed_node, similar_node_1),
                            )
                        )

        return target_node, positivep_connections

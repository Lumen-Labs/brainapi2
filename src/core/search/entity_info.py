"""
File: /entity_info.py
Created Date: Sunday January 11th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday January 11th 2026 9:52:59 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import List, Set
from datetime import datetime
import numpy as np

from services.kg_agent.main import (
    embeddings_adapter,
    graph_adapter,
    vector_store_adapter,
)
from src.constants.kg import EntityInfo
from src.utils.similarity.vectors import cosine_similarity


# ================================================================
# NOTE Currently not directly supported by the Event-Centric v2 kg
# Need to figure out how to handle this.
# ================================================================


class EventSynergyRetriever:
    def __init__(self, memory_id: str):
        self.memory_id = memory_id
        self.graph_adapter = graph_adapter

    def _recursive_explorer(
        self,
        current_node_id: str,
        query_embedding: List[float],
        depth: int,
        visited_ids: Set[str],
    ):
        # -> List[SynergyPath]:
        """
        Recursively crawls the graph to find high-similarity paths
        beyond the initial Triangle.
        """
        if depth <= 0:
            return []

        neighbors = self.graph_adapter.fetch_entity_with_relevant_nodes(
            current_node_id, list(visited_ids)
        )

        found_paths = []

        neighbor_v_ids = [n.node.properties["v_id"] for n in neighbors.relationships]
        if not neighbor_v_ids:
            return []

        similarities = self._batch_measure_similarity(neighbor_v_ids, query_embedding)

        for i, rel in enumerate(neighbors.relationships):
            sim = similarities[i]
            target_node = rel.to_node

            # if sim > 0.4:
            #     new_visited = visited_ids | {current_node_id}

            #     child_paths = self._recursive_explorer(
            #         target_node.id, query_embedding, depth - 1, new_visited
            #     )

            #     found_paths.append(
            #         SynergyPath(
            #             node=target_node,
            #             relationship=Relationship(
            #                 **rel.model_dump(),
            #                 similarity=sim,
            #                 # Incorporate Recency and Amount into the relationship
            #                 # if it's an Event or Transaction link
            #             ),
            #             similarity=sim,
            #             depth=depth,
            #             children=child_paths,
            #         )
            #     )

        return sorted(found_paths, key=lambda x: x.similarity, reverse=True)

    def retrieve_matches(self, member_uuid: str, query: str, max_depth: int = 3):
        """
        Starts from a Member and recursively finds the most relevant
        synergy paths in the graph.
        """

        query_embedding = embeddings_adapter.embed_text(query)

        synergy_paths = self._recursive_explorer(
            member_uuid, query_embedding, depth=max_depth, visited_ids=set()
        )

        scored_results = []
        for path in synergy_paths:
            child_bonus = sum([c.similarity for c in path.children]) * 0.1

            days_ago = (datetime.now() - path.relationship.happened_at).days
            recency = 1 / (1 + np.log1p(days_ago))

            total_score = (path.similarity * 0.6) + (recency * 0.2) + child_bonus

            scored_results.append({"path": path, "score": total_score})

        return sorted(scored_results, key=lambda x: x["score"], reverse=True)

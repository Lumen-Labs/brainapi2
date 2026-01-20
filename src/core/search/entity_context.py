"""
File: /entity_context.py
Created Date: Sunday January 11th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 12th 2026 8:31:46 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from typing import Tuple
from src.constants.kg import Node
from src.services.kg_agent.main import (
    embeddings_adapter,
    vector_store_adapter,
    graph_adapter,
)


class EntityContext:
    def __init__(self, target: str, brain_id: str = "default"):
        self.target = target
        self.brain_id = brain_id

    def get_context(
        self, context_depth: int = 3
    ) -> Tuple[Node, list[dict], list[dict]]:
        """
        Gets context around an entity.
        """

        text_contexts = set()
        natural_language_web = list()

        target_embedding = embeddings_adapter.embed_text(self.target)
        target_node_vs = vector_store_adapter.search_vectors(
            target_embedding.embeddings, store="nodes", brain_id=self.brain_id
        )

        if not target_node_vs:
            return (None, [])

        target_node_v = target_node_vs[0]
        target_node_id = target_node_v.metadata.get("uuid")

        target_node = graph_adapter.get_by_uuid(target_node_id, brain_id=self.brain_id)

        if not target_node:
            return (None, [])

        neighborhood = graph_adapter.get_neighborhood(
            target_node_id, context_depth, brain_id=self.brain_id
        )

        def _get_text_context(
            neighbors: list[dict], append_to_web: bool = False
        ) -> list[dict]:
            entries = []
            for nn in neighbors:
                node = nn.get("node")
                if node and hasattr(node, "description") and node.description:
                    text_contexts.add(node.description)
                predicate = nn.get("predicate")
                if (
                    predicate
                    and hasattr(predicate, "description")
                    and predicate.description
                ):
                    text_contexts.add(predicate.description)
                nested_neighbors = nn.get("neighbors")
                nested_info = (
                    _get_text_context(nested_neighbors) if nested_neighbors else None
                )
                entry = {
                    "description": predicate.description,
                    "information_direction": predicate.direction,
                    predicate.name: {
                        node.name: node.description,
                        "info": nested_info,
                    },
                }
                if append_to_web:
                    natural_language_web.append(entry)
                entries.append(entry)
            return entries if entries else None

        _get_text_context(neighborhood, append_to_web=True)

        return (target_node, neighborhood, list(text_contexts), natural_language_web)

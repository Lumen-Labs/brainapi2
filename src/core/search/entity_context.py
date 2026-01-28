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
        """
        Initialize the EntityContext with a target entity and knowledge brain identifier.
        
        Parameters:
            target (str): Text of the entity to resolve in the knowledge graph.
            brain_id (str): Identifier of the brain/store to query. Defaults to "default".
        """
        self.target = target
        self.brain_id = brain_id

    def get_context(
        self, context_depth: int = 3
    ) -> Tuple[Node, list[dict], list[dict]]:
        """
        Retrieve contextual information related to the target entity up to a specified depth.
        
        Returns the target node, its neighborhood structure, a list of unique textual descriptions from the node and predicate metadata, and a structured natural language representation of the neighborhood.
          
        @param context_depth The depth of the neighborhood context to retrieve around the target entity.
        
        @returns A tuple containing:
        - The target node object.
        - A list representing the neighborhood of the target node.
        - A list of unique descriptive texts collected from the nodes and predicates.
        - A structured list suitable for natural language processing, representing relationships and descriptions within the neighborhood.
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
            """
            Builds structured textual context entries from a neighborhood of nodes.
            
            Processes each neighbor dictionary to collect node and predicate descriptions into the outer `text_contexts` set, and constructs an entry that includes the predicate description, information direction, and a nested structure mapping predicate and node names to descriptions and further nested info. Nested neighbors are processed recursively and included under the `info` key. If `append_to_web` is True, each entry is appended to the outer `natural_language_web` list.
            
            Parameters:
                neighbors (list[dict]): A list of neighbor mappings; each dict is expected to contain keys like `'node'`, `'predicate'`, and optionally `'neighbors'`.
                append_to_web (bool): If True, append each constructed entry to the module-level `natural_language_web` list.
            
            Returns:
                list[dict] or None: A list of constructed entry dictionaries, or `None` if no entries were produced.
            """
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
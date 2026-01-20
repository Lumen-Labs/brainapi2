"""
File: /entity_info.py
Created Date: Sunday January 11th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 12th 2026 8:31:46 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from __future__ import annotations

from typing import List, Set, Tuple, Optional
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field

from src.constants.kg import Node, Predicate
from src.services.kg_agent.main import graph_adapter
from src.services.kg_agent.main import embeddings_adapter
from src.services.kg_agent.main import vector_store_adapter
from src.utils.similarity.vectors import cosine_similarity


# ================================================================
# NOTE Currently not fully supported by the Event-Centric v2 kg
# Need to figure out how to handle this.
# Currently kinda works but it's raw, unprecise and w/ poor devx
# ================================================================


class MatchPath(BaseModel):
    target_node: Optional[Node] = None
    path: Tuple[Predicate, Optional[Node]]
    similarity: float
    children: List["MatchPath"] = Field(default_factory=list)


class EventSynergyRetriever:
    def __init__(self, memory_id: str):
        self.memory_id = memory_id
        self.brain_id = memory_id

    def _recursive_explorer(
        self,
        current_node_id: str,
        query_embedding: List[float],
        depth: int,
        visited_ids: Set[str],
        most_similar_conn_rel_tuple: Optional[
            Tuple[Tuple[Predicate, Node], float]
        ] = None,
    ):
        # -> List[SynergyPath]:
        """
        Recursively crawls the graph to find high-similarity paths
        beyond the initial Triangle.
        """
        if depth <= 0:
            return []

        if current_node_id in visited_ids:
            return []

        visited_ids.add(current_node_id)

        neighbors = graph_adapter.get_neighbors(
            [current_node_id], brain_id=self.brain_id
        )

        if not neighbors or current_node_id not in neighbors:
            return []

        conn_rels: List[Tuple[Predicate, Node]] = neighbors[current_node_id]

        conn_rels_w_embeddings: List[Tuple[Tuple[Predicate, Node], List[float]]] = []

        for rel_tuple in conn_rels:
            cr = rel_tuple[0]
            v_id = cr.properties.get("v_id")
            if v_id is None:
                continue
            cr_vs = vector_store_adapter.get_by_ids(
                [v_id],
                store="relationships",
                brain_id=self.brain_id,
            )
            if cr_vs:
                conn_rels_w_embeddings.append((rel_tuple, cr_vs[0].embeddings))
        if not conn_rels_w_embeddings:
            return []

        conn_rels_similarities: List[Tuple[Tuple[Predicate, Node], float]] = [
            (rel_tuple, cosine_similarity(cr_embedding, query_embedding))
            for rel_tuple, cr_embedding in conn_rels_w_embeddings
        ]

        conn_rels_similarities_sorted = sorted(
            conn_rels_similarities, key=lambda x: x[1], reverse=True
        )

        if not conn_rels_similarities_sorted:
            return []

        current_best = conn_rels_similarities_sorted[0]

        predicate, node = current_best[0]
        if not node or not predicate:
            return []

        if (
            most_similar_conn_rel_tuple is None
            or current_best[1] > most_similar_conn_rel_tuple[1]
        ):
            most_similar_conn_rel_tuple = current_best

        next_node = current_best[0][1]
        if not next_node or not next_node.uuid:
            return [most_similar_conn_rel_tuple] if most_similar_conn_rel_tuple else []

        results = self._recursive_explorer(
            next_node.uuid,
            query_embedding,
            depth - 1,
            visited_ids,
            most_similar_conn_rel_tuple,
        )

        if results:
            return results
        else:
            return [most_similar_conn_rel_tuple] if most_similar_conn_rel_tuple else []

    def retrieve_matches(
        self, target: str, query: str, max_depth: int = 3
    ) -> MatchPath:
        """
        Starts from a Member and recursively finds the most relevant
        synergy paths in the graph.
        """

        query_embedding = embeddings_adapter.embed_text(query)
        target_embedding = embeddings_adapter.embed_text(target)

        target_node_vs = vector_store_adapter.search_vectors(
            target_embedding.embeddings, store="nodes", brain_id=self.brain_id
        )

        if not target_node_vs:
            return MatchPath(
                target_node=None,
                path=(Predicate(name="", description=""), None),
                similarity=0.0,
                children=[],
            )

        target_node_v = target_node_vs[0]
        target_node_id = target_node_v.metadata.get("uuid")

        if not target_node_id:
            return MatchPath(
                target_node=None,
                path=(Predicate(name="", description=""), None),
                similarity=0.0,
                children=[],
            )

        target_node = graph_adapter.get_by_uuid(target_node_id, brain_id=self.brain_id)

        if not target_node:
            return MatchPath(
                target_node=None,
                path=(Predicate(name="", description=""), None),
                similarity=0.0,
                children=[],
            )

        synergy_paths = self._recursive_explorer(
            target_node_id,
            query_embedding.embeddings,
            depth=max_depth,
            visited_ids=set(),
        )

        if not synergy_paths:
            return MatchPath(
                target_node=target_node,
                path=(Predicate(name="", description=""), target_node),
                similarity=0.0,
                children=[],
            )

        synergy_paths = list(reversed(synergy_paths))

        match_paths = []
        for path_tuple, similarity in synergy_paths:
            predicate, node = path_tuple
            if not node or not predicate:
                continue

            days_ago = 0
            if node.happened_at:
                days_ago = (datetime.now() - node.happened_at).days

            recency = 1 / (1 + np.log1p(days_ago)) if days_ago > 0 else 1.0
            base_score = similarity * 0.6 + recency * 0.2

            match_path = MatchPath(
                target_node=target_node,
                path=path_tuple,
                similarity=base_score,
                children=[],
            )
            match_paths.append(match_path)

        if not match_paths:
            return MatchPath(
                target_node=target_node,
                path=(Predicate(name="", description=""), target_node),
                similarity=0.0,
                children=[],
            )

        root_path = match_paths[0]
        current_path = root_path
        for i in range(1, len(match_paths)):
            child_bonus = match_paths[i].similarity * 0.1
            root_path.similarity += child_bonus
            current_path.children.append(match_paths[i])
            current_path = match_paths[i]

        return root_path

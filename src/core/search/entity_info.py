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
from src.core.search.entity_resolve import resolve_entity_node
from src.lib.tracing.profiler import profile_stage
from src.services.kg_agent.main import graph_adapter
from src.services.kg_agent.main import embeddings_adapter
from src.services.kg_agent.main import vector_store_adapter
from src.utils.similarity.vectors import cosine_similarity
from src.utils.dates import parse_date_string, to_naive_utc

_BRANCH_FACTOR = 3
_MAX_EXPLORATION_WORK = 50


# ================================================================
# NOTE Currently not fully supported by the Event-Centric v2 kg
# Need to figure out how to handle this.
# Currently kinda works but it's raw, unprecise and w/ poor devx
# ================================================================


class MatchPath(BaseModel):
    target_node: Optional[Node] = None
    path: Optional[Tuple[Predicate, Optional[Node]]] = None
    similarity: float
    children: List["MatchPath"] = Field(default_factory=list)


def _recency_score(node: Node) -> float:
    raw = node.happened_at
    if not raw:
        return 1.0
    happened_at = parse_date_string(raw) if isinstance(raw, str) else raw
    if not isinstance(happened_at, datetime):
        print(
            f"[DEBUG (_recency_score)]: unparseable happened_at {raw!r} "
            f"on node {node.uuid}, recency left neutral"
        )
        return 1.0
    days_ago = max(
        0,
        (datetime.now().date() - to_naive_utc(happened_at).date()).days,
    )
    if days_ago <= 0:
        return 1.0
    return 1 / (1 + np.log1p(days_ago))


class EventSynergyRetriever:
    def __init__(self, memory_id: str):
        """
        Initialize the retriever with a memory identifier that is also used as the brain identifier.
        
        Parameters:
            memory_id (str): Identifier for the memory store; assigned to both `memory_id` and `brain_id` on the instance.
        """
        self.memory_id = memory_id
        self.brain_id = memory_id

    def _score_neighbors(
        self,
        current_node_id: str,
        query_embedding: List[float],
    ) -> List[Tuple[Tuple[Predicate, Node], float]]:
        with profile_stage("dossiers.get_neighbors"):
            neighbors = graph_adapter.get_neighbors(
                [current_node_id], brain_id=self.brain_id
            )
        if not neighbors or current_node_id not in neighbors:
            return []

        conn_rels: List[Tuple[Predicate, Node]] = neighbors[current_node_id]
        scored: List[Tuple[Tuple[Predicate, Node], float]] = []
        for rel_tuple in conn_rels:
            cr = rel_tuple[0]
            v_id = cr.properties.get("v_id") if cr.properties else None
            if v_id is None:
                continue
            with profile_stage("dossiers.edge_vector_fetch"):
                cr_vs = vector_store_adapter.get_by_ids(
                    [v_id],
                    store="relationships",
                    brain_id=self.brain_id,
                )
            if not cr_vs or not getattr(cr_vs[0], "embeddings", None):
                continue
            scored.append(
                (rel_tuple, cosine_similarity(cr_vs[0].embeddings, query_embedding))
            )
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _recursive_explorer(
        self,
        current_node_id: str,
        query_embedding: List[float],
        depth: int,
        visited_ids: Set[str],
        work_counter: Optional[List[int]] = None,
        branch_factor: int = _BRANCH_FACTOR,
    ) -> List[Tuple[Tuple[Predicate, Node], float]]:
        """
        Traverse graph neighbors from the given node to locate the connection path most similar to the provided query embedding, searching up to the specified depth.
        
        Parameters:
            current_node_id (str): UUID of the node to start exploration from.
            query_embedding (List[float]): Embedding vector representing the query used to score neighbor relations.
            depth (int): Remaining recursion depth; exploration stops when depth <= 0.
            visited_ids (Set[str]): Set of visited node UUIDs used to avoid cycles on the current path.
            work_counter (Optional[List[int]]): Mutable single-element list tracking explored nodes; stops when cap is reached.
            branch_factor (int): Number of top-scoring neighbors to expand at each hop.
        
        Returns:
            results (List[Tuple[Tuple[Predicate, Node], float]]): Best path as an ordered list of (relation tuple, similarity) hops from the start node, or an empty list if no suitable connections are found.
        """
        if work_counter is None:
            work_counter = [0]

        if depth <= 0 or work_counter[0] >= _MAX_EXPLORATION_WORK:
            return []

        if current_node_id in visited_ids:
            return []

        visited_ids.add(current_node_id)
        work_counter[0] += 1

        scored = self._score_neighbors(current_node_id, query_embedding)
        if not scored:
            return []

        best_path: List[Tuple[Tuple[Predicate, Node], float]] = []
        best_score = float("-inf")

        for rel_tuple, similarity in scored[: max(1, branch_factor)]:
            predicate, node = rel_tuple
            if not node or not predicate or not node.uuid:
                continue

            hop = (rel_tuple, similarity)
            child_path = self._recursive_explorer(
                node.uuid,
                query_embedding,
                depth - 1,
                visited_ids.copy(),
                work_counter,
                branch_factor=branch_factor,
            )
            candidate = [hop] + child_path
            path_score = max(score for _, score in candidate)
            if path_score > best_score:
                best_score = path_score
                best_path = candidate

        return best_path

    def retrieve_matches(
        self, target: str, query: str, max_depth: int = 3
    ) -> MatchPath:
        """
        Finds and assembles the most relevant graph traversal path(s) that link a target entity to content matching a query.
        
        Parameters:
            target (str): Text used to locate the starting node in the node vector store.
            query (str): Text used to compute an embedding that guides relevance scoring of neighboring relationships.
            max_depth (int): Maximum recursion depth when exploring neighboring nodes.
        
        Returns:
            MatchPath: A hierarchical MatchPath rooted at the resolved target node containing the best-match path(s) and an aggregated similarity score. If the target node or any required embeddings are missing, returns a MatchPath with an empty path and similarity 0.0.
        """

        with profile_stage("dossiers.resolve_target"):
            target_node = resolve_entity_node(target, self.brain_id)

        if not target_node:
            return MatchPath(
                target_node=None,
                path=None,
                similarity=0.0,
                children=[],
            )

        target_node_id = target_node.uuid

        with profile_stage("dossiers.embed", embeds=1):
            query_embedding = embeddings_adapter.embed_text(query)

        work_counter = [0]
        with profile_stage("dossiers.explore", max_depth=max_depth) as detail:
            synergy_paths = self._recursive_explorer(
                target_node_id,
                query_embedding.embeddings,
                depth=max_depth,
                visited_ids=set(),
                work_counter=work_counter,
            )
            detail["visited_nodes"] = work_counter[0]

        if not synergy_paths:
            return MatchPath(
                target_node=target_node,
                path=None,
                similarity=0.0,
                children=[],
            )

        match_paths = []
        for path_tuple, similarity in synergy_paths:
            predicate, node = path_tuple
            if not node or not predicate:
                continue

            base_score = similarity * 0.6 + _recency_score(node) * 0.2

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
                path=None,
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

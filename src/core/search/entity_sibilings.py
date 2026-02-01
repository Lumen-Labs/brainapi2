"""
File: /entity_sibilings.py
Created Date: Monday January 12th 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday January 29th 2026 8:43:59 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from dataclasses import dataclass
from typing import List, Literal, Tuple
import numpy as np

from src.adapters.interfaces.graph import PredicateWithFlowKey
from src.constants.kg import EntitySynergy, Node
from src.services.kg_agent.main import (
    embeddings_adapter,
    graph_adapter,
    vector_store_adapter,
)
from src.utils.similarity.vectors import cosine_similarity

DIRECT_MULTIPLIER = 1.30
REMOTE_MULTIPLIER = 0.70


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
    ) -> Tuple[Node, List[EntitySynergy], List[Node], List[Node]]:
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
            return None, [], []
        target_node_id = target_node_vs[0].metadata.get("uuid")
        target_node = graph_adapter.get_by_uuid(target_node_id, brain_id=self.brain_id)
        if not target_node:
            return None, [], []

        _neighbors = graph_adapter.get_neighbors(
            [target_node_id], brain_id=self.brain_id
        )
        neighbors = _neighbors[target_node_id]

        positivep_connections: List[EntitySynergy] = []
        seen_set = set([target_node.uuid])

        all_seed_nodes: List[Node] = []
        all_similar_seeds: List[Node] = []

        target_vid = target_node.properties.get("v_id")
        target_embedding = None
        if target_vid:
            _target_embedding = vector_store_adapter.get_by_ids(
                [target_vid],
                store="nodes",
                brain_id=self.brain_id,
            )
            if _target_embedding:
                target_embedding = _target_embedding[0].embeddings

        for neighbor in neighbors:
            seed_nodes_for_neighbor = []
            target_embeddings = []
            _neighbor_embeddings = vector_store_adapter.get_by_ids(
                [neighbor[0].properties.get("v_id")],
                store="relationships",
                brain_id=self.brain_id,
            )
            if not _neighbor_embeddings:
                continue
            neighbor_embeddings = _neighbor_embeddings[0].embeddings
            if "EVENT" in neighbor[1].labels:
                flow_key = neighbor[0].flow_key or neighbor[0].properties.get(
                    "flow_key"
                )
                if not flow_key:
                    continue
                next_rels = graph_adapter.get_nexts_by_flow_key(
                    [
                        PredicateWithFlowKey(
                            predicate_uuid=neighbor[0].uuid,
                            flow_key=flow_key,
                        )
                    ],
                    brain_id=self.brain_id,
                )
                for t in next_rels[neighbor[0].uuid]:
                    if target_node.uuid in [t[2].uuid, t[0].uuid]:
                        continue
                    v_id = t[1].properties.get("v_id")
                    if not v_id:
                        continue
                    _target_embedding = vector_store_adapter.get_by_ids(
                        [v_id],
                        store="relationships",
                        brain_id=self.brain_id,
                    )
                    if not _target_embedding:
                        continue
                    target_embedding = _target_embedding[0].embeddings
                    target_embeddings.append(
                        np.mean(
                            [target_embedding, neighbor_embeddings],
                            axis=0,
                        )
                    )
                    seed_nodes_for_neighbor.append(t[2])
            else:
                seed_nodes_for_neighbor = [neighbor[1]]
                target_embeddings = neighbor_embeddings

            all_seed_nodes.extend(seed_nodes_for_neighbor)

            print(
                f"[SEED NODES] {[n.name for n in seed_nodes_for_neighbor]}"
            )  # PERFECT

            def _process_seed(seed_node: Node, direct: bool):
                _neighbors_direct = graph_adapter.get_neighbors(
                    [seed_node.uuid],
                    of_types=target_node.labels,
                    brain_id=self.brain_id,
                )

                _neighbors_event_edges = graph_adapter.get_neighbors(
                    [seed_node.uuid],
                    of_types=["EVENT"],
                    brain_id=self.brain_id,
                )
                edges_map = {
                    edge[0].uuid: edge[0]
                    for edge in _neighbors_event_edges[seed_node.uuid]
                }
                _neighbors_event = graph_adapter.get_nexts_by_flow_key(
                    [
                        PredicateWithFlowKey(
                            predicate_uuid=edge_uuid,
                            flow_key=edges_map[edge_uuid].flow_key,
                        )
                        for edge_uuid in edges_map.keys()
                    ],
                    brain_id=self.brain_id,
                )

                _neighbors = []
                for edge_uuid, n in _neighbors_event.items():
                    for e in n:
                        if e[2].uuid != target_node.uuid:
                            if e[2].uuid in seen_set:
                                continue
                            if not set(e[2].labels).intersection(target_node.labels):
                                continue
                            edge_vid = edges_map[edge_uuid].properties.get("v_id")
                            if not edge_vid:
                                continue
                            _edge_embedding = vector_store_adapter.get_by_ids(
                                [edge_vid],
                                store="relationships",
                                brain_id=self.brain_id,
                            )
                            if not _edge_embedding:
                                continue
                            edge_embedding = _edge_embedding[0].embeddings
                            _en_embedding = vector_store_adapter.get_by_ids(
                                [e[1].properties.get("v_id")],
                                store="relationships",
                                brain_id=self.brain_id,
                            )
                            if not _en_embedding:
                                continue
                            en_embedding = _en_embedding[0].embeddings
                            found_embeddings = np.mean(
                                [
                                    en_embedding,
                                    edge_embedding,
                                ],
                                axis=0,
                            )
                            neighbor_vid = e[1].properties.get("v_id")
                            if not neighbor_vid:
                                continue
                            _neighbor_embedding = vector_store_adapter.get_by_ids(
                                [neighbor_vid],
                                store="relationships",
                                brain_id=self.brain_id,
                            )
                            if not _neighbor_embedding:
                                continue
                            neighbor_embedding = _neighbor_embedding[0].embeddings
                            tn_score = cosine_similarity(
                                neighbor_embedding, target_embedding
                            )
                            rel_score = cosine_similarity(
                                found_embeddings, target_embedding
                            )
                            print(
                                f"[{e[2].name}] tn_score: {tn_score}, rel_score: {rel_score}, {DIRECT_MULTIPLIER if direct else REMOTE_MULTIPLIER}"
                            )
                            _neighbors.append(
                                (
                                    (
                                        np.mean([tn_score, rel_score], axis=0)
                                        * (
                                            DIRECT_MULTIPLIER
                                            if direct
                                            else REMOTE_MULTIPLIER
                                        )
                                    ),
                                    e[2],
                                )
                            )

                for dn in _neighbors_direct[seed_node.uuid]:
                    if dn[1].uuid != target_node.uuid:
                        if dn[1].uuid in seen_set:
                            continue
                        n_vid = dn[0].properties.get("v_id")
                        if not n_vid:
                            continue
                        _n_embedding = vector_store_adapter.get_by_ids(
                            [n_vid],
                            store="relationships",
                            brain_id=self.brain_id,
                        )
                        if not _n_embedding:
                            continue
                        n_embedding = _n_embedding[0].embeddings
                        tn_score = cosine_similarity(n_embedding, target_embedding)
                        rel_vid = dn[0].properties.get("v_id")
                        if not rel_vid:
                            continue
                        _rel_embedding = vector_store_adapter.get_by_ids(
                            [rel_vid],
                            store="relationships",
                            brain_id=self.brain_id,
                        )
                        if not _rel_embedding:
                            continue
                        rel_embedding = _rel_embedding[0].embeddings
                        rel_score = cosine_similarity(rel_embedding, target_embedding)
                        _neighbors.append(
                            (
                                (
                                    np.mean([tn_score, rel_score], axis=0)
                                    * (
                                        DIRECT_MULTIPLIER
                                        if direct
                                        else REMOTE_MULTIPLIER
                                    )
                                ),
                                dn[1],
                            )
                        )

                if not _neighbors or len(_neighbors) == 0:
                    return

                for neighbor in _neighbors:
                    positivep_connections.append(
                        EntitySynergy(
                            node=neighbor[1],
                            connected_by=seed_node,
                            association_score=neighbor[0],
                        )
                    )
                    seen_set.add(neighbor[1].uuid)

            for seed_node in seed_nodes_for_neighbor:
                seed_vid = seed_node.properties.get("v_id")
                if not seed_vid:
                    continue
                _seed_embedding = vector_store_adapter.get_by_ids(
                    [seed_vid],
                    store="nodes",
                    brain_id=self.brain_id,
                )
                if not _seed_embedding:
                    continue
                seed_embedding = _seed_embedding[0].embeddings

                similar_seed_vs = vector_store_adapter.search_vectors(
                    seed_embedding,
                    store="nodes",
                    brain_id=self.brain_id,
                )

                similar_seeds = graph_adapter.get_by_uuids(
                    [vs.metadata.get("uuid") for vs in similar_seed_vs],
                    brain_id=self.brain_id,
                )

                _process_seed(seed_node, direct=True)

                all_similar_seeds.extend(similar_seeds)

                for similar_seed in similar_seeds:
                    _process_seed(similar_seed, direct=False)

        unique_seeds = {n.uuid: n for n in all_seed_nodes}
        return (
            target_node,
            sorted(
                positivep_connections, key=lambda x: x.association_score, reverse=True
            ),
            list(unique_seeds.values()),
            all_similar_seeds,
        )

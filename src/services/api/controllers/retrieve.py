"""
File: /retrieve.py
Created Date: Sunday October 26th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday October 26th 2025 4:03:07 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import asyncio
import time
from typing import List, Optional

from fastapi import HTTPException

from src.constants.kg import IdentificationParams, Node, Predicate, Relationship
from src.services.api.constants.requests import (
    RetrieveRequestResponse,
    RetrieveNeighborsRequestResponse,
    RetrievedNeighborNode,
)
from src.services.kg_agent.main import graph_adapter, kg_agent
from src.services.data.main import data_adapter
from src.services.kg_agent.main import embeddings_adapter, vector_store_adapter
from src.utils.logging import log, logt


async def retrieve_data(
    text: str, limit: int, preferred_entities: str
) -> RetrieveRequestResponse:
    """
    Retrieve data from the knowledge graph and data store.
    """
    if preferred_entities:
        preferred_entities = [
            e.strip() for e in preferred_entities.split(",") if e.strip()
        ]
    else:
        preferred_entities = []

    def _get_data():
        text_embeddings = embeddings_adapter.embed_text(text)

        data_vectors = vector_store_adapter.search_vectors(
            text_embeddings.embeddings, "data", limit
        )
        triple_vectors = vector_store_adapter.search_vectors(
            text_embeddings.embeddings, "triplets", limit
        )

        search_result = data_adapter.search(text)

        ts_text_chunks = search_result.text_chunks
        ts_observations = search_result.observations

        v_text_chunks, v_observations = data_adapter.get_text_chunks_by_ids(
            [dv.metadata.get("resource_id") for dv in data_vectors], True
        )

        node_ids = [
            node_id
            for tv in triple_vectors
            for node_id in tv.metadata.get("node_ids", [])
        ]
        nodes = graph_adapter.get_nodes_by_uuid(
            uuids=node_ids,
            with_relationships=True,
            relationships_depth=1,
            relationships_type=[
                tv.metadata.get("predicate")
                for tv in triple_vectors
                if tv.metadata.get("predicate", None)
            ],
            preferred_labels=preferred_entities or [],
        )
        return ts_text_chunks, ts_observations, v_text_chunks, v_observations, nodes

    ts_text_chunks, ts_observations, v_text_chunks, v_observations, nodes = (
        await asyncio.to_thread(_get_data)
    )

    return RetrieveRequestResponse(
        data=[*ts_text_chunks, *v_text_chunks],
        observations=[*ts_observations, *v_observations],
        relationships=nodes,
    )


async def retrieve_neighbors(
    uuid: Optional[str] = None,
    identification_params: Optional[IdentificationParams] = None,
    limit: int = 10,
) -> RetrieveNeighborsRequestResponse:
    def _get_neighbors():

        # ================= GETTING THE MAIN NODE =================
        node = None
        if uuid:
            node = graph_adapter.get_by_uuid(uuid)
        elif identification_params:
            node = graph_adapter.get_by_identification_params(
                identification_params, entity_types=identification_params.entity_types
            )
        if not node:
            raise HTTPException(status_code=404, detail="Entity not found")

        # ================= GETTING THE 2nd LEVEL NEIGHBORS =================
        # This gets the 2nd level neighbors (main)-[r1]-(common)-[r2]-(neighbor)
        # That share a common node with the main node
        # The neighbors are of the same type (share at least one label) of the
        # main node
        raw = graph_adapter.get_neighbors(node, limit)

        normalized: List[RetrievedNeighborNode] = []

        for item in raw:
            if isinstance(item, tuple) and len(item) >= 2:
                n, pred, common = item
            else:
                n, pred = item, None

            if isinstance(pred, Predicate):
                relation = Relationship(
                    direction=pred.direction or "out", predicate=pred
                )
            else:
                relation = Relationship(
                    direction="out",
                    predicate=Predicate(name="RELATED_TO", description=""),
                )

            if not n.uuid == node.uuid:
                normalized.append(
                    RetrievedNeighborNode(
                        neighbor=Node(**n.model_dump()),
                        relationship=Predicate(
                            name=relation.predicate.name,
                            description=relation.predicate.description,
                            direction=relation.predicate.direction,
                            observations=[],
                            level="1",
                        ),
                        most_common=Node(**common.model_dump()),
                    )
                )

        # ================= GETTING THE SIMILAR NEIGHBORS =================
        # If not enough direct 2nd level connected neighbors are found,
        # Search for nodes that are similar to the 1st level connected nodes of the
        # main node and than get the connected nodes that share
        # at least one label with the main node
        if len(normalized) < limit:
            neighbor_nodes = graph_adapter.get_nodes_by_uuid(
                uuids=[node.uuid], with_relationships=True
            )

            # Mapping the ids and the nodes into a dictionary and ids array
            related_nodes = {}
            related_node_ids = []
            for n_dict in neighbor_nodes:
                k = n_dict.get("related_nodes").uuid
                rel_node = n_dict.get("related_nodes")
                related_nodes[k] = rel_node
                related_node_ids.append(k)

            # Searching for similar nodes of the 1st level connected nodes
            # Returns a dictionary where the keys are the searched ids
            # And the values are the similar nodes found, respectively
            v_neighbors = vector_store_adapter.search_similar_by_ids(
                vector_ids=related_node_ids,
                store="nodes",
                min_similarity=0.2,
                limit=(limit - len(normalized)) * 10,
            )
            node_labels_lower = [n.lower() for n in node.labels]
            same_main_labels_v_neighbors = {}
            others_v_neighbors = {}

            for k, vectors in v_neighbors.items():
                # Filtering the vectors that are of the same labels as the main node
                # Directly similar nodes to the directly 1st level connected nodes
                matching_vectors = [
                    vector
                    for vector in vectors
                    if any(
                        label.lower() in node_labels_lower
                        for label in vector.metadata.get("labels", [])
                    )
                ]
                log("[found same label vectors]", len(matching_vectors))

                # Filtering the vectors that are NOT of the same labels as the main node
                # Similar nodes to the directly 1st level, a step to get the connected nodes
                # of the same label as the main node is needed
                non_matching_vectors = [
                    vector
                    for vector in vectors
                    if not any(
                        label.lower() in node_labels_lower
                        for label in vector.metadata.get("labels", [])
                    )
                ]
                log("[found non same label vectors]", len(non_matching_vectors))

                if matching_vectors:
                    same_main_labels_v_neighbors[k] = matching_vectors
                if non_matching_vectors:
                    others_v_neighbors[k] = non_matching_vectors

            # Getting and adding directly the similar nodes of same label as the main node
            for k, v in same_main_labels_v_neighbors.items():
                neighbor_nodes = graph_adapter.get_by_uuids(
                    uuids=[vector.metadata.get("uuid") for vector in v],
                )
                for n in neighbor_nodes:
                    if len(normalized) >= limit:
                        break
                    if not any(
                        [
                            norm.neighbor.uuid == n.uuid
                            or norm.most_common.uuid == related_nodes[k].uuid
                            for norm in normalized
                        ]
                    ):
                        normalized.append(
                            RetrievedNeighborNode(
                                neighbor=Node(**n.model_dump()),
                                relationship=Predicate(
                                    name="<SIMILAR_TO>",
                                    description=f"<SYS_DESCRIPTION>The neighbor and the most common are similar. Most common is directly connected to {node.name}.</SYS_DESCRIPTION>",
                                    direction="neutral",
                                    level="2",
                                ),
                                most_common=related_nodes[k],
                                observations=[],
                            )
                        )

            # Getting the connected nodes of the similar nodes and filtering
            # the ones the are with same label as the main node
            for k, v in others_v_neighbors.items():
                neighbor_nodes = graph_adapter.get_connected_nodes(
                    uuids=[vector.metadata.get("uuid") for vector in v],
                    limit=limit - len(normalized),
                    with_labels=node.labels,
                )
                for n, pred, common in neighbor_nodes:
                    if len(normalized) >= limit:
                        break
                    if not any([norm.neighbor.uuid == n.uuid for norm in normalized]):
                        normalized.append(
                            RetrievedNeighborNode(
                                neighbor=Node(**n.model_dump()),
                                relationship=Predicate(
                                    name=pred.name,
                                    description=pred.description,
                                    direction=pred.direction,
                                    level="3",
                                ),
                                most_common=Node(**common.model_dump()),
                            )
                        )

        # if len(normalized) < limit:
        #     v_data = vector_store_adapter.search_similar_by_ids(
        #         vector_ids=[node.uuid],
        #         store="data",
        #         min_similarity=0.5,
        #         limit=limit - len(normalized),
        #     )
        #     print("[v_data]", v_data)  # === same as above ===
        #     # TODO: get v_neighbor nodes by ids and the k (the common)
        #     # TODO + push to normalized
        #     # TODO (directly the nodes with label same as main, the others search for neighbors with same label of main)

        # Returning all the nodes found respecting the limit
        return RetrieveNeighborsRequestResponse(
            count=len(normalized), neighbors=normalized
        )

    return await asyncio.to_thread(_get_neighbors)


async def retrieve_neighbors_ai_mode(
    identification_params: IdentificationParams,
    looking_for: Optional[list[str]],
    limit: int,
) -> RetrieveNeighborsRequestResponse:
    """
    Retrieve neighbors of an entity from the knowledge graph.
    """

    def _get_neighbors():
        node = graph_adapter.get_by_identification_params(
            identification_params, entity_types=identification_params.entity_types
        )
        if not node:
            raise HTTPException(status_code=404, detail="Entity not found")

        result = kg_agent.retrieve_neighbors(node, looking_for, limit)

        ids = [neighbor.uuid for neighbor in result.neighbors]
        descriptions = [neighbor.description for neighbor in result.neighbors]

        nodes = graph_adapter.get_nodes_by_uuid(uuids=ids)
        paired = list(zip(nodes, descriptions))

        return RetrieveNeighborsRequestResponse(neighbors=paired)

    result = await asyncio.to_thread(_get_neighbors)

    return result

"""
File: /retrieve.py
Created Date: Sunday October 26th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday January 12th 2026 8:26:26 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import asyncio
from typing import Optional

from fastapi import HTTPException
from starlette.responses import JSONResponse

from src.constants.embeddings import Vector
from src.constants.kg import IdentificationParams, Node, Predicate
from src.core.search.entities import search_entities
from src.core.search.relationships import search_relationships
from src.services.api.constants.requests import (
    RetrieveRequestResponse,
    RetrieveNeighborsRequestResponse,
    RetrievedNeighborNode,
)
from src.services.kg_agent.main import graph_adapter, kg_agent
from src.services.data.main import data_adapter
from src.services.kg_agent.main import embeddings_adapter, vector_store_adapter
from src.utils.similarity.vectors import cosine_similarity


async def retrieve_data(
    text: str, limit: int, preferred_entities: str, brain_id: str = "default"
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
            text_embeddings.embeddings, "data", brain_id, limit
        )
        triple_vectors = vector_store_adapter.search_vectors(
            text_embeddings.embeddings, "triplets", brain_id, limit
        )

        search_result = data_adapter.search(text, brain_id)

        ts_text_chunks = search_result.text_chunks
        ts_observations = search_result.observations

        v_text_chunks, v_observations = data_adapter.get_text_chunks_by_ids(
            [dv.metadata.get("resource_id") for dv in data_vectors], True, brain_id
        )

        node_ids = [
            node_id
            for tv in triple_vectors
            for node_id in tv.metadata.get("node_ids", [])
        ]
        nodes = graph_adapter.get_nodes_by_uuid(
            uuids=node_ids,
            brain_id=brain_id,
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
    look_for: Optional[str] = None,
    identification_params: Optional[IdentificationParams] = None,
    limit: int = 10,
    brain_id: str = "default",
) -> RetrieveNeighborsRequestResponse:
    async def _get_neighbors():

        # ---------------------------------------------------------
        # ================= GETTING THE MAIN NODE =================
        # ---------------------------------------------------------
        def _get_node() -> Node:
            node = None
            if uuid:
                node = graph_adapter.get_by_uuid(uuid, brain_id)
            elif identification_params:
                node = graph_adapter.get_by_identification_params(
                    identification_params,
                    brain_id=brain_id,
                    entity_types=identification_params.entity_types,
                )
            if not node:
                raise HTTPException(status_code=404, detail="Entity not found")
            return node

        node = await asyncio.to_thread(_get_node)
        target_node_types = node.labels

        looking_for_v = embeddings_adapter.embed_text(look_for) if look_for else None

        # ---------------------------------------------------------
        # ===== Getting 1st degree neighbors of the main node =====
        # ---------------------------------------------------------
        def _get_fd_neighbors() -> (
            tuple[dict[str, list[tuple[Predicate, Node]]], list[str]]
        ):
            fd_neighbors = graph_adapter.get_neighbors(
                [node.uuid], limit=limit, brain_id=brain_id
            )
            fd_v_neighbors_ids = [
                fd[1].properties["v_id"] for fd in fd_neighbors[node.uuid]
            ]
            if look_for:
                fd_v_neighbors_embeddings = vector_store_adapter.get_by_ids(
                    fd_v_neighbors_ids, brain_id=brain_id, store="nodes"
                )
                fd_v_neighbors_embeddings_map = {
                    v.id: v.embeddings
                    for v in fd_v_neighbors_embeddings
                    if (
                        cosine_similarity(looking_for_v.embeddings, v.embeddings) > 0.5
                        and v.id
                        and not v.id.replace(
                            "-", ""
                        ).isalpha()  # likely not a UUID if all numeric (may have hyphens for uuid standard)
                        and v.id.isdigit()
                    )
                }
                fd_v_neighbors_ids = list(fd_v_neighbors_embeddings_map.keys())

            return fd_neighbors, fd_v_neighbors_ids

        fd_neighbors, fd_v_neighbors_ids = await asyncio.to_thread(_get_fd_neighbors)

        # ---------------------------------------------------------
        # === Getting nodes similar to the 1st degree neighbors ===
        # ---------------------------------------------------------
        fd_v_similar_node_futures = []
        for fd_v_neighbor_id in fd_v_neighbors_ids:
            fd_v_similar_node_futures.append(
                asyncio.to_thread(
                    vector_store_adapter.search_similar_by_ids,
                    [fd_v_neighbor_id],
                    brain_id,
                    "nodes",
                    0.5,
                    limit,
                )
            )
        fd_v_similar_nodes_results: list[dict[str, list[Vector]]] = (
            await asyncio.gather(*fd_v_similar_node_futures)
        )
        fd_similar_node_ids = [
            v.metadata["uuid"]
            for result_dict in fd_v_similar_nodes_results
            for vectors in result_dict.values()
            for v in vectors
        ]
        fd_similar_nodes = await asyncio.to_thread(
            graph_adapter.get_by_uuids, fd_similar_node_ids, brain_id
        )
        fd_similar_nodes_by_uuid = {n.uuid: n for n in fd_similar_nodes}

        # ---------------------------------------------------------
        # === Getting neighbors of the 1st degree similar nodes ===
        # ---------------------------------------------------------
        def _get_fd_similar_node_neighbors() -> dict[str, list[tuple[Predicate, Node]]]:
            fd_similar_node_neighbors = graph_adapter.get_neighbors(
                fd_similar_node_ids,
                limit=limit,
                brain_id=brain_id,
                of_types=list(set(target_node_types)),
            )
            return fd_similar_node_neighbors

        fd_similar_node_neighbors = await asyncio.to_thread(
            _get_fd_similar_node_neighbors
        )

        seen_neighbor_uuids = set()
        unique_neighbors = []
        for source_uuid, neighbors_list in fd_similar_node_neighbors.items():
            for neighbor_tuple in neighbors_list:
                neighbor_uuid = neighbor_tuple[1].uuid
                if neighbor_uuid not in seen_neighbor_uuids:
                    seen_neighbor_uuids.add(neighbor_uuid)
                    unique_neighbors.append(
                        RetrievedNeighborNode(
                            neighbor=neighbor_tuple[1],
                            relationship=neighbor_tuple[0],
                            most_common=fd_similar_nodes_by_uuid.get(source_uuid),
                        )
                    )

        return RetrieveNeighborsRequestResponse(
            count=len(unique_neighbors),
            main_node=node,
            neighbors=unique_neighbors[:limit],
        )

    return await _get_neighbors()


async def retrieve_neighbors_ai_mode(
    identification_params: IdentificationParams,
    looking_for: Optional[list[str]],
    limit: int,
    brain_id: str = "default",
) -> RetrieveNeighborsRequestResponse:
    """
    Retrieve neighbors of an entity from the knowledge graph.
    """

    def _get_neighbors():
        node = graph_adapter.get_by_identification_params(
            identification_params,
            brain_id=brain_id,
            entity_types=identification_params.entity_types,
        )
        if not node:
            raise HTTPException(status_code=404, detail="Entity not found")

        result = kg_agent.retrieve_neighbors(node, looking_for, limit)

        ids = [neighbor.uuid for neighbor in result.neighbors]
        descriptions = [neighbor.description for neighbor in result.neighbors]

        nodes = graph_adapter.get_nodes_by_uuid(uuids=ids, brain_id=brain_id)
        paired = list(zip(nodes, descriptions))

        return RetrieveNeighborsRequestResponse(neighbors=paired)

    result = await asyncio.to_thread(_get_neighbors)

    return result


async def get_relationships(
    limit: int = 10,
    skip: int = 0,
    relationship_types: Optional[list[str]] = None,
    from_node_labels: Optional[list[str]] = None,
    to_node_labels: Optional[list[str]] = None,
    query_text: Optional[str] = None,
    query_search_target: Optional[str] = "all",
    brain_id: str = "default",
):
    """
    Retrieve relationships from the knowledge graph with optional filtering and pagination.

    Parameters:
        relationship_types (list[str], optional): Filter results to specific relationship types.
        from_node_labels (list[str], optional): Filter relationships originating from nodes with these labels.
        to_node_labels (list[str], optional): Filter relationships targeting nodes with these labels.
        query_text (str, optional): Text to search within relationship or node content.
        query_search_target (str, optional): Field to target for text search; commonly "all", "from", or "to".
        limit (int, optional): Maximum number of relationships to return.
        skip (int, optional): Number of relationships to skip (offset).
        brain_id (str, optional): Identifier of the brain/graph to query.

    Returns:
        JSONResponse: A response whose JSON content contains:
            - message: Confirmation string.
            - relationships: List of serialized relationship objects.
            - total: Total number of matching relationships.
    """
    relationships = await asyncio.to_thread(
        search_relationships,
        limit,
        skip,
        relationship_types,
        from_node_labels,
        to_node_labels,
        query_text,
        query_search_target,
        brain_id,
    )

    return JSONResponse(
        content={
            "message": "Relationships retrieved successfully",
            "relationships": [r.model_dump(mode="json") for r in relationships.results],
            "total": relationships.total,
        }
    )


async def get_entities(
    limit: int = 10,
    skip: int = 0,
    node_labels: Optional[list[str]] = None,
    query_text: Optional[str] = None,
    brain_id: str = "default",
):
    """
    Retrieve entities from the knowledge graph with optional label and text filters.

    Parameters:
        limit (int): Maximum number of entities to return (pagination).
        skip (int): Number of entities to skip (pagination offset).
        node_labels (Optional[list[str]]): If provided, only return entities whose labels match any value in this list.
        query_text (Optional[str]): If provided, filter entities by matching text content.
        brain_id (str): Identifier of the knowledge graph/brain to query.

    Returns:
        JSONResponse: Object containing:
            - message (str): Informational message.
            - entities (list): Serialized entity objects.
            - total (int): Total number of matching entities.
    """
    entities = await asyncio.to_thread(
        search_entities, limit, skip, node_labels, query_text, brain_id
    )

    return JSONResponse(
        content={
            "message": "Entities retrieved successfully",
            "entities": [e.model_dump(mode="json") for e in entities.results],
            "total": entities.total,
        }
    )

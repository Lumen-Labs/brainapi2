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
import json

from fastapi import HTTPException

from src.constants.kg import Node, Predicate, Relationship
from src.services.api.constants.requests import (
    RetrieveRequestResponse,
    RetrieveNeighborsRequestResponse,
    RetrievedNeighborNode,
)
from src.services.kg_agent.main import graph_adapter
from src.services.data.main import data_adapter
from src.services.kg_agent.main import embeddings_adapter, vector_store_adapter


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
        print("[preferred entities]", preferred_entities, type(preferred_entities))
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


async def retrieve_neighbors(uuid: str, limit: int) -> RetrieveNeighborsRequestResponse:
    """
    Retrieve neighbors of an entity from the knowledge graph.
    """
    node = graph_adapter.get_by_uuid(uuid)
    if not node:
        raise HTTPException(status_code=404, detail="Entity not found")
    results = graph_adapter.get_nodes_by_uuid(
        uuids=[node.uuid],
        with_relationships=True,
        relationships_depth=5,
        # preferred_labels=[node.label],
    )

    neighbors = []

    for result in results:
        related_node = result.get("related_nodes")
        relationships = result.get("relationships")

        if not related_node or not relationships:
            continue

        relationship = relationships[0] if relationships else None

        neighbor_node = RetrievedNeighborNode(
            uuid=related_node.get("uuid"),
            name=related_node.get("name"),
            labels=related_node.get("labels") if related_node.get("labels") else [],
            description=related_node.get("description"),
            properties=(
                related_node.get("properties") if related_node.get("properties") else {}
            ),
            observations=[],
            relation=Relationship(
                predicate=Predicate(
                    name=(
                        relationship.type
                        if hasattr(relationship, "type")
                        else str(relationship)
                    ),
                    description="",
                ),
                direction="in",
            ),
        )
        neighbors.append(neighbor_node)

    return RetrieveNeighborsRequestResponse(neighbors=neighbors)

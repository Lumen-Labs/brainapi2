"""
File: /retrieve.py
Created Date: Saturday October 25th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday October 25th 2025 12:32:21 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import asyncio
from typing import List
from fastapi import APIRouter, Query

from src.services.api.constants.requests import RetrieveRequestResponse
from src.services.data.main import data_adapter
from src.services.kg_agent.main import embeddings_adapter, vector_store_adapter
from src.services.kg_agent.main import graph_adapter

retrieve_router = APIRouter(prefix="/retrieve", tags=["retrieve"])


@retrieve_router.get("/", response_model=RetrieveRequestResponse)
async def retrieve(
    text: str = Query(..., description="The text to search for."),
    limit: int = Query(10, description="The number of results to return."),
    preferred_entities: List[str] = Query(
        ..., description="The entities to prioritize in the relationships."
    ),
):
    """
    Retrieve data from the knowledge graph and data store.
    """

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

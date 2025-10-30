"""
File: /retrieve.py
Created Date: Saturday October 25th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday October 25th 2025 12:32:21 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from typing import Optional
from fastapi import APIRouter, Body, Query

from src.services.api.constants.requests import (
    RetrieveNeighborsAiModeRequestBody,
    RetrieveNeighborsWithIdentificationParamsRequestBody,
    RetrieveRequestResponse,
    RetrieveNeighborsRequestResponse,
)
from src.services.api.controllers.retrieve import (
    retrieve_neighbors as retrieve_neighbors_controller,
    retrieve_neighbors_ai_mode as retrieve_neighbors_ai_mode_controller,
)
from src.services.api.controllers.retrieve import (
    retrieve_data as retrieve_data_controller,
)

retrieve_router = APIRouter(prefix="/retrieve", tags=["retrieve"])


@retrieve_router.get("/", response_model=RetrieveRequestResponse)
async def retrieve(
    text: str = Query(..., description="The text to search for."),
    limit: int = Query(10, description="The number of results to return."),
    preferred_entities: str = Query(
        ...,
        description="The entities to prioritize in the relationships, separated by commas.",
    ),
):
    """
    Retrieve data from the knowledge graph and data store.
    """
    return await retrieve_data_controller(text, limit, preferred_entities)


@retrieve_router.get(
    "/entities/neighbors", response_model=RetrieveNeighborsRequestResponse
)
async def get_neighbors(
    uuid: str = Query(..., description="The UUID of the entity to get neighbors for."),
    limit: int = Query(10, description="The number of neighbors to return."),
):
    """
    Get the neighbors of an entity.
    """
    return await retrieve_neighbors_controller(uuid=uuid, limit=limit)


@retrieve_router.post(
    "/entities/neighbors", response_model=RetrieveNeighborsRequestResponse
)
async def get_neighbors_with_identification_params(
    request: RetrieveNeighborsWithIdentificationParamsRequestBody,
):
    """
    Get the neighbors of an entity.
    """
    return await retrieve_neighbors_controller(
        identification_params=request.identification_params, limit=request.limit
    )


@retrieve_router.post("/entities/neighbors/ai-mode")
async def get_neighbors_ai_mode(
    request: RetrieveNeighborsAiModeRequestBody = Body(
        ...,
        description="The request body for the retrieve neighbors AI mode endpoint.",
    ),
):
    """
    Get the neighbors of an entity in AI mode.
    """
    return await retrieve_neighbors_ai_mode_controller(
        request.identification_params, request.looking_for, request.limit
    )


@retrieve_router.post("/context")
async def get_context(request):
    """
    Get the context of an entity.
    """

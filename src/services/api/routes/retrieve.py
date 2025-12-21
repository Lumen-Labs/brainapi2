"""
File: /retrieve.py
Created Date: Saturday October 25th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 13th 2025
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
    get_relationships as retrieve_get_relationships_controller,
    get_entities as retrieve_get_entities_controller,
)
from src.services.api.controllers.retrieve import (
    retrieve_data as retrieve_data_controller,
)
from src.services.api.controllers.structured_data import (
    get_structured_data_by_id as get_structured_data_by_id_controller,
    get_structured_data_list as get_structured_data_list_controller,
    get_structured_data_types as get_structured_data_types_controller,
)
from src.services.api.controllers.observations import (
    get_observation_by_id as get_observation_by_id_controller,
    get_observations_list as get_observations_list_controller,
    get_observation_labels as get_observation_labels_controller,
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
    brain_id: str = "default",
):
    """
    Retrieve data from the knowledge graph and data store.
    """
    return await retrieve_data_controller(text, limit, preferred_entities, brain_id)


@retrieve_router.get(
    "/entities/neighbors", response_model=RetrieveNeighborsRequestResponse
)
async def get_neighbors(
    uuid: str = Query(..., description="The UUID of the entity to get neighbors for."),
    limit: int = Query(10, description="The number of neighbors to return."),
    brain_id: str = "default",
):
    """
    Get the neighbors of an entity.
    """
    return await retrieve_neighbors_controller(
        uuid=uuid, limit=limit, brain_id=brain_id
    )


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
        identification_params=request.identification_params,
        limit=request.limit,
        brain_id=request.brain_id,
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


@retrieve_router.get(path="/relationships")
async def get_relationships(
    limit: int = 10,
    skip: int = 0,
    relationship_types: Optional[str] = None,
    from_node_labels: Optional[str] = None,
    to_node_labels: Optional[str] = None,
    query_text: Optional[str] = None,
    query_search_target: Optional[str] = "all",
    brain_id: str = "default",
):
    """
    Get the relationships of the graph.
    """
    if relationship_types:
        relationship_types = relationship_types.split(",")
    if from_node_labels:
        from_node_labels = from_node_labels.split(",")
    if to_node_labels:
        to_node_labels = to_node_labels.split(",")
    return await retrieve_get_relationships_controller(
        limit,
        skip,
        relationship_types,
        from_node_labels,
        to_node_labels,
        query_text,
        query_search_target,
    )


@retrieve_router.get(path="/entities")
async def get_entities(
    limit: int = 10,
    skip: int = 0,
    node_labels: Optional[str] = None,
    query_text: Optional[str] = None,
    brain_id: str = "default",
):
    """
    Get the entities of the graph.
    """
    if node_labels:
        node_labels = node_labels.split(",")
    return await retrieve_get_entities_controller(
        limit, skip, node_labels, query_text, brain_id
    )

@retrieve_router.get(path="/structured-data/types")
async def get_structured_data_types():
    """
    Get all unique types from structured data.
    """
    return await get_structured_data_types_controller()

@retrieve_router.get(path="/structured-data/{id}")
async def get_structured_data_by_id(
    id: str,
):
    """
    Get structured data by ID.
    """
    return await get_structured_data_by_id_controller(id)

@retrieve_router.get(path="/structured-data")
async def get_structured_data_list(
    limit: int = 10,
    skip: int = 0,
    types: Optional[str] = None,
    query_text: Optional[str] = None,
):
    """
    Get a list of structured data.
    """
    if types:
        types = types.split(",")
    return await get_structured_data_list_controller(limit, skip, types, query_text)

@retrieve_router.get(path="/observations/labels")
async def get_observation_labels():
    """
    Get all unique labels from observations.
    """
    return await get_observation_labels_controller()

@retrieve_router.get(path="/observations/{id}")
async def get_observation_by_id(
    id: str,
):
    """
    Get observation by ID.
    """
    return await get_observation_by_id_controller(id)


@retrieve_router.get(path="/observations")
async def get_observations_list(
    limit: int = 10,
    skip: int = 0,
    resource_id: Optional[str] = None,
    labels: Optional[str] = None,
    query_text: Optional[str] = None,
):
    """
    Get a list of observations.
    """
    if labels:
        labels = labels.split(",")
    return await get_observations_list_controller(limit, skip, resource_id, labels, query_text)
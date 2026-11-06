"""
File: /retrieve.py
Created Date: Saturday October 25th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Thursday February 19th 2026 7:45:12 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import json
from typing import List, Literal, Optional
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from src.services.api.dependencies import get_brain_id
from src.services.api.constants.requests import (
    GetEntityContextResponse,
    GetEntityInfoResponse,
    GetEntitySibilingsResponse,
    GetEntityStatusResponse,
    GetContextRequestBody,
    GetContextResponse,
    RecommendResponse,
    RecommendRequestBody,
    RetrieveNeighborsAiModeRequestBody,
    RetrieveNeighborsWithIdentificationParamsRequestBody,
    RetrieveRequestResponse,
    RetrieveNeighborsRequestResponse,
    SearchRequestBody,
    SearchResponse,
)
from src.services.api.constants.responses import (
    ChangelogItemResponse,
    ChangelogListResponse,
    EntityListResponse,
    HopListResponse,
    LabelListResponse,
    ObservationItemResponse,
    ObservationListResponse,
    RelationshipListResponse,
    StructuredDataItemResponse,
    StructuredDataListResponse,
    TextChunkListResponse,
    TypeListResponse,
    VectorListResponse,
    VectorStoreListResponse,
)
from src.services.api.controllers.retrieve import (
    retrieve_neighbors as retrieve_neighbors_controller,
    retrieve_neighbors_ai_mode as retrieve_neighbors_ai_mode_controller,
    get_relationships as retrieve_get_relationships_controller,
    get_entities as retrieve_get_entities_controller,
    get_context as retrieve_get_context_controller,
)
from src.services.api.controllers.kg import get_hops as get_hops_controller
from src.services.api.controllers.retrieve import (
    retrieve_data as retrieve_data_controller,
)
from src.services.api.controllers.search import search as search_controller
from src.services.api.controllers.structured_data import (
    get_structured_data_by_id as get_structured_data_by_id_controller,
    get_structured_data_list as get_structured_data_list_controller,
    get_structured_data_types as get_structured_data_types_controller,
    get_text_chunks as get_text_chunks_controller,
)
from src.services.api.controllers.observations import (
    get_observation_by_id as get_observation_by_id_controller,
    get_observations_list as get_observations_list_controller,
    get_observation_labels as get_observation_labels_controller,
)
from src.services.api.controllers.changelogs import (
    get_changelog_by_id as get_changelog_by_id_controller,
    get_changelogs_list as get_changelogs_list_controller,
    get_changelog_types as get_changelog_types_controller,
)
from src.services.api.controllers.entities import (
    get_entity_info as get_entity_info_controller,
    get_entity_context as get_entity_context_controller,
    get_entity_sibilings as get_entity_sibilings_controller,
    get_entity_status as get_entity_status_controller,
    get_recommendations as get_recommendations_controller,
)
from src.services.api.controllers.vectors import (
    get_vector_stores as get_vector_stores_controller,
    list_vectors as list_vectors_controller,
)

from src.services.api.routes.explore import explore_router


retrieve_router = APIRouter(prefix="/retrieve", tags=["retrieve"])
retrieve_router.include_router(explore_router)


@retrieve_router.get("/", response_model=RetrieveRequestResponse)
async def retrieve(
    text: str = Query(..., description="The text to search for."),
    limit: int = Query(10, description="The number of results to return."),
    preferred_entities: str = Query(
        "",
        description="The entities to prioritize in the relationships, separated by commas.",
    ),
    brain_id: str = Depends(get_brain_id),
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
    look_for: Optional[str] = Query(
        None,
        description="The description of what in the neighbors should share with the target.",
    ),
    brain_id: str = Depends(get_brain_id),
):
    """
    Get the neighbors of an entity.
    """
    return await retrieve_neighbors_controller(
        uuid=uuid, limit=limit, look_for=look_for, brain_id=brain_id
    )


@retrieve_router.post(
    "/entities/neighbors", response_model=RetrieveNeighborsRequestResponse
)
async def get_neighbors_with_identification_params(
    request: RetrieveNeighborsWithIdentificationParamsRequestBody,
    brain_id: str = Depends(get_brain_id),
):
    """
    Get the neighbors of an entity.
    """
    request.brain_id = brain_id
    return await retrieve_neighbors_controller(
        identification_params=request.identification_params,
        limit=request.limit,
        look_for=request.look_for if request.look_for else None,
        brain_id=request.brain_id,
    )


@retrieve_router.post(
    "/entities/neighbors/ai-mode", response_model=RetrieveNeighborsRequestResponse
)
async def get_neighbors_ai_mode(
    request: RetrieveNeighborsAiModeRequestBody = Body(
        ...,
        description="The request body for the retrieve neighbors AI mode endpoint.",
    ),
):
    """
    Retrieve neighboring entities using AI-mode identification parameters.

    Parameters:
        request (RetrieveNeighborsAiModeRequestBody): Request body containing:
            - identification_params: parameters that identify the target entity or entities.
            - looking_for: description of the desired neighbors or relation types.
            - limit: maximum number of neighbor results to return.

    Returns:
        list: Neighboring entities that match the AI-mode identification and search criteria.
    """
    return await retrieve_neighbors_ai_mode_controller(
        request.identification_params, request.looking_for, request.limit
    )


@retrieve_router.post("/context", response_model=GetContextResponse)
async def get_context(
    request: GetContextRequestBody,
    brain_id: str = Depends(get_brain_id),
) -> GetContextResponse:
    """
    Handle an HTTP request to retrieve an entity's contextual information.

    Parameters:
        request (GetContextRequestBody): Request body containing:
            - text: The text to search context for.
            - brain_id: The brain/workspace identifier to query.

    Returns:
        GetContextResponse: Response containing the context information.
    """
    request.brain_id = brain_id
    return await retrieve_get_context_controller(request)


@retrieve_router.get(path="/relationships", response_model=RelationshipListResponse)
async def get_relationships(
    limit: int = 10,
    skip: int = 0,
    relationship_types: Optional[str] = None,
    from_node_labels: Optional[str] = None,
    to_node_labels: Optional[str] = None,
    query_text: Optional[str] = None,
    query_search_target: Optional[str] = "all",
    brain_id: str = Depends(get_brain_id),
):
    """
    Retrieve relationships filtered by types, node labels, and query criteria.

    Parameters:
        relationship_types (Optional[str]): Comma-separated relationship types to include.
        from_node_labels (Optional[str]): Comma-separated source node labels to filter.
        to_node_labels (Optional[str]): Comma-separated target node labels to filter.
        query_text (Optional[str]): Text to match against relationships or nodes.
        query_search_target (Optional[str]): Target of the text query, such as "all", "source", or "target".
        limit (int): Maximum number of relationships to return.
        skip (int): Number of relationships to skip.
        brain_id (str): Brain (dataset) identifier.

    Returns:
        List of relationship records matching the filters.
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
        brain_id,
    )


@retrieve_router.get(path="/entities", response_model=EntityListResponse)
async def get_entities(
    limit: int = 10,
    skip: int = 0,
    node_labels: Optional[str] = None,
    query_text: Optional[str] = None,
    brain_id: str = Depends(get_brain_id),
):
    """
    Retrieve entities matching optional label and text filters with pagination.

    Parameters:
        node_labels (Optional[str]): Comma-separated node labels to filter by (e.g. "Person,Company"); when provided, only entities with any of these labels are returned.
        query_text (Optional[str]): Free-text filter to match entity properties or content.
        brain_id (str): Identifier of the brain/knowledge store to query.

    Returns:
        list: A list of entity records that match the provided filters and pagination parameters.
    """
    if node_labels:
        node_labels = node_labels.split(",")
    return await retrieve_get_entities_controller(
        limit, skip, node_labels, query_text, brain_id
    )


@retrieve_router.get(path="/structured-data/types", response_model=TypeListResponse)
async def get_structured_data_types(
    brain_id: str = Depends(get_brain_id),
):
    """
    Get all unique types from structured data.
    """
    return await get_structured_data_types_controller(brain_id)


@retrieve_router.get(
    path="/structured-data/{id}", response_model=StructuredDataItemResponse
)
async def get_structured_data_by_id(
    id: str,
    brain_id: str = Depends(get_brain_id),
):
    """
    Get structured data by ID.
    """
    return await get_structured_data_by_id_controller(id, brain_id)


@retrieve_router.get(path="/structured-data", response_model=StructuredDataListResponse)
async def get_structured_data_list(
    limit: int = 10,
    skip: int = 0,
    types: Optional[str] = None,
    query_text: Optional[str] = None,
    brain_id: str = Depends(get_brain_id),
):
    """
    Get a list of structured data.
    """
    if types:
        types = types.split(",")
    return await get_structured_data_list_controller(
        limit, skip, types, query_text, brain_id
    )


@retrieve_router.get(path="/text-chunks", response_model=TextChunkListResponse)
async def get_text_chunks(
    brain_id: str = Depends(get_brain_id),
    limit: int = 10,
    skip: int = 0,
    query_text: Optional[str] = None,
    metadata_eq: Optional[str] = Query(default=None),
    order: Literal["asc", "desc"] = Query(default="desc"),
):
    """
    Get text chunks by a query text and optional metadata equality filters.
    """
    metadata_eq_filters: Optional[dict[str, str]] = None
    if metadata_eq:
        try:
            parsed_metadata_eq = json.loads(metadata_eq)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail="metadata_eq must be a valid JSON object string",
            ) from exc

        if not isinstance(parsed_metadata_eq, dict):
            raise HTTPException(
                status_code=422,
                detail="metadata_eq must be a JSON object string",
            )

        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in parsed_metadata_eq.items()
        ):
            raise HTTPException(
                status_code=422,
                detail="metadata_eq must be a JSON object with string keys and string values",
            )

        metadata_eq_filters = parsed_metadata_eq

    results, total = await get_text_chunks_controller(
        brain_id, limit, skip, query_text, metadata_eq_filters, order
    )

    return JSONResponse(
        content={
            "message": "Text chunks retrieved successfully",
            "data": [r.model_dump(mode="json") for r in results],
            "total": total,
        }
    )


@retrieve_router.get(path="/observations/labels", response_model=LabelListResponse)
async def get_observation_labels(
    brain_id: str = Depends(get_brain_id),
):
    """
    Get all unique labels from observations.
    """
    return await get_observation_labels_controller(brain_id)


@retrieve_router.get(
    path="/observations/{id}", response_model=ObservationItemResponse
)
async def get_observation_by_id(
    id: str,
    brain_id: str = Depends(get_brain_id),
):
    """
    Get observation by ID.
    """
    return await get_observation_by_id_controller(id, brain_id)


@retrieve_router.get(path="/observations", response_model=ObservationListResponse)
async def get_observations_list(
    limit: int = 10,
    skip: int = 0,
    resource_id: Optional[str] = None,
    labels: Optional[str] = None,
    query_text: Optional[str] = None,
    brain_id: str = Depends(get_brain_id),
):
    """
    Retrieve a list of observations filtered by the provided parameters.

    Parameters:
        limit (int): Maximum number of observations to return.
        skip (int): Number of observations to skip (offset).
        resource_id (Optional[str]): If provided, only return observations for this resource identifier.
        labels (Optional[str]): Comma-separated observation labels to filter by; when provided, labels are treated as a list of strings.
        query_text (Optional[str]): Full-text query to filter observations.
        brain_id (str): Identifier of the brain/tenant to query.

    Returns:
        list: A list of observation records that match the provided filters.
    """
    if labels:
        labels = labels.split(",")

    return await get_observations_list_controller(
        limit, skip, resource_id, labels, query_text, brain_id
    )


@retrieve_router.get(path="/changelogs/types", response_model=TypeListResponse)
async def get_changelog_types(
    brain_id: str = Depends(get_brain_id),
):
    """
    Retrieve all unique changelog types for the specified brain.

    Parameters:
        brain_id (str): Identifier of the brain to query; defaults to "default".

    Returns:
        list[str]: List of unique changelog type names.
    """
    return await get_changelog_types_controller(brain_id)


@retrieve_router.get(path="/changelogs/{id}", response_model=ChangelogItemResponse)
async def get_changelog_by_id(
    id: str,
    brain_id: str = Depends(get_brain_id),
):
    """
    Retrieve a changelog entry by its unique identifier.

    Parameters:
        id (str): Unique identifier of the changelog entry.
        brain_id (str): Brain identifier to query. Defaults to "default".

    Returns:
        The changelog entry corresponding to the specified identifier.
    """
    return await get_changelog_by_id_controller(id, brain_id)


@retrieve_router.get(path="/changelogs", response_model=ChangelogListResponse)
async def get_changelogs_list(
    limit: int = 10,
    skip: int = 0,
    types: Optional[str] = None,
    query_text: Optional[str] = None,
    brain_id: str = Depends(get_brain_id),
):
    """
    Retrieve a paginated list of changelogs.

    Parameters:
        types (Optional[str]): Optional comma-separated changelog types to filter by; each type will be applied as a filter.
        query_text (Optional[str]): Optional text used to filter changelogs by content or metadata.

    Returns:
        list: Changelog records matching the provided filters, constrained by `limit` and `skip`.
    """
    if types:
        types = types.split(",")
    return await get_changelogs_list_controller(
        limit, skip, types, query_text, brain_id
    )


@retrieve_router.get(path="/hops", response_model=HopListResponse)
async def get_hops(
    query: str,
    degrees: Literal[2] = 2,
    flattened: bool = True,
    brain_id: str = Depends(get_brain_id),
):  # noqa: F821
    """
    Compute graph hops for the given query.

    Parameters:
        query (str): Search text or entity identifier to start hop traversal from.
        degrees (int): Maximum number of hop degrees to traverse (default 2).
        flattened (bool): If True, return a flattened list of hops; otherwise preserve nested hop structure.
        brain_id (str): Identifier of the brain (knowledge graph) to query.

    Returns:
        list: Hops (paths) connecting matching entities up to the specified degree; the exact structure varies based on `flattened`.
    """
    return await get_hops_controller(query, degrees, flattened, brain_id)


@retrieve_router.get(path="/entity/info", response_model=GetEntityInfoResponse)
async def get_entity_info(
    target: str,
    query: str,
    max_depth: int = 3,
    brain_id: str = Depends(get_brain_id),
):
    """
    Retrieve detailed information for an entity identified by the given target and query.

    Parameters:
        target (str): Identifier or name of the target entity to retrieve.
        query (str): Query text used to refine or disambiguate the requested entity information.
        max_depth (int): Maximum graph depth to traverse when collecting related information.
        brain_id (str): Identifier of the brain/namespace to query.

    Returns:
        dict: A mapping containing the entity's attributes and related contextual information.
    """
    return await get_entity_info_controller(target, query, max_depth, brain_id)


@retrieve_router.get(path="/entity/context", response_model=GetEntityContextResponse)
async def get_entity_context(
    target: str,
    context_depth: int = 3,
    brain_id: str = Depends(get_brain_id),
):
    """
    Retrieve contextual information for an entity identified by `target`.

    Parameters:
        target (str): Identifier or name of the target entity.
        context_depth (int): Maximum depth of related context to include.
        brain_id (str): Identifier of the brain/knowledge graph to query.

    Returns:
        dict: A mapping containing the entity's contextual information.
    """
    return await get_entity_context_controller(target, context_depth, brain_id)


@retrieve_router.get(path="/entity/synergies", response_model=GetEntitySibilingsResponse)
async def get_entity_synergies(
    target: str,
    polarity: Literal["same", "opposite"] = "same",
    do: bool = False,
    pa: bool = False,
    ppa: bool = False,
    top_k: int = 50,
    labels: Optional[List[str]] = None,
    brain_id: str = Depends(get_brain_id),
):
    """
    Retrieve synergies for a specified entity.

    Parameters:
        target: Identifier or name of the target entity.
        polarity: "same" for matching node polarity; "opposite" for positive↔negative.
        do: If True, only direct synergies are returned.
        pa: If True, potential anchors are returned.
        ppa: If True, seed anchors are returned.
        top_k: Maximum synergies to return (default 50).
        labels: Optional candidate label filter.
        brain_id: Identifier of the knowledge brain to query.

    Returns:
        Synergy records for the target entity matching the requested polarity.
    """
    return await get_entity_sibilings_controller(
        target, polarity, do, pa, ppa, top_k, labels, brain_id
    )


@retrieve_router.get(
    path="/recommend",
    response_model=RecommendResponse,
    summary="Recommend entities (Preview)",
    openapi_extra={"x-stability": "preview"},
)
async def get_recommend(
    target: str,
    polarity: Literal["same", "opposite"] = "same",
    top_k: int = 20,
    labels: Optional[List[str]] = None,
    include_asymmetric: bool = True,
    include_multi_interest: bool = True,
    include_attribute_pref: bool = False,
    diversify: bool = True,
    asymmetric_direction: Literal["outbound", "inbound", "both"] = "outbound",
    exclude_seen: bool = False,
    recency_half_life_days: Optional[float] = None,
    dampen_degree: bool = False,
    brain_id: str = Depends(get_brain_id),
):
    """
    Preview: ranked event-graph recommendations composing synergies, asymmetric
    complementary walks, multi-interest medoids, and optional attribute
    preferences (retrieval-time only; no training).
    """
    return await get_recommendations_controller(
        target,
        polarity,
        top_k,
        labels,
        include_asymmetric,
        include_multi_interest,
        diversify,
        asymmetric_direction,
        brain_id,
        exclude_seen,
        recency_half_life_days,
        dampen_degree,
        include_attribute_pref,
        None,
    )


@retrieve_router.post(
    path="/recommend",
    response_model=RecommendResponse,
    summary="Recommend entities from a request body (Preview)",
    openapi_extra={"x-stability": "preview"},
)
async def post_recommend(
    body: RecommendRequestBody,
    brain_id: str = Depends(get_brain_id),
):
    """Preview recommendation surface; request and response compatibility is kept."""
    return await get_recommendations_controller(
        body.target,
        body.polarity,
        body.top_k,
        body.labels,
        body.include_asymmetric,
        body.include_multi_interest,
        body.diversify,
        body.asymmetric_direction,
        brain_id or body.brain_id,
        body.exclude_seen,
        body.recency_half_life_days,
        body.dampen_degree,
        body.include_attribute_pref,
        body.behavior_weights,
    )


@retrieve_router.get(path="/search", response_model=SearchResponse)
async def get_search(
    query: str = Query(..., description="The search query."),
    k: int = Query(10, ge=1, le=200, description="Number of hits to return."),
    channels: Optional[str] = Query(
        "passages",
        description=(
            "Comma-separated channels: passages, entities, events, "
            "communities, and/or plugin:<name>."
        ),
    ),
    node_labels: Optional[str] = Query(
        None,
        description="Comma-separated node labels to filter the entities channel.",
    ),
    community_labels: Optional[str] = Query(
        None,
        description="Comma-separated hub labels for the communities channel.",
    ),
    expand: Literal["none", "neighbors"] = Query(
        "none",
        description="Optional 1-hop expansion from graph channel seeds.",
    ),
    fusion: Optional[Literal["rrf", "cc"]] = Query(
        None, description="Fusion override. Default is SEARCH_FUSION."
    ),
    rerank: Optional[str] = Query(
        None,
        description="none or plugin:<name>. Unknown plugin names return 400.",
    ),
    target: Optional[str] = Query(
        None,
        description="Optional USER uuid or id for query-gated rerank of retrieved hits.",
    ),
    profile_stages: bool = Query(False),
    brain_id: str = Depends(get_brain_id),
):
    channel_list = [
        item.strip()
        for item in (channels or "passages").split(",")
        if item.strip()
    ] or ["passages"]
    node_label_list = [
        item.strip()
        for item in (node_labels or "").split(",")
        if item.strip()
    ] or None
    community_label_list = [
        item.strip()
        for item in (community_labels or "").split(",")
        if item.strip()
    ] or None
    request = SearchRequestBody(
        query=query,
        k=k,
        channels=channel_list,
        node_labels=node_label_list,
        community_labels=community_label_list,
        expand=expand,
        fusion=fusion,
        rerank=rerank,
        target=target,
        profile_stages=profile_stages,
        brain_id=brain_id,
    )
    return await search_controller(request)


@retrieve_router.post(path="/search", response_model=SearchResponse)
async def post_search(
    body: SearchRequestBody,
    brain_id: str = Depends(get_brain_id),
):
    body.brain_id = brain_id or body.brain_id
    return await search_controller(body)


@retrieve_router.get(path="/entity/status", response_model=GetEntityStatusResponse)
async def get_entity_status(
    target: str,
    types: Optional[List[str]] = None,
    brain_id: str = Depends(get_brain_id),
):
    """
    Retrieve status information for a specified entity.

    Parameters:
        target (str): Identifier or name of the entity to inspect.
        types (Optional[List[str]]): Optional list of entity types to filter the status computation.
        brain_id (str): Identifier of the brain (knowledge graph) to query.

    Returns:
        status (dict): A dictionary containing the entity's status details.
    """
    return await get_entity_status_controller(target, types or [], brain_id)


@retrieve_router.get(path="/vectors/stores", response_model=VectorStoreListResponse)
async def get_vector_stores():
    """
    List available vector store names and dimensions.
    """
    return await get_vector_stores_controller()


@retrieve_router.get(path="/vectors/{store}", response_model=VectorListResponse)
async def get_vectors_list(
    store: str,
    limit: int = 10,
    skip: int = 0,
    include_embeddings: bool = False,
    brain_id: str = Depends(get_brain_id),
):
    """
    List vectors in a store with pagination.
    """
    return await list_vectors_controller(
        store, brain_id, limit, skip, include_embeddings
    )

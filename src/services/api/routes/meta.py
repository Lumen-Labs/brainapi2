from src.services.api.controllers.meta import get_ingest_group_meter as get_ingest_group_meter_controller
from src.services.api.constants.responses import IngestGroupMeterResponse, IngestionCapabilitiesResponse
"""
File: /meta.py
Created Date: Friday November 7th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 27th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from fastapi import APIRouter, Depends, Request

from src.services.api.dependencies import get_brain_id
from src.services.api.controllers.meta import (
    get_entities_labels as get_entities_labels_controller,
    get_login_info as get_login_info_controller,
    get_relationships_properties as get_relationships_properties_controller,
    get_entity_properties as get_entity_properties_controller,
)
from src.services.api.constants.responses import LoginInfoResponse, StringListResponse

meta_router = APIRouter(prefix="/meta", tags=["meta"])

@meta_router.get(path="/login-info", response_model=LoginInfoResponse)
async def get_login_info(request: Request):
    """
    Resolve whether a BrainPAT is the system token or scoped to a single brain.
    """
    return await get_login_info_controller(request)

@meta_router.get(path="/relationships-properties", response_model=StringListResponse)
async def get_relationships_properties(
    brain_id: str = Depends(get_brain_id),
):
    """
    Retrieve all unique relationship types present in the graph for the specified brain.
    
    Parameters:
        brain_id (str): Identifier of the brain/graph to query. Defaults to "default".
    
    Returns:
        A list of relationship type names (strings) found in the specified graph.
    """
    return await get_relationships_properties_controller(brain_id)

@meta_router.get(path="/entity-labels", response_model=StringListResponse)
async def get_entities_labels(
    brain_id: str = Depends(get_brain_id),
):
    """
    Retrieve all unique node labels present in the graph.
    
    Parameters:
        brain_id (str): Identifier of the brain/graph to query. Defaults to "default".
    
    Returns:
        labels (list[str]): List of unique node label names.
    """
    return await get_entities_labels_controller(brain_id)

@meta_router.get(path="/entity-properties", response_model=StringListResponse)
async def get_entity_properties(
    brain_id: str = Depends(get_brain_id),
):
    """
    Retrieve all unique property keys used by entities in the specified brain/graph.
    
    Parameters:
        brain_id (str): Identifier of the brain/graph to query; defaults to "default".
    
    Returns:
        list[str]: A list of property key names present on entities in the graph.
    """
    return await get_entity_properties_controller(brain_id)


@meta_router.get(
    path="/ingest-groups/{group_tracking}",
    response_model=IngestGroupMeterResponse,
)
async def get_ingest_group_meter(
    group_tracking: str,
    request: Request,
    brain_id: str = Depends(get_brain_id),
):
    """
    Return the durable ingest-cost rollup for a group_tracking key in this brain.
    """
    import os
    from fastapi import HTTPException
    from src.utils.tokens import pat_matches
    token = request.headers.get("BrainPAT") or request.headers.get("Authorization", "").removeprefix("Bearer ")
    if not pat_matches(token, os.getenv("BRAINPAT_TOKEN")):
        raise HTTPException(403, "Internal usage requires platform administration")
    return await get_ingest_group_meter_controller(brain_id, group_tracking)



@meta_router.get("/ingestion-capabilities", response_model=IngestionCapabilitiesResponse)
async def ingestion_capabilities():
    import os
    from src.services.data.main import data_adapter
    from src.core.saving.pipeline_store import supports_storage
    return {"progressive": os.getenv("PROGRESSIVE_INGESTION_ENABLED") == "1" and supports_storage(data_adapter.data),
        "version":"sections-v1","concurrency":2}



@meta_router.post("/runtime-models", include_in_schema=False)
async def runtime_models(request: Request, brain_id: str = Depends(get_brain_id)):
    from fastapi import HTTPException
    from src.core.saving.ingest_manifest import verify_control
    from src.core.saving.pipeline_runtime import configured_model
    body = await request.json()
    if not verify_control(brain_id,"runtime","models",body.get("token")):
        raise HTTPException(403,"Platform administration required")
    defaults = {}
    for role in ("small","large"):
        provider, model = configured_model(role)
        defaults[role] = {"provider":provider,"model":model}
    return {"defaults":defaults}

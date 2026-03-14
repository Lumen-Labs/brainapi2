"""
File: /model.py
Created Date: Saturday December 27th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Sunday January 12th 2026
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from fastapi import APIRouter, Depends

from src.services.api.dependencies import get_brain_id
from src.services.api.constants.requests import (
    AddEntityRequest,
    UpdateEntityRequest,
    AddRelationshipRequest,
    UpdateRelationshipRequest,
)
from src.services.api.controllers.model import (
    add_nodes as add_nodes_controller,
    update_node as update_node_controller,
    add_relationship as add_relationship_controller,
    update_relationship as update_relationship_controller,
)

model_router = APIRouter(prefix="/model", tags=["model"])


@model_router.post(path="/entity")
async def add_entity(
    request: AddEntityRequest,
    brain_id: str = Depends(get_brain_id),
):
    """
    Create a new entity (node) in the knowledge graph.
    
    Returns:
    	Response payload containing details of the created node.
    """
    request.brain_id = brain_id
    properties = request.properties or {}
    
    return await add_nodes_controller(
        nodes=[{
            "name": request.name,
            "labels": request.labels,
            "description": request.description,
            "properties": properties,
        }],
        brain_id=request.brain_id,
        identification_params=request.identification_params,
        metadata=request.metadata,
    )


@model_router.put(path="/entity")
async def update_entity(
    request: UpdateEntityRequest,
    brain_id: str = Depends(get_brain_id),
):
    """
    Update an entity (node) in the graph.
    
    @returns The controller's response containing the updated node representation or an operation result.
    """
    request.brain_id = brain_id
    return await update_node_controller(
        uuid=request.uuid,
        brain_id=request.brain_id,
        new_name=request.new_name,
        new_description=request.new_description,
        new_labels=request.new_labels,
        new_properties=request.new_properties,
        properties_to_remove=request.properties_to_remove,
    )


@model_router.post(path="/relationship")
async def add_relationship(
    request: AddRelationshipRequest,
    brain_id: str = Depends(get_brain_id),
):
    """
    Create a relationship linking two nodes in the graph.
    
    Returns:
        relationship (dict): Representation of the created relationship.
    """
    request.brain_id = brain_id
    return await add_relationship_controller(
        subject_uuid=request.subject_uuid,
        predicate_name=request.predicate_name,
        predicate_description=request.predicate_description,
        object_uuid=request.object_uuid,
        brain_id=request.brain_id,
    )


@model_router.put(path="/relationship")
async def update_relationship(
    request: UpdateRelationshipRequest,
    brain_id: str = Depends(get_brain_id),
):
    """
    Update properties of an existing relationship.
    
    Uses the following fields from `request`: `uuid`, `brain_id`, `new_properties` (defaults to an empty dict if missing), and `properties_to_remove` (defaults to an empty list if missing).
    
    Returns:
        The updated relationship representation or a result object describing the outcome of the update.
    """
    request.brain_id = brain_id
    return await update_relationship_controller(
        uuid=request.uuid,
        brain_id=request.brain_id,
        new_properties=request.new_properties or {},
        properties_to_remove=request.properties_to_remove or [],
    )

"""
File: /model.py
Created Date: Saturday December 27th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 27th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from fastapi import APIRouter

from src.services.api.constants.requests import AddEntityRequest, UpdateEntityRequest, AddRelationshipRequest, UpdateRelationshipRequest
from src.services.api.controllers.model import (
    add_entity as add_entity_controller,
    update_entity as update_entity_controller,
    add_relationship as add_relationship_controller,
    update_relationship as update_relationship_controller,
)

model_router = APIRouter(prefix="/model", tags=["model"])


@model_router.post(path="/entity")
async def add_entity(request: AddEntityRequest):
    """
    Add a single entity (node) to the graph.
    """
    return await add_entity_controller(
        name=request.name,
        brain_id=request.brain_id,
        labels=request.labels,
        description=request.description,
        properties=request.properties,
        identification_params=request.identification_params,
        metadata=request.metadata,
    )


@model_router.put(path="/entity")
async def update_entity(request: UpdateEntityRequest):
    """
    Update an entity (node) in the graph.
    """
    return await update_entity_controller(
        uuid=request.uuid,
        brain_id=request.brain_id,
        new_name=request.new_name,
        new_description=request.new_description,
        new_labels=request.new_labels,
        new_properties=request.new_properties,
        properties_to_remove=request.properties_to_remove,
    )


@model_router.post(path="/relationship")
async def add_relationship(request: AddRelationshipRequest):
    """
    Add a relationship between two nodes in the graph.
    """
    return await add_relationship_controller(
        subject_uuid=request.subject_uuid,
        predicate_name=request.predicate_name,
        predicate_description=request.predicate_description,
        object_uuid=request.object_uuid,
        brain_id=request.brain_id,
    )

@model_router.put(path="/relationship")
async def update_relationship(request: UpdateRelationshipRequest):
    """
    Update a relationship's properties in the graph.
    """
    return await update_relationship_controller(
        uuid=request.uuid,
        brain_id=request.brain_id,
        new_properties=request.new_properties or {},
        properties_to_remove=request.properties_to_remove or [],
    )
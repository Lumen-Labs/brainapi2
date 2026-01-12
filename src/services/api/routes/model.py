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

@model_router.post(path="/entity")
async def add_entity(request: AddEntityRequest):
    """
    Create a new entity (node) in the knowledge graph.
    
    Returns:
    	Response payload containing details of the created node.
    """
    return await add_nodes_controller(
        nodes=[{
            "name": request.name,
            "labels": request.labels,
            "description": request.description,
            "properties": request.properties or {},
        }],
        brain_id=request.brain_id,
        identification_params=request.identification_params,
        metadata=request.metadata,
    )

@model_router.put(path="/entity")
async def update_entity(request: UpdateEntityRequest):
    """
    Update an entity (node) in the graph.
    
    @returns The controller's response containing the updated node representation or an operation result.
    """
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
async def add_relationship(request: AddRelationshipRequest):
    """
    Create a relationship linking two nodes in the graph.
    
    Returns:
        relationship (dict): Representation of the created relationship.
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
    Update properties of an existing relationship.
    
    Uses the following fields from `request`: `uuid`, `brain_id`, `new_properties` (defaults to an empty dict if missing), and `properties_to_remove` (defaults to an empty list if missing).
    
    Returns:
        The updated relationship representation or a result object describing the outcome of the update.
    """
    return await update_relationship_controller(
        uuid=request.uuid,
        brain_id=request.brain_id,
        new_properties=request.new_properties or {},
        properties_to_remove=request.properties_to_remove or [],
    )
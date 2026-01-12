"""
File: /meta.py
Created Date: Friday November 7th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 27th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from fastapi import APIRouter

from src.services.api.controllers.meta import (
    get_entities_labels as get_entities_labels_controller,
    get_relationships_properties as get_relationships_properties_controller,
    get_entity_properties as get_entity_properties_controller,
)

meta_router = APIRouter(prefix="/meta", tags=["meta"])

@meta_router.get(path="/relationships-properties")
async def get_relationships_properties(
    brain_id: str = "default",
):
    """
    Retrieve all unique relationship types present in the graph for the specified brain.
    
    Parameters:
        brain_id (str): Identifier of the brain/graph to query. Defaults to "default".
    
    Returns:
        A list of relationship type names (strings) found in the specified graph.
    """
    return await get_relationships_properties_controller(brain_id)

@meta_router.get(path="/entity-labels")
async def get_entities_labels(
    brain_id: str = "default",
):
    """
    Retrieve all unique node labels present in the graph.
    
    Parameters:
        brain_id (str): Identifier of the brain/graph to query. Defaults to "default".
    
    Returns:
        labels (list[str]): List of unique node label names.
    """
    return await get_entities_labels_controller(brain_id)

@meta_router.get(path="/entity-properties")
async def get_entity_properties(
    brain_id: str = "default",
):
    """
    Retrieve all unique property keys used by entities in the specified brain/graph.
    
    Parameters:
        brain_id (str): Identifier of the brain/graph to query; defaults to "default".
    
    Returns:
        list[str]: A list of property key names present on entities in the graph.
    """
    return await get_entity_properties_controller(brain_id)
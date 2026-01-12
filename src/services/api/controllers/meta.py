"""
File: /meta.py
Created Date: Friday November 7th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 27th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import asyncio
from src.services.kg_agent.main import graph_adapter


async def get_entities_labels(brain_id: str):
    """
    Retrieve node labels (types) for the graph identified by `brain_id`.
    
    Parameters:
        brain_id (str): Identifier of the brain/graph to query.
    
    Returns:
        list[str]: Node label/type names present in the graph.
    """
    result = await asyncio.to_thread(graph_adapter.get_graph_node_types, brain_id)
    return result

async def get_relationships_properties(brain_id: str):
    """
    Retrieve relationship types and their property keys for the graph associated with the given brain.
    
    Parameters:
        brain_id (str): Identifier of the brain whose graph to query.
    
    Returns:
        relationship_properties: A collection mapping relationship types to their associated property keys.
    """
    result = await asyncio.to_thread(graph_adapter.get_graph_relationship_types, brain_id)
    return result

async def get_entity_properties(brain_id: str):
    """
    Retrieve all unique property keys for node entities in the specified graph.
    
    Parameters:
        brain_id (str): Identifier of the brain/graph to query.
    
    Returns:
        list[str]: Unique property key names present on nodes in the graph.
    """
    result = await asyncio.to_thread(graph_adapter.get_graph_node_properties, brain_id)
    return result
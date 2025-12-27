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
    Get the entity properties of the graph.
    """
    result = await asyncio.to_thread(graph_adapter.get_graph_node_types, brain_id)
    return result

async def get_relationships_properties(brain_id: str):
    """
    Get the entity properties of the graph.
    """
    result = await asyncio.to_thread(graph_adapter.get_graph_relationship_types, brain_id)
    return result

async def get_entity_properties(brain_id: str):
    """
    Get all unique property keys from entities in the graph.
    """
    result = await asyncio.to_thread(graph_adapter.get_graph_node_properties, brain_id)
    return result
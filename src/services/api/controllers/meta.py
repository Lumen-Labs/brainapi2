"""
File: /meta.py
Created Date: Friday November 7th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday November 7th 2025 9:59:06 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import asyncio
from src.services.kg_agent.main import graph_adapter


async def get_entity_properties():
    """
    Get the entity properties of the graph.
    """
    result = await asyncio.to_thread(graph_adapter.get_graph_property_keys)
    return result

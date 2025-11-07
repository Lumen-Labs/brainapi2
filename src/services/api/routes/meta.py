"""
File: /meta.py
Created Date: Friday November 7th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Friday November 7th 2025 9:58:46 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

from fastapi import APIRouter

from src.services.api.controllers.meta import (
    get_entity_properties as get_entity_properties_controller,
)

meta_router = APIRouter(prefix="/meta", tags=["meta"])


@meta_router.get(path="/entity-properties")
async def get_entity_properties():
    """
    Get the entity properties of the graph.
    """
    return await get_entity_properties_controller()

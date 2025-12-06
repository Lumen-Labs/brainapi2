"""
File: /system.py
Created Date: Monday December 1st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday December 1st 2025 10:13:27 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import asyncio

from pydantic import BaseModel
from src.services.data.main import data_adapter


async def get_brains_list():
    """
    Get the list of brains.
    """
    result = await asyncio.to_thread(data_adapter.get_brains_list)
    return result


class CreateBrainRequest(BaseModel):
    """
    Request body for the create brain endpoint.
    """

    brain_id: str


async def create_new_brain(request: CreateBrainRequest):
    """
    Create a new brain
    """
    result = await asyncio.to_thread(data_adapter.create_brain, request.brain_id)
    return result

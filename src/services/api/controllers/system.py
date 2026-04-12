"""
File: /system.py
Created Date: Monday December 1st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday December 1st 2025 10:13:27 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

import asyncio

from pydantic import BaseModel
from src.services.api.constants.requests import CreateBrainRequest
from src.services.data.main import data_adapter


async def get_brains_list():
    """
    Get the list of brains.
    """
    result = await asyncio.to_thread(data_adapter.get_brains_list)
    return result


async def create_new_brain(request: CreateBrainRequest):
    """
    Create a new brain
    """
    if not (request.brain_id.isalnum() and request.brain_id[0].isalpha()):
        raise ValueError("brain_id must be alphanumeric and start with a letter")
    result = await asyncio.to_thread(data_adapter.create_brain, request.brain_id)
    return result

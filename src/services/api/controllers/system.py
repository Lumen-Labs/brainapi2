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
from src.services.data.main import data_adapter


async def get_brains_list():
    """
    Get the list of brains.
    """
    result = await asyncio.to_thread(data_adapter.get_brains_list)
    return result


async def create_new_brain(brain_id: str):
    """
    Create a new brain
    """
    result = await asyncio.to_thread(data_adapter.create_brain, brain_id)
    return result

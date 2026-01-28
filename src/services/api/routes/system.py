"""
File: /system.py
Created Date: Monday December 1st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday December 1st 2025 10:12:48 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from fastapi import APIRouter
from src.services.api.constants.requests import CreateBrainRequest
from src.services.api.controllers.system import (
    get_brains_list as get_brains_list_controller,
    create_new_brain as create_new_brain_controller,
)

system_router = APIRouter(prefix="/system", tags=["system"])


@system_router.get(path="/brains-list")
async def get_brains_list():
    """
    Get the list of brains.
    """
    return await get_brains_list_controller()


@system_router.post(path="/brains")
async def create_brain(request: CreateBrainRequest):
    """
    Create a new brain
    """
    return await create_new_brain_controller(request)


@system_router.get(path="/brains/{brain_id}/reset")  # TODO
async def reset(brain_id: str):
    """
    Resets the brain
    """
    raise NotImplementedError("Not implemented")


@system_router.get(path="/brains/{brain_id}/delete")  # TODO
async def delete(brain_id: str):
    """
    Deletes the brain and all its data
    """
    raise NotImplementedError("Not implemented")


@system_router.post(path="/brains/{brain_id}/create-backup")  # TODO
async def create_backup(brain_id: str):
    """
    Creates a backup of the brain and returns a task ID
    """
    raise NotImplementedError("Not implemented")

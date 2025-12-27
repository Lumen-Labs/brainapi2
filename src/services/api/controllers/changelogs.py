"""
File: /changelogs.py
Created Date: Saturday December 27th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 27th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""
import asyncio
from typing import Optional
from fastapi import HTTPException
from starlette.responses import JSONResponse
from src.services.data.main import data_adapter

async def get_changelog_by_id(
    id: str, brain_id: str = "default"
):
    """
    Get changelog by ID.
    """
    def _get_changelog():
        changelog = data_adapter.get_changelog_by_id(id, brain_id)
        if not changelog:
            raise HTTPException(status_code=404, detail="Changelog not found")
        return changelog
    
    result = await asyncio.to_thread(_get_changelog)
    
    return JSONResponse(
        content={
            "message": "Changelog retrieved successfully",
            "changelog": result.model_dump(mode="json")
        }
    )

async def get_changelogs_list(
    limit: int = 10,
    skip: int = 0,
    types: Optional[list[str]] = None,
    query_text: Optional[str] = None,
    brain_id: str = "default",
):
    """
    Get a list of changelogs.
    """
    def _get_list():
        return data_adapter.get_changelogs_list(
            brain_id, limit, skip, types, query_text
        )
    
    results = await asyncio.to_thread(_get_list)
    
    return JSONResponse(
        content={
            "message": "Changelogs list retrieved successfully",
            "changelogs": [r.model_dump(mode="json") for r in results],
            "count": len(results)
        }
    )

async def get_changelog_types(
    brain_id: str = "default",
):
    """
    Get all unique types from changelogs.
    """
    def _get_types():
        return data_adapter.get_changelog_types(brain_id)
    
    results = await asyncio.to_thread(_get_types)
    
    return JSONResponse(
        content={
            "message": "Changelog types retrieved successfully",
            "types": results,
            "count": len(results)
        }
    )
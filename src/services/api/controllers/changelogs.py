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
    Retrieve a changelog by its ID for a specified brain.
    
    Parameters:
        id (str): Identifier of the changelog to retrieve.
        brain_id (str): Identifier of the brain to query; defaults to "default".
    
    Returns:
        JSONResponse: Response with a payload containing:
            - message (str): "Changelog retrieved successfully"
            - changelog (dict): The changelog serialized as JSON.
    
    Raises:
        HTTPException: Raised with status code 404 if the changelog is not found.
    """
    def _get_changelog():
        """
        Retrieve the changelog with the given ID for the current brain, raising a 404 error if it does not exist.
        
        Raises:
            HTTPException: with status 404 and detail "Changelog not found" when no changelog is found.
        
        Returns:
            The retrieved changelog model instance.
        """
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
    Retrieve a paginated, optionally filtered list of changelogs for a brain.
    
    Parameters:
    	limit (int): Maximum number of changelogs to return.
    	skip (int): Number of changelogs to skip (offset).
    	types (Optional[list[str]]): If provided, restrict results to these changelog types.
    	query_text (Optional[str]): If provided, filter changelogs by matching text.
    	brain_id (str): Identifier of the brain from which to fetch changelogs.
    
    Returns:
    	JSONResponse: Response content contains:
    		- message: Success message string.
    		- changelogs: List of changelog objects serialized as JSON.
    		- count: Integer count of returned changelogs.
    """
    def _get_list():
        """
        Retrieve a list of changelog entries using the surrounding scope's filters and pagination.
        
        Returns:
            list: Changelog objects matching the provided brain_id, limit, skip, types, and query_text.
        """
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
    Retrieve all unique changelog types for a given brain.
    
    Parameters:
        brain_id (str): Identifier of the brain to query; defaults to "default".
    
    Returns:
        JSONResponse: Response content contains:
            - message (str): Success message.
            - types (list[str]): List of unique changelog types.
            - count (int): Number of types returned.
    """
    def _get_types():
        """
        Retrieve the list of unique changelog types for the current brain.
        
        Returns:
            list[str]: A list of changelog type names.
        """
        return data_adapter.get_changelog_types(brain_id)
    
    results = await asyncio.to_thread(_get_types)
    
    return JSONResponse(
        content={
            "message": "Changelog types retrieved successfully",
            "types": results,
            "count": len(results)
        }
    )